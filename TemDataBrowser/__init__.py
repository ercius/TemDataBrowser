from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("TemDataBrowser")
except PackageNotFoundError:
    __version__ = "0.0.0"

import argparse
import functools
import json
import re
import sys
import threading

from ScopeFoundry import BaseApp
from ScopeFoundry.helper_funcs import load_qt_ui_from_pkg
from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
from qtpy import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np

import imageio.v3 as iio
import ncempy

# Use row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major')

# Every line of the FEI tomography-parameter log is prefixed with a fixed-width
# "MM/DD/YY HH:MM:SS " timestamp; what follows it is indented to show which section a
# parameter belongs to.
_TIMESTAMP_RE = re.compile(r'^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ')
_VALUE_KEY = '_value'


def _parse_fei_value(raw):
    raw = raw.strip()
    if raw == '':
        return None
    if raw in ('Yes', 'ON'):
        return True
    if raw in ('No', 'OFF'):
        return False
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_fei_parameters(lines):
    """Parse the vendor tomography-parameter log into a nested dict.

    Section headers (e.g. "STEM imaging mode", "Check Focus") repeat parameter names
    like "Periodicity (high tilt range)" under different settings, so a flat dict would
    have later sections silently overwrite earlier ones. Indentation depth tells sections
    apart from their children, so it is used to nest rather than flatten them.

    A line can be a leaf, a header with no value of its own ("STEM imaging mode"), or
    both at once ("Check Focus: Yes" has its own value and also has Periodicity settings
    indented beneath it) -- every line is therefore pushed as a potential parent, and
    _collapse resolves what it actually turned out to be once all its children are known.
    """
    root = {}
    stack = [(-1, root)]
    for raw_line in lines:
        line = _TIMESTAMP_RE.sub('', raw_line)
        stripped = line.strip()
        if not stripped:
            continue

        # The stack must unwind to this line's depth before deciding whether to skip it,
        # or a skipped section header (e.g. a "-----" rule right after a depth-1 line)
        # would leave a stale frame on the stack and misparent everything that follows.
        depth = len(line) - len(line.lstrip(' '))
        while stack[-1][0] >= depth:
            stack.pop()

        if set(stripped) == {'-'}:
            continue  # decorative rule; never has children of its own
        parent = stack[-1][1]

        if ':' in stripped:
            key, _, value = stripped.partition(':')
            key, value = key.strip(), _parse_fei_value(value)
        else:
            key, value = stripped, None

        node = {_VALUE_KEY: value}
        parent[key] = node
        stack.append((depth, node))

    _collapse(root)
    return root


def _collapse(node):
    """Resolve each {_value, ...children} node into its final shape.

    No children and no value -> True (a bare flag like "STEM imaging mode" turned out
    to introduce no sub-parameters). No children, a value -> that value. Children and no
    value -> a dict of just the children. Both -> a dict of the children plus 'value'.
    """
    for key, child in node.items():
        value = child.pop(_VALUE_KEY)
        _collapse(child)
        if not child:
            node[key] = value if value is not None else True
        elif value is not None:
            child['value'] = value
            node[key] = child
        else:
            node[key] = child


class _AsyncFileLoader(QtCore.QObject):
    """Runs blocking file reads on a background thread and delivers results on the GUI thread.

    A monotonic generation token cancels stale loads when the user switches files quickly
    so only the most recently requested file updates the view.
    """

    _loaded = QtCore.Signal(int, str, object, object)  # gen, fname, result, error

    def __init__(self, on_success, on_error, parent=None):
        super().__init__(parent)
        self._on_success = on_success
        self._on_error = on_error
        self._gen = 0
        self._closed = False
        self._loaded.connect(self._deliver, QtCore.Qt.ConnectionType.QueuedConnection)

    def close(self):
        self._closed = True
        self._gen += 1

    def load(self, fname, reader_func):
        self._gen += 1
        gen = self._gen
        fname_str = str(fname)

        def _worker():
            try:
                result = reader_func(fname_str)
                err = None
            except Exception as exc:
                result = None
                err = exc
            if not self._closed and gen == self._gen:
                self._loaded.emit(gen, fname_str, result, err)

        threading.Thread(target=_worker, daemon=True).start()
        return gen

    @QtCore.Slot(int, str, object, object)
    def _deliver(self, gen, fname, result, error):
        if self._closed or gen != self._gen:
            return
        if error is not None:
            self._on_error(fname, error)
        else:
            self._on_success(fname, result)


def _read_tem_file(fname):
    """Read and normalize a TEM file in a background thread (no Qt calls)."""
    is_stemtomo = False
    if Path(fname).suffix.lower() == '.emd':
        # Check for special STEMTomo7 Berkeley EMD files
        try:
            with ncempy.io.emd.fileEMD(fname) as f0:
                if 'data' in f0.file_hdl:
                    if 'stemtomo version' in f0.file_hdl['data'].attrs:
                        is_stemtomo = True
        except ncempy.io.emd.NoEmdDataSets:
            # Velox files throw this error, they are not Berkeley EMD files
            is_stemtomo = False

    file = ncempy.read(fname)

    # Remove singular dimensions
    data = np.squeeze(file['data'])

    # Test for > 3D data and reduce if possible
    if data.ndim == 4 and is_stemtomo:
        print('Warning: only showing 1 image per tilt angle for STEMTomo7 data.')
        data = data[:, 0, :, :]
    elif data.ndim == 4:
        print(f'Warning: Reducing {data.ndim}-D data to 3-D.')
        data = data[0, :, :, :]
    elif data.ndim > 4:
        print(f'{data.ndim}-D data files are not supported.')

    xscale = file['pixelSize'][-2]
    yscale = file['pixelSize'][-1]
    pixel_unit = file['pixelUnit'][-1]
    return data, xscale, yscale, pixel_unit


class imageioView(DataBrowserView):
    """ Handles most normal image types like TIF, PNG, etc."""

    # This name is used in the GUI for the DataBrowser
    name = 'Image viewer (imageio)'

    def __init__(self, databrowser):
        super().__init__(databrowser)
        self.data = None
        self._loader = _AsyncFileLoader(
            self._on_load_success,
            self._on_load_error,
            parent=self.q_object,
        )
        if hasattr(self.databrowser, 'qtapp') and self.databrowser.qtapp is not None:
            self.databrowser.qtapp.aboutToQuit.connect(self._loader.close)

    def setup(self):
        # create the GUI and viewer settings, runs once at program start up
        # self.ui should be a QWidget of some sort, here we use a pyqtgraph ImageView
        self.ui = self.imview = pg.ImageView()

    def is_file_supported(self, fname):
        # Tells the DataBrowser whether this plug-in would likely be able
        # to read the given file name
        # here we are using the file extension to make a guess
        ext = Path(fname).suffix
        return ext.lower() in ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.webp']

    def on_change_data_filename(self, fname):
        # A new file has been selected by the user, load asynchronously and display it
        self.databrowser.ui.statusbar.showMessage(f'Loading {Path(fname).name}...')
        self._loader.load(fname, iio.imread)

    def _on_load_success(self, fname, data):
        self.data = np.asarray(data)
        if self.data.ndim == 3 and self.data.shape[-1] in (3, 4):
            self.imview.setImage(self.data, axes={'y': 0, 'x': 1, 'c': 2})
        elif self.data.ndim == 3:
            self.imview.setImage(self.data, axes={'t': 0, 'y': 1, 'x': 2})
        else:
            self.imview.setImage(self.data)
        self.databrowser.ui.statusbar.showMessage(f'Loaded {Path(fname).name}', 3000)

    def _on_load_error(self, fname, err):
        # When a failure to load occurs, zero out image and show error message
        self.imview.setImage(np.zeros((10, 10)))
        self.databrowser.ui.statusbar.showMessage(
            "failed to load %s:\n%s" % (fname, err)
        )


class TemView(DataBrowserView):
    """ Data browser for common S/TEM file types """

    # This name is used in the GUI for the DataBrowser
    name = 'TEM data viewer'

    def __init__(self, databrowser):
        super().__init__(databrowser)
        self.data = None
        self._loader = _AsyncFileLoader(
            self._on_load_success,
            self._on_load_error,
            parent=self.q_object,
        )
        if hasattr(self.databrowser, 'qtapp') and self.databrowser.qtapp is not None:
            self.databrowser.qtapp.aboutToQuit.connect(self._loader.close)

    def setup(self):
        """ create the GUI and viewer settings, runs once at program start up
            self.ui should be a QWidget of some sort, here we use a pyqtgraph ImageView
        """
        self.plt = pg.PlotItem(labels={'bottom': ('X', ''), 'left': ('Y', '')})
        self.ui = self.imview = pg.ImageView(view=self.plt)
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()

    def is_file_supported(self, fname):
        """ Tells the DataBrowser whether this plug-in would likely be able
        to read the given file name. Here we are using the file extension
        to make a guess
        """
        ext = Path(fname).suffix.lower()
        return ext in ['.dm3', '.dm4', '.mrc', '.ali', '.rec', '.emd', '.ser', '.img']

    def on_change_data_filename(self, fname):
        """ A new file has been selected by the user, load asynchronously and display it """
        print(f'Loading {fname}...')
        self.databrowser.ui.statusbar.showMessage(f'Loading {Path(fname).name}...')
        self._loader.load(fname, _read_tem_file)

    def _on_load_success(self, fname, payload):
        data, xscale, yscale, pixel_unit = payload
        self.data = data
        self.imview.setImage(self.data)
        img = self.imview.getImageItem()

        if pixel_unit in ('um', 'µm', '[u_m]', 'u_m'):
            unit_scale = 1e-6
            unit = 'm'
        elif pixel_unit in ('m',):
            unit_scale = 1
            unit = 'm'
        elif pixel_unit in ('nm', '[n_m]', 'n_m'):
            unit_scale = 1e-9
            unit = 'm'
        elif pixel_unit in ('A', 'Ang', 'Å'):
            unit_scale = 1e-10
            unit = 'm'
        else:
            unit_scale = 1
            xscale = 1
            yscale = 1
            unit = 'pixels'
        tr = QtGui.QTransform()
        img.setTransform(tr.scale(xscale * unit_scale, yscale * unit_scale))

        # Set the labels
        p1_bottom = self.plt.getAxis('bottom')
        p1_bottom.setLabel('X', units=unit)
        p1_left = self.plt.getAxis('left')
        p1_left.setLabel('Y', units=unit)

        self.plt.autoRange()
        self.databrowser.ui.statusbar.showMessage(f'Loaded {Path(fname).name}', 3000)

    def _on_load_error(self, fname, err):
        # When a failure to load occurs, zero out image and show error message
        self.imview.setImage(np.zeros((10, 10)))
        self.databrowser.ui.statusbar.showMessage(
            f'failed to load {fname}:\n{err}'
        )


class TemMetadataView(DataBrowserView):
    """ A viewer to read meta data from a file and display it as text. """

    name = 'TEM metadata viewer'

    def setup(self):
        self.ui = QtWidgets.QTextEdit("File metadata")

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_dm_metadata(fname):
        """ Reads important metadata from DM files"""
        with ncempy.io.dm.fileDM(fname) as f:
            meta_data = f.getMetadata(index=0)

        return meta_data

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_mrc_metadata(path):
        """ Reads important metadata from MRC and related files."""
        with ncempy.io.mrc.fileMRC(path) as f:
            meta_data = f.getMetadata()

        # Read tilt angles from .rawtlt file if it exists
        rawtltName = Path(path).with_suffix('.rawtlt')
        if rawtltName.exists():
            with open(rawtltName, 'r') as f1:
                tilts = list(map(float, f1))
            meta_data['tilt angles'] = tilts

        # Read FEI parameters from .txt file if it exists
        FEIparameters = Path(path).with_suffix('.txt')
        if FEIparameters.exists():
            try:
                with open(FEIparameters, 'r', encoding='utf-8-sig') as f2:
                    lines = f2.readlines()
            except UnicodeDecodeError:
                with open(FEIparameters, 'r', encoding='cp1252') as f2:
                    lines = f2.readlines()
            meta_data['fei_parameters'] = _parse_fei_parameters(lines)

        return meta_data

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_emd_metadata(path):
        """ Reads important metadata from EMD Berkeley files."""
        with ncempy.io.emd.fileEMD(path) as f:
            meta_data = f.getMetadata(0)  # 0 = index into f.list_data
        return meta_data

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_velox_metadata(path):
        """ Reads important metadata from Velox EMD files."""
        with ncempy.io.emdVelox.fileEMDVelox(path) as f:
            if f.list_data is None:
                return None
            meta_data = f.getMetadata(0)  # 0 = index into f.list_data
        return meta_data

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_ser_metadata(path):
        with ncempy.io.ser.fileSER(path) as f:
            # Returns global metadata for the whole file
            meta_data = f.getMetadata()

            # Get additional metadata from the first dataset
            _, extra_metadata = f.getDataset(0)
            meta_data.update(extra_metadata)

            # Get the header information
            meta_data.update(f.head)

            # Clean the dictionary
            for k, v in meta_data.items():
                if isinstance(v, bytes):
                    meta_data[k] = v.decode('UTF8')

            return meta_data

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_emi_metadata(fname):
        return ncempy.io.ser.read_emi(fname)

    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_img_metadata(fname):
        with ncempy.io.smv.fileSMV(fname) as f0:
            meta_data = f0.getMetadata()
        return meta_data

    def on_change_data_filename(self, fname):
        ext = Path(fname).suffix.lower()

        meta_data = {'file name': str(fname)}
        if ext in ('.dm3', '.dm4'):
            meta_data = self.get_dm_metadata(fname)
        elif ext in ('.mrc', '.rec', '.ali'):
            meta_data = self.get_mrc_metadata(fname)
        elif ext in ('.emd',):
            try:
                meta_data = self.get_emd_metadata(fname)
            except ncempy.io.emd.NoEmdDataSets:
                velox_meta = self.get_velox_metadata(fname)
                if velox_meta is not None:
                    meta_data = velox_meta
                else:
                    meta_data = {'file name': str(fname), 'error': 'No EMD datasets found'}
        elif ext in ('.ser',):
            meta_data = self.get_ser_metadata(fname)
        elif ext in ('.emi',):
            meta_data = self.get_emi_metadata(fname)
        elif ext in ('.img',):
            meta_data = self.get_img_metadata(fname)

        txt = f'file name = {fname}\n'
        for k, v in meta_data.items():
            if isinstance(v, dict):
                v = json.dumps(v, indent=2, default=str)
            line = f'{k} = {v}\n'
            txt += line
        self.ui.setText(txt)

    def is_file_supported(self, fname):
        ext = Path(fname).suffix
        return ext.lower() in ('.dm3', '.dm4', '.mrc', '.ali', '.rec', '.ser', '.emi', '.img', '.emd')


class TemDataBrowserApp(DataBrowser):
    """DataBrowser subclass that avoids duplicate file loads on view switch."""

    def on_change_data_filename(self):
        fname = self.settings["data_filename"]
        if not Path(fname).is_file():
            print("invalid data_filename", fname)
            return

        print("file", fname)

        if self.settings["auto_select_view"]:
            view_name = self.auto_select_view(fname)
            if view_name != self.current_view.name:
                prev_view_setting = self.settings["view_name"]
                self.settings["view_name"] = view_name
                if prev_view_setting != view_name:
                    # on_change_view_name() already ran setup_view() and loaded fname
                    return
                self.current_view = self.views[view_name]
                self.setup_view(self.current_view)

        self.current_view.on_change_data_filename(fname)


def parse_cli_args(argv=None):
    """Parse optional positional startup path without clobbering DataBrowser flags."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        'path',
        nargs='?',
        default=None,
        help='Optional directory to browse or file to open on startup',
    )
    raw = sys.argv[1:] if argv is None else list(argv)
    return parser.parse_known_args(raw)


def _has_flag(args, flag_name):
    prefix = f'{flag_name}='
    return any(a == flag_name or a.startswith(prefix) for a in args)


def create_app(argv=None, qsettings=None):
    """Create and configure the TemDataBrowser application instance."""
    cli_args, remaining = parse_cli_args(argv)

    # DataBrowser.__init__ parses sys.argv[1:] directly via argparse
    orig_argv = sys.argv
    try:
        sys.argv = [orig_argv[0], *remaining]
        app = TemDataBrowserApp(sys.argv)
    finally:
        sys.argv = orig_argv

    app.ui.setWindowTitle("TemDataBrowser")

    # Load views here (last loaded is the first one tried by auto_select_view)
    app.add_view(TemMetadataView(app))
    app.add_view(imageioView(app))
    app.add_view(TemView(app))

    if qsettings is None:
        qsettings = QtCore.QSettings("TemDataBrowser", "TemDataBrowser")
    app._qsettings = qsettings

    # Restore saved window geometry if available
    saved_geom = qsettings.value("window_geometry")
    if saved_geom is not None:
        app.ui.restoreGeometry(saved_geom)

    # Resolve startup browse_dir and optional startup file
    startup_file = None
    explicit_browse_dir = _has_flag(remaining, '--browse_dir')
    if cli_args.path is not None:
        target = Path(cli_args.path).expanduser().resolve()
        if target.is_dir():
            app.settings['browse_dir'] = str(target)
        elif target.is_file():
            app.settings['browse_dir'] = str(target.parent)
            startup_file = str(target)
        else:
            app.ui.statusbar.showMessage(f'Path not found: {target}', 5000)
    elif not explicit_browse_dir:
        saved_dir = qsettings.value("browse_dir")
        if saved_dir and Path(str(saved_dir)).is_dir():
            app.settings['browse_dir'] = str(saved_dir)
        else:
            app.settings['browse_dir'] = str(Path.home())

    def _save_ui_state():
        browse_dir = str(app.settings['browse_dir'])
        if browse_dir and Path(browse_dir).is_dir():
            qsettings.setValue("browse_dir", browse_dir)
        if app.ui is not None:
            qsettings.setValue("window_geometry", app.ui.saveGeometry())
        qsettings.sync()

    app.save_ui_state = _save_ui_state
    app.qtapp.aboutToQuit.connect(_save_ui_state)

    if startup_file is not None:
        app.settings['data_filename'] = startup_file

    return app


def open_file():
    """Start the graphical user interface from inside a python interpreter."""
    main()


def main(argv=None):
    """ This starts the graphical user interface and loads the views."""
    app = create_app(argv)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
