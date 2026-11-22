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

_SPATIAL_UNIT_SCALES = {
    'm': (1.0, 'm', 'm'),
    'mm': (1e-3, 'm', 'mm'),
    'um': (1e-6, 'm', 'um'),
    'µm': (1e-6, 'm', 'um'),
    'μm': (1e-6, 'm', 'um'),
    '[u_m]': (1e-6, 'm', 'um'),
    'u_m': (1e-6, 'm', 'um'),
    'nm': (1e-9, 'm', 'nm'),
    '[n_m]': (1e-9, 'm', 'nm'),
    'n_m': (1e-9, 'm', 'nm'),
    'A': (1e-10, 'm', 'A'),
    'Ang': (1e-10, 'm', 'A'),
    'Å': (1e-10, 'm', 'A'),
    'pm': (1e-12, 'm', 'pm'),
}


def _apply_image_calibration(imview, plt, pixel_size=None, pixel_unit=None, allow_custom_unit=False):
    """Apply pixel scale transform and axis labels to a PyQtGraph ImageView + PlotItem.

    Returns ((xscale, yscale), reported_unit) in matching units.
    """
    if pixel_size is None:
        xscale = yscale = 1.0
    elif isinstance(pixel_size, (tuple, list, np.ndarray)):
        if len(pixel_size) == 1:
            xscale = yscale = float(pixel_size[0])
        elif len(pixel_size) >= 2:
            xscale = float(pixel_size[0])
            yscale = float(pixel_size[1])
        else:
            xscale = yscale = 1.0
    else:
        xscale = yscale = float(pixel_size)

    unit_str = str(pixel_unit).strip() if pixel_unit is not None else ''
    if unit_str in _SPATIAL_UNIT_SCALES:
        unit_scale, axis_unit, reported_unit = _SPATIAL_UNIT_SCALES[unit_str]
    elif allow_custom_unit and unit_str:
        unit_scale = 1.0
        axis_unit = unit_str
        reported_unit = unit_str
    else:
        unit_scale = 1.0
        xscale = 1.0
        yscale = 1.0
        axis_unit = 'pixels'
        reported_unit = 'pixels'

    img = imview.getImageItem()
    tr = QtGui.QTransform()
    img.setTransform(tr.scale(xscale * unit_scale, yscale * unit_scale))

    p1_bottom = plt.getAxis('bottom')
    p1_bottom.setLabel('X', units=axis_unit)
    p1_left = plt.getAxis('left')
    p1_left.setLabel('Y', units=axis_unit)

    plt.autoRange()
    return (xscale, yscale), reported_unit


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


class imageioView(DataBrowserView):
    """ Handles most normal image types like TIF, PNG, etc."""

    # This name is used in the GUI for the DataBrowser
    name = 'Image viewer (imageio)'

    def __init__(self, databrowser):
        super().__init__(databrowser)
        self.data = None

    def setup(self):
        # create the GUI and viewer settings, runs once at program start up
        # self.ui should be a QWidget of some sort, here we use a pyqtgraph ImageView
        self.ui = self.imview = pg.ImageView()

    def is_file_supported(self, fname):
        # Tells the DataBrowser whether this plug-in would likely be able
        # to read the given file name
        # here we are using the file extension to make a guess
        ext = Path(fname).suffix
        return ext.lower() in ['.png', '.tif', '.tiff', '.jpg']

    def on_change_data_filename(self, fname):
        # A new file has been selected by the user, load and display it
        try:
            self.data = iio.imread(fname)
            self.imview.setImage(self.data.swapaxes(0, 1))
        except Exception as err:
            # When a failure to load occurs, zero out image
            # and show error message
            self.imview.setImage(np.zeros((10, 10)))
            self.databrowser.ui.statusbar.showMessage(
                "failed to load %s:\n%s" % (fname, err))
            raise err


class TemView(DataBrowserView):
    """ Data browser for common S/TEM file types """

    # This name is used in the GUI for the DataBrowser
    name = 'TEM data viewer'

    def __init__(self, databrowser):
        super().__init__(databrowser)
        self.data = None
        self.pixel_size = None
        self.pixel_unit = None

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
        """ A new file has been selected by the user, load and display it """
        try:
            is_stemtomo = False
            print(f'Loading {fname}...')
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
            self.data = np.squeeze(file['data'])

            # Test for > 3D data and reduce if possible
            if self.data.ndim == 4 and is_stemtomo:
                print('Warning: only showing 1 image per tilt angle for STEMTomo7 data.')
                self.data = self.data[:, 0, :, :]
            elif self.data.ndim == 4:
                print(f'Warning: Reducing {self.data.ndim}-D data to 3-D.')
                self.data = self.data[0, :, :, :]
            elif self.data.ndim > 4:
                print(f'{self.data.ndim}-D data files are not supported.')

            xscale = float(file['pixelSize'][-2])
            yscale = float(file['pixelSize'][-1])
            raw_unit = file['pixelUnit'][-1]
            self.imview.setImage(self.data)
            self.pixel_size, self.pixel_unit = _apply_image_calibration(
                self.imview,
                self.plt,
                pixel_size=(xscale, yscale),
                pixel_unit=raw_unit,
                allow_custom_unit=False,
            )

        except Exception as err:
            # When a failure to load occurs, zero out image
            # and show error message
            self.imview.setImage(np.zeros((10, 10)))
            self.databrowser.ui.statusbar.showMessage(
                f'failed to load {fname}:\n{err}')
            raise err


class ArrayImageView(DataBrowserView):
    """Viewer for in-memory 2D/3D numpy arrays and decoded images (e.g. from MCP tools)."""

    name = 'In-memory image viewer'

    def __init__(self, databrowser):
        super().__init__(databrowser)
        self.data = None
        self.title = ''
        self.pixel_size = None
        self.pixel_unit = None

    def setup(self):
        self.plt = pg.PlotItem(labels={'bottom': ('X', ''), 'left': ('Y', '')})
        self.ui = self.imview = pg.ImageView(view=self.plt)
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()

    def is_file_supported(self, fname):
        # Never hijack normal file clicks in the DataBrowser file tree
        return False

    def set_array(self, data, title='', pixel_size=None, pixel_unit=None):
        """Display a 2D grayscale, 2D RGB/RGBA, or 3D grayscale stack numpy array."""
        arr = np.asarray(data)
        if np.iscomplexobj(arr):
            raise ValueError(
                'Complex arrays cannot be displayed directly; pass magnitude or phase.'
            )

        # Squeeze leading/extra singular dimensions while keeping at least 2D
        if arr.ndim > 3 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4)):
            squeezed = np.squeeze(arr)
            if squeezed.ndim >= 2:
                arr = squeezed

        if arr.ndim not in (2, 3):
            raise ValueError(
                f'Expected a 2D or 3D image array, got shape {arr.shape} ({arr.ndim}D).'
            )

        self.data = arr
        self.title = title or ''

        # Explicitly tell PyQtGraph whether 3D data is color (H, W, 3|4) or stack (N, H, W)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            self.imview.setImage(arr, axes={'y': 0, 'x': 1, 'c': 2})
        elif arr.ndim == 3:
            self.imview.setImage(arr, axes={'t': 0, 'y': 1, 'x': 2})
        else:
            self.imview.setImage(arr, axes={'y': 0, 'x': 1})

        self.pixel_size, self.pixel_unit = _apply_image_calibration(
            self.imview,
            self.plt,
            pixel_size=pixel_size,
            pixel_unit=pixel_unit,
            allow_custom_unit=True,
        )
        self.plt.setTitle(self.title)


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
        meta_data = read_tem_metadata(fname)

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


def read_tem_metadata(fname):
    """Read metadata dictionary for a TEM file using TemMetadataView's cached readers."""
    fname_str = str(fname)
    ext = Path(fname_str).suffix.lower()

    meta_data = {'file name': fname_str}
    if ext in ('.dm3', '.dm4'):
        meta_data = TemMetadataView.get_dm_metadata(fname_str)
    elif ext in ('.mrc', '.rec', '.ali'):
        meta_data = TemMetadataView.get_mrc_metadata(fname_str)
    elif ext in ('.emd',):
        try:
            meta_data = TemMetadataView.get_emd_metadata(fname_str)
        except ncempy.io.emd.NoEmdDataSets:
            velox_meta = TemMetadataView.get_velox_metadata(fname_str)
            if velox_meta is not None:
                meta_data = velox_meta
            else:
                meta_data = {'file name': fname_str, 'error': 'No EMD datasets found'}
    elif ext in ('.ser',):
        meta_data = TemMetadataView.get_ser_metadata(fname_str)
    elif ext in ('.emi',):
        meta_data = TemMetadataView.get_emi_metadata(fname_str)
    elif ext in ('.img',):
        meta_data = TemMetadataView.get_img_metadata(fname_str)
    return dict(meta_data)


def parse_mcp_args(argv=None):
    """Parse TemDataBrowser MCP CLI switches, returning (mcp_args, remaining_argv)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--mcp',
        action='store_true',
        help='Auto-start the embedded MCP HTTP server on launch',
    )
    parser.add_argument(
        '--mcp-host',
        default='127.0.0.1',
        help='Bind host for the MCP HTTP server (default: 127.0.0.1)',
    )
    parser.add_argument(
        '--mcp-port',
        type=int,
        default=8000,
        help='TCP port for the MCP HTTP server (default: 8000)',
    )
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    return parser.parse_known_args(raw_argv)


def create_app(argv=None):
    """Create and configure the TemDataBrowser DataBrowser application and plugins.

    Returns (app, mcp_plugin) where mcp_plugin may be None if fastmcp is not installed.
    """
    mcp_args, remaining = parse_mcp_args(argv)

    # ScopeFoundry's DataBrowser.__init__ parses sys.argv[1:] directly via argparse,
    # so temporarily strip MCP-specific flags while preserving built-in DataBrowser flags.
    orig_argv = sys.argv
    try:
        sys.argv = [orig_argv[0], *remaining]
        app = DataBrowser(sys.argv)
    finally:
        sys.argv = orig_argv

    if not remaining or '--browse_dir' not in remaining:
        app.settings['browse_dir'] = str(Path.home())
    app.ui.setWindowTitle("TemDataBrowser")

    # Load views here (last loaded is the first one tried by auto_select_view)
    app.add_view(ArrayImageView(app))
    app.add_view(TemMetadataView(app))
    app.add_view(imageioView(app))
    app.add_view(TemView(app))

    mcp_plugin = None
    try:
        from .mcp_server import FASTMCP_AVAILABLE, MCPServerPlugIn

        if FASTMCP_AVAILABLE:
            mcp_plugin = MCPServerPlugIn(
                app,
                host=mcp_args.mcp_host,
                port=mcp_args.mcp_port,
            )
            app.add_plugin(mcp_plugin)
            if mcp_args.mcp:
                mcp_plugin.start_server()
        elif mcp_args.mcp:
            raise ImportError(
                "FastMCP is required for --mcp mode. Install with: pip install 'TemDataBrowser[mcp]'"
            )
    except ImportError:
        if mcp_args.mcp:
            raise

    return app, mcp_plugin


def open_file():
    """Start the graphical user interface from inside a python interpreter."""
    main()


def main(argv=None):
    """ This starts the graphical user interface and loads the views."""
    app, _mcp_plugin = create_app(argv)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
