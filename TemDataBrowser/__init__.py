#from __future__ import division, print_function, absolute_import

from pathlib import Path
from collections import OrderedDict
import functools

from ScopeFoundry import BaseApp
from ScopeFoundry.helper_funcs import load_qt_ui_from_pkg
from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
from qtpy import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
from ScopeFoundry.logged_quantity import LQCollection
import argparse

import imageio.v3 as iio
import ncempy

# Use row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major')

class imageioView(DataBrowserView):
    """ Handles most normal image types like TIF, PNG, etc."""
    
    # This name is used in the GUI for the DataBrowser
    name = 'Image viewer (imageio)'
    
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
        #  A new file has been selected by the user, load and display it
        try:
            self.data = iio.imread(fname)
            self.imview.setImage(self.data.swapaxes(0, 1))
        except Exception as err:
        	# When a failure to load occurs, zero out image
        	# and show error message
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage(
            	"failed to load %s:\n%s" %(fname, err))
            raise(err)

class TemView(DataBrowserView):
    """ Data browser for common S/TEM file types
    
    """

    # This name is used in the GUI for the DataBrowser
    name = 'TEM data viewer'
    
    def setup(self):
        """ create the GUI and viewer settings, runs once at program start up
            self.ui should be a QWidget of some sort, here we use a pyqtgraph ImageView
        """
        #self.ui = self.imview = pg.ImageView()
        #self.viewbox = self.imview.getView()
        self.plt = pg.PlotItem(labels={'bottom':('X',''),'left':('Y','')})
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
        """  A new file has been selected by the user, load and display it
        """
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
                print(f'Warning: only showing 1 image per tilt angle for STEMTomo7 data.')
                self.data = self.data[:,0,:,:]
            elif self.data.ndim == 4:
                print(f'Warning: Reducing {self.data.ndim}-D data to 3-D.')
                self.data = self.data[0,:,:,:]
            elif self.data.ndim > 4:
                print(f'{self.data.ndim}-D data files are not supported.')
            
            xscale = file['pixelSize'][-2]
            yscale = file['pixelSize'][-1]
            self.imview.setImage(self.data)
            img = self.imview.getImageItem()
            
            if file['pixelUnit'][-1] in ('um', 'µm', '[u_m]', 'u_m'):
                unit_scale = 1e-6
                unit = 'm'
            elif file['pixelUnit'][-1] in ('m', ):
                unit_scale = 1
                unit = 'm'
            elif file['pixelUnit'][-1] in ('nm', '[n_m]', 'n_m'):
                unit_scale = 1e-9
                unit = 'm'
            elif file['pixelUnit'][-1] in ('A', 'Ang', ):
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
            
        except Exception as err:
        	# When a failure to load occurs, zero out image
        	# and show error message
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage(
            	f'failed to load {fname}:\n{err}')
            raise(err)

class TemMetadataView(DataBrowserView):
    """ A viewer to read meta data from a file and display it as text.
    
    """
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
                tilts = map(float, f1)
            meta_data['tilt angles'] = tilts
        
        # Read FEI parameters from .txt file if it exists
        FEIparameters = Path(path).with_suffix('.txt')
        if FEIparameters.exists():
            with open(FEIparameters, 'r') as f2:
                lines = f2.readlines()
            pp1 = list([ii[18:].strip().split(':')] for ii in lines[3:-1])
            pp2 = {}
            for ll in pp1:
                try:
                    pp2[ll[0]] = float(ll[1])
                except:
                    pass  # skip lines with no data
            meta_data.update(pp2)

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
        ext = Path(fname).suffix

        meta_data = {'file name': str(fname)}
        if ext in ('.dm3', '.dm4'):
            meta_data = self.get_dm_metadata(fname)
        elif ext in ('.mrc', '.rec', '.ali'):
            meta_data = self.get_mrc_metadata(fname)
        elif ext in ('.emd',):
            try: 
                # Parse the file to see if any EMD datasets exist
                # if not then it throws a NoEmdDataSets error
                with ncempy.io.emd.fileEMD(fname) as f0:
                    meta_data = f0.getMetadata(0)
            except ncempy.io.emd.NoEmdDataSets:
                with ncempy.io.emdVelox.fileEMDVelox(fname) as f0:
                    if f0.list_data is not None:
                        meta_data = f0.getMetadata(0)
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
            line = f'{k} = {v}\n'
            txt += line
        self.ui.setText(txt)
    
    def is_file_supported(self, fname):
        ext = Path(fname).suffix
        return ext.lower() in ('.dm3', '.dm4', '.mrc', '.ali', '.rec', '.ser', '.emi', '.img')

def open_file():
    """Start the graphical user interface from inside a python interpreter."""
    main()

def main():
    """ This starts the graphical user interface and loads the views."""
    import sys
    
    app = DataBrowser(sys.argv)
    app.settings['browse_dir'] = Path.home()
    app.ui.setWindowTitle("TemDataBrowser")
    
    # Load views here
    # Last loaded is the first one tried
    app.load_view(TemMetadataView(app))
    app.load_view(imageioView(app))
    app.load_view(TemView(app))
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()
    
