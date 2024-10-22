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
            if Path(fname).suffix.lower() == '.emd':
                # Check for special STEMTomo7 Berkeley EMD files
                is_stemtomo = False
                if Path(fname).suffix.lower() == '.emd':
                    with ncempy.io.emd.fileEMD(fname) as f0:
                        if 'data' in f0.file_hdl:
                            if 'stemtomo version' in f0.file_hdl['data'].attrs:
                                is_stemtomo = True
            
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
        metaData = {}
        with ncempy.io.dm.fileDM(fname, on_memory=True) as dm1:
            # Only keep the most useful tags as meta data
            for kk, ii in dm1.allTags.items():
                # Most useful starting tags
                prefix1 = 'ImageList.{}.ImageTags.'.format(dm1.numObjects)
                prefix2 = 'ImageList.{}.ImageData.'.format(dm1.numObjects)
                pos1 = kk.find(prefix1)
                pos2 = kk.find(prefix2)
                if pos1 > -1:
                    sub = kk[pos1 + len(prefix1):]
                    metaData[sub] = ii
                elif pos2 > -1:
                    sub = kk[pos2 + len(prefix2):]
                    metaData[sub] = ii

                # Remove unneeded keys
                for jj in list(metaData):
                    if jj.find('frame sequence') > -1:
                        del metaData[jj]
                    elif jj.find('Private') > -1:
                        del metaData[jj]
                    elif jj.find('Reference Images') > -1:
                        del metaData[jj]
                    elif jj.find('Frame.Intensity') > -1:
                        del metaData[jj]
                    elif jj.find('Area.Transform') > -1:
                        del metaData[jj]
                    elif jj.find('Parameters.Objects') > -1:
                        del metaData[jj]
                    elif jj.find('Device.Parameters') > -1:
                        del metaData[jj]

            # Store the X and Y pixel size, offset and unit
            try:
                metaData['PhysicalSizeX'] = metaData['Calibrations.Dimension.1.Scale']
                metaData['PhysicalSizeXOrigin'] = metaData['Calibrations.Dimension.1.Origin']
                metaData['PhysicalSizeXUnit'] = metaData['Calibrations.Dimension.1.Units']
                metaData['PhysicalSizeY'] = metaData['Calibrations.Dimension.2.Scale']
                metaData['PhysicalSizeYOrigin'] = metaData['Calibrations.Dimension.2.Origin']
                metaData['PhysicalSizeYUnit'] = metaData['Calibrations.Dimension.2.Units']
            except:
                metaData['PhysicalSizeX'] = 1
                metaData['PhysicalSizeXOrigin'] = 0
                metaData['PhysicalSizeXUnit'] = ''
                metaData['PhysicalSizeY'] = 1
                metaData['PhysicalSizeYOrigin'] = 0
                metaData['PhysicalSizeYUnit'] = ''

        return metaData
    
    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_mrc_metadata(path):
        """ Reads important metadata from MRC and related files."""
        metaData = {}

        # Open file and parse the header
        with ncempy.io.mrc.fileMRC(path) as mrc1:
            pass

        # Save most useful metaData
        metaData['axisOrientations'] = mrc1.axisOrientations  # meta data information from the mrc header
        metaData['cellAngles'] = mrc1.cellAngles

        if hasattr(mrc1, 'FEIinfo'):
            # add in the special FEIinfo if it exists
            try:
                metaData.update(mrc1.FEIinfo)
            except TypeError:
                pass

        # Store the X and Y pixel size, offset and unit
        # Test for bad pixel sizes which happens often
        if mrc1.voxelSize[2] > 0:
            metaData['PhysicalSizeX'] = mrc1.voxelSize[2] * 1e-10  # change Angstroms to meters
            metaData['PhysicalSizeXOrigin'] = 0
            metaData['PhysicalSizeXUnit'] = 'm'
        else:
            metaData['PhysicalSizeX'] = 1
            metaData['PhysicalSizeXOrigin'] = 0
            metaData['PhysicalSizeXUnit'] = ''
        if mrc1.voxelSize[1] > 0:
            metaData['PhysicalSizeY'] = mrc1.voxelSize[1] * 1e-10  # change Angstroms to meters
            metaData['PhysicalSizeYOrigin'] = 0
            metaData['PhysicalSizeYUnit'] = 'm'
        else:
            metaData['PhysicalSizeY'] = 1
            metaData['PhysicalSizeYOrigin'] = 0
            metaData['PhysicalSizeYUnit'] = ''

        metaData['FileName'] = path

        rawtltName = Path(path).with_suffix('.rawtlt')
        if rawtltName.exists():
            with open(rawtltName, 'r') as f1:
                tilts = map(float, f1)
            metaData['tilt angles'] = tilts

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
            metaData.update(pp2)

        return metaData
    
    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_emd_metadata(path):
        """ Reads important metadata from EMD Berkeley files."""
        metaData = {}
        with ncempy.io.emd.fileEMD(path) as emd0:        
            try:
                metaData.update(emd0.user.attrs)
            except AttributeError:
                pass
            try:
                metaData.update(emd0.microscope.attrs)
            except AttributeError:
                pass
            try:
                metaData.update(emd0.sample.attrs)
            except AttributeError:
                pass
            dims = emd0.get_emddims(emd0.list_emds[0])
            dimX = dims[-1]
            dimY = dims[-2]
            metaData['PhysicalSizeX'] = dimX[0][1] - dimX[0][0]
            metaData['PhysicalSizeXOrigin'] = dimX[0][0]
            metaData['PhysicalSizeXUnit'] = dimX[2].replace('_', '')
            metaData['PhysicalSizeY'] = dimY[0][1] - dimY[0][0]
            metaData['PhysicalSizeYOrigin'] = dimY[0][0]
            metaData['PhysicalSizeYUnit'] = dimY[2].replace('_', '')
        return metaData


    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_velox_metadata(path):
        """ Reads important metadata from Velox EMD files."""
        import json
        metaData = {}
        with ncempy.io.emdVelox.fileEMDVelox(path) as f0:
            dataGroup = emd_obj.list_data[0]
            dataset0 = dataGroup['Data']

            # Convert JSON metadata to dict
            mData = emd_obj.list_data[0]['Metadata'][:, 0]
            validMetaDataIndex = npwhere(mData > 0)  # find valid metadata
            mData = mData[validMetaDataIndex].tostring()  # change to string
            mDataS = json.loads(mData.decode('utf-8', 'ignore'))  # load UTF-8 string as JSON and output dict
            metaData['pixel sizes'] = []
            metaData['pixel units'] = []
            try:
                # Store the X and Y pixel size, offset and unit
                metaData['PhysicalSizeX'] = float(mDataS['BinaryResult']['PixelSize']['width'])
                metaData['PhysicalSizeXOrigin'] = float(mDataS['BinaryResult']['Offset']['x'])
                metaData['PhysicalSizeXUnit'] = mDataS['BinaryResult']['PixelUnitX']
                metaData['PhysicalSizeY'] = float(mDataS['BinaryResult']['PixelSize']['height'])
                metaData['PhysicalSizeYOrigin'] = float(mDataS['BinaryResult']['Offset']['y'])
                metaData['PhysicalSizeYUnit'] = mDataS['BinaryResult']['PixelUnitY']
            except:
                metaData['PhysicalSizeX'] = 1
                metaData['PhysicalSizeXOrigin'] = 0
                metaData['PhysicalSizeXUnit'] = ''
                metaData['PhysicalSizeY'] = 1
                metaData['PhysicalSizeYOrigin'] = 0
                metaData['PhysicalSizeYUnit'] = ''

            metaData.update(mDataS)

            metaData['shape'] = dataset0.shape
        return metaData
    
    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_ser_metadata(fname):
            metaData = {}
            with ncempy.io.ser.fileSER(fname) as ser1:
                data, metaData = ser1.getDataset(0)  # have to get 1 image and its meta data

                # Add extra meta data from the EMI file if it exists
                if ser1._emi is not None:
                    metaData.update(ser1._emi)

            metaData.update(ser1.head)  # some header data for the ser file

            # Clean the dictionary
            for k, v in metaData.items():
                if isinstance(v, bytes):
                    metaData[k] = v.decode('UTF8')

            # Store the X and Y pixel size, offset and unit
            try:
                metaData['PhysicalSizeX'] = metaData['Calibration'][0]['CalibrationDelta']
                metaData['PhysicalSizeXOrigin'] = metaData['Calibration'][0]['CalibrationOffset']
                metaData['PhysicalSizeXUnit'] = 'm'  # always meters
                metaData['PhysicalSizeY'] = metaData['Calibration'][1]['CalibrationDelta']
                metaData['PhysicalSizeYOrigin'] = metaData['Calibration'][1]['CalibrationOffset']
                metaData['PhysicalSizeYUnit'] = 'm'  # always meters
            except:
                metaData['PhysicalSizeX'] = 1
                metaData['PhysicalSizeXOrigin'] = 0
                metaData['PhysicalSizeXUnit'] = ''
                metaData['PhysicalSizeY'] = 1
                metaData['PhysicalSizeYOrigin'] = 0
                metaData['PhysicalSizeYUnit'] = ''

            return metaData
    
    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_emi_metadata(fname):
        return ncempy.io.ser.read_emi(fname)
    
    @staticmethod
    @functools.lru_cache(maxsize=10, typed=False)
    def get_img_metadata(fname):
        print(fname)
        with ncempy.io.smv.fileSMV(fname) as f0:
            md = f0.header_info
        return md
        
    def on_change_data_filename(self, fname):
        ext = Path(fname).suffix

        meta_data = {'file name': str(fname)}
        if ext in ('.dm3', '.dm4'):
            meta_data = self.get_dm_metadata(fname)
        elif ext in ('.mrc', '.rec', '.ali'):
            meta_data = self.get_mrc_metadata(fname)
        elif ext in ('.emd',):
            with ncempy.io.emd.fileEMD(fname) as emd0:
                if len(emd0.list_emds) > 0:
                    meta_data = self.get_emd_metadata(fname)
                else:
                    meta_data = self.get_velox_metadata(fname)
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
    # Load views here
    # Last loaded is the first one tried
    app.load_view(TemMetadataView(app))
    app.load_view(imageioView(app))
    app.load_view(TemView(app))
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()
    
