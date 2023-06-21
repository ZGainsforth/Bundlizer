# Created 2023, Zack Gainsforth
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import hyperspy.api as hs
import pandas as pd
import re, os, io, gc
import shutil
import glob2
import yaml
from yaml.representer import SafeRepresenter
from ncempy.io import dm, ser # Reader for dm3/dm4 files and ser (TIA) files.
import importlib
hf = importlib.import_module('helperfuncs', '..')
from tifffile import imwrite, TiffWriter
# from tifffile import imsave as tiffsave
import json

# For unique product ID's we start a counter.
productId = 0

# These file extensions indicate data types that we can convert 
raw_extensions = ['dm3', # Gatan files
                  'dm4',
                  'ser', # TIA files.
                  'bcf', # Bruker EDS files.
                 ]

# Figure out if this is a TEM image, a STEM image or what -- for ser files.
def get_image_type(metadata=None):
    match str(metadata['Mode []']):
        case value if 'STEM' in value:
            return 'STEM'
        case _:
            raise ValueError('Cannot guess image type from metadata.  Please expand case statement.')

# Given this is a STEM image, is this a HAADF image, BF-STEM, etc?
def get_STEM_type(metadata=None):
    # For now, only HAADF.
    return 'HAADF'

def write_TEM(fileName=None, sessionId=None, statusOutput=print, img=None, core_metadata=None, addl_metadata=None):
    global productId
    productId += 1

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        "channel": core_metadata['channel'],
        "pixelScaleX": core_metadata['PhysicalSizeX'],
        "pixelScaleY": core_metadata['PhysicalSizeY'],
        "pixelUnits": core_metadata['PhysicalSizeXUnit'],
    }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.ome.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    ome_metadata={
        'axes': 'YX',
        'PixelType': 'float32',
        'BigEndian': False,
        'SizeX': img.shape[0],
        'SizeY': img.shape[1],
        'PhysicalSizeX': core_metadata['PhysicalSizeX'], # Pixels/unit
        'PhysicalSizeXUnit': core_metadata['PhysicalSizeXUnit'],
        'PhysicalSizeY': core_metadata['PhysicalSizeY'], # Pixels/unit
        'PhysicalSizeYUnit': core_metadata['PhysicalSizeYUnit'],
    }
    # Add all the metadata here so it is all embedded in the OME XML header.
    ome_metadata.update(hf.sanitize_dict_for_yaml(core_metadata))
    ome_metadata.update(hf.sanitize_dict_for_yaml(addl_metadata))
    resolution = hf.ome_to_resolution_cm(ome_metadata)

    with TiffWriter(os.path.join(os.path.dirname(fileName), f'{productName}.ome.tif')) as tif:
        tif.write(img, photometric='minisblack', metadata=ome_metadata, resolution=resolution, resolutionunit='CENTIMETER')

    # Write the supplementary yaml too with all the instrument data.
    yamlFileName = os.path.join(os.path.dirname(fileName), f"{sessionId}_instrumentMetadata_{productId:05d}.ome.yaml")
    with open(yamlFileName, 'w') as f:
        yaml.dump(ome_metadata, f, default_flow_style=False, sort_keys=False)

    # If this is an electron diffraction pattern, then there is supposed to be a PDF for the calibration.
    if core_metadata['dataComponentType'] == 'TEMPatternsImage':
        calibrationFileName = os.path.splitext(fileName)[0] + ".pdf"
        try:
            shutil.copyfile(calibrationFileName, f"{sessionId}_calibrationFile_{productId:05d}.ome.pdf")
        except Exception as e:
            statusOutput(f'Could not find calibration file {calibrationFileName}.  Bundle format will not be valid.')

    return productName

def preprocess_ser(fileName=None, sessionId=None, statusOutput=print, file=None):
    core_metadata = { 
        "description": f'{os.path.basename(fileName)}',
        "dataComponentType": 'STEMImage',
        "channel": get_STEM_type(metadata=file['metadata']),
        "PhysicalSizeX": float(file['pixelSize'][0]),
        "PhysicalSizeXUnit": str(file['pixelUnit'][1]),
        "PhysicalSizeY": float(file['pixelSize'][1]),
        "PhysicalSizeYUnit": str(file['pixelUnit'][0]),
    }

    return write_TEM(fileName=fileName, sessionId=sessionId, statusOutput=statusOutput, img=file['data'][:,:].astype('float32'), core_metadata=core_metadata, addl_metadata=file['metadata'])

def preprocess_dm(fileName=None, sessionId=None, statusOutput=print, file=None):
    if '/' in file.axes_manager['y'].units:
        dataComponentType = 'TEMPatternsImage'
        channel = ''
    else:
        dataComponentType = 'TEMImage'
        channel = 'TEM'

    core_metadata = { 
        "description": f'{os.path.basename(fileName)}',
        "dataComponentType": dataComponentType,
        "channel": channel,
        "PhysicalSizeX": float(file.axes_manager['x'].scale),
        "PhysicalSizeXUnit": file.axes_manager['x'].units,
        "PhysicalSizeY": float(file.axes_manager['y'].scale),
        "PhysicalSizeYUnit": file.axes_manager['y'].units,
    }

    return write_TEM(fileName=fileName, sessionId=sessionId, statusOutput=statusOutput, img=file.data.astype('float32'), core_metadata=core_metadata, addl_metadata=file.metadata.as_dictionary())

def preprocess_EDSCube(fileName=None, sessionId=None, statusOutput=print, file=None):
    return

def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print):
    # In the case of STXM, all products are pointed to by a hdr file.
    # Extract the file name.
    fullName, ext = os.path.splitext(fileName)
    baseName = os.path.basename(fullName) # This is the name of the file.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    match ext:
        case '.dm3' | '.dm4':
            file = hs.load(fileName)
            fileType = file.metadata.Acquisition_instrument.TEM.acquisition_mode
            return preprocess_dm(fileName, sessionId, statusOutput, file)
        case '.ser':
            file = ser.serReader(fileName)
            fileType = get_image_type(file['metadata'])
            return preprocess_ser(fileName, sessionId, statusOutput, file)
        case '.bcf':
            HAADF, EDS = hs.load(fileName)
            fileType = 'EDSCube'
            return 'cube'
        case _:
            raise ValueError(f"{ext} is an invalid data product type.")

def preprocess_all_products(dirName=None, sessionId=None, statusOutput=print, statusProgress=None):
    # The user is telling us a directory which contains raw products from this instrument.
    if dirName is None:
        dirName =  os.getcwd()
    # On STXM, all data stacks of all types are identified by .hdr files.
    rawFiles = []
    for ext in raw_extensions:
        rawFiles.extend(glob2.glob(os.path.join(dirName, '**', f'*.{ext}')))
    # rawFiles = rawFiles.sorted()

    productsList = {}
    for i, f in enumerate(rawFiles):
        try:
            dsd = preprocess_one_product(f, sessionId=sessionId, statusOutput=statusOutput)
            productsList.update(dsd)
            statusOutput(f'Preprocessed {f} -> {dsd}.')
        except Exception as e:
            # If that one Xim failed, go on and process the next, it won't be included in the data products.
            statusOutput(f'Preprocessed {f} -> failed {e}.')
            pass
        # We are tearing through RAM with these stacks.  Sometimes, the garbage collection can't keep up and we run out of memory.
        # Or we force it to clean up after each stack.
        gc.collect()
        # Update the status in the calling page.
        if statusProgress is not None:
            statusProgress.progress(i/len(rawFiles), text='Processing raw data.  This may take a minute.')
    # And always leave a final status of 100%
    if statusProgress is not None:
        statusProgress.progress(1.0, text='Raw data processing complete.')
    return productsList

if __name__ == '__main__':
    # preprocess_all_products()
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/017 EDS on Green phase/Before_1.ser', sessionId=314, statusOutput=print)
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/BundlizerData/20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser', sessionId=314, statusOutput=print)
    tempdir = '/Users/Zack/Desktop' # Mac
    # tempdir = '/home/zack/Rocket/WorkDir/BundlizerData' # Linux
    preprocess_all_products(os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer'))
    # preprocess_one_product(fileName=os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser'), sessionId=314, statusOutput=print)
    # preprocess_one_product(fileName=os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0005.dm3'), sessionId=314, statusOutput=print)
    print ('Done')