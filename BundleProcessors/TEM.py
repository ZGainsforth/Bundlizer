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
import json

# For unique product ID's we start a counter.
productId = 0

def new_productId():
    global productId
    productId += 1
    return productId

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

def write_TEM_image(fileName=None, sessionId=None, statusOutput=print, img=None, core_metadata=None, addl_metadata=None):
    productId = new_productId()

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        "pixelScaleX": core_metadata['PhysicalSizeX'],
        "pixelScaleY": core_metadata['PhysicalSizeY'],
        "pixelUnits": core_metadata['PhysicalSizeXUnit'],
        }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.ome.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    # Write an OME-TIF with all the metadata, pixel scales, etc.
    hf.write_ome_tif_image(fileName, sessionId, productId, img, core_metadata, addl_metadata)

    # If this is an electron diffraction pattern, then there is supposed to be a PDF for the calibration.
    if core_metadata['dataComponentType'] == 'TEMPatternsImage':
        calibrationFileName = os.path.splitext(fileName)[0] + ".pdf"
        try:
            shutil.copyfile(calibrationFileName, f"{sessionId}_calibrationFile_{productId:05d}.ome.pdf")
        except Exception as e:
            statusOutput(f'Could not find calibration file {calibrationFileName}.')

    return

def preprocess_ser(fileName=None, sessionId=None, statusOutput=print, file=None, samisData=None):
    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')
    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": 'STEMImage',
        "PhysicalSizeX": float(file['pixelSize'][0]),
        "PhysicalSizeXUnit": str(file['pixelUnit'][1]),
        "PhysicalSizeY": float(file['pixelSize'][1]),
        "PhysicalSizeYUnit": str(file['pixelUnit'][0]),
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    write_TEM_image(fileName=fileName, sessionId=sessionId, statusOutput=statusOutput, img=file['data'][:,:].astype('float32'), core_metadata=core_metadata, addl_metadata=file['metadata'])
    return

def preprocess_dm(fileName=None, sessionId=None, statusOutput=print, file=None, samisData=None):
    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    if '/' in file.axes_manager['y'].units:
        dataComponentType = 'TEMPatternsImage'
    else:
        dataComponentType = 'TEMImage'

    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": dataComponentType,
        "PhysicalSizeX": float(file.axes_manager['x'].scale),
        "PhysicalSizeXUnit": file.axes_manager['x'].units,
        "PhysicalSizeY": float(file.axes_manager['y'].scale),
        "PhysicalSizeYUnit": file.axes_manager['y'].units,
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    write_TEM_image(fileName=fileName, sessionId=sessionId, statusOutput=statusOutput, img=file.data.astype('float32'), core_metadata=core_metadata, addl_metadata=file.metadata.as_dictionary())
    return

# Preprocess Bruker EDS cube.
def preprocess_bcf(fileName=None, sessionId=None, statusOutput=print, file=None, samisData=None):
    productId = new_productId()

    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": 'STEMEDSCube',
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    return

def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    # In the case of STXM, all products are pointed to by a hdr file.
    # Extract the file name.
    fullName, ext = os.path.splitext(fileName)
    baseName = os.path.basename(fullName) # This is the name of the file.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    match ext:
        case '.dm3' | '.dm4':
            file = hs.load(fileName)
            preprocess_dm(fileName, sessionId, statusOutput, file, samisData=samisData)
        case '.ser':
            file = ser.serReader(fileName)
            preprocess_ser(fileName, sessionId, statusOutput, file, samisData=samisData)
        case '.bcf':
            HAADF, EDS = hs.load(fileName)
        case _:
            raise ValueError(f"{ext} is an invalid data product type.")

    return

def preprocess_all_products(dirName=None, sessionId=None, statusOutput=print, statusProgress=None):
    # The user is telling us a directory which contains raw products from this instrument.
    if dirName is None:
        dirName =  os.getcwd()
    # Create a list of all files with extensions that we can process.
    rawFiles = []
    for ext in raw_extensions:
        rawFiles.extend(glob2.glob(os.path.join(dirName, '**', f'*.{ext}')))
    rawFiles.sort()

    # If there is a csv with additional metadata fields supplied by the user then load it.  Ususally this is used for descriptions.
    try:
        samisData = pd.read_csv(os.path.join(dirName, 'samisdata.csv'))
        statusOutput('Loaded metadata from samisdata.csv.')
    except:
        samisData = None
        statusOutput('There is no samisdata.csv.  Where are your descriptions going to come from?  Consider making a csv...')

    for i, f in enumerate(rawFiles):
        try:
            preprocess_one_product(f, sessionId=sessionId, statusOutput=statusOutput, samisData=samisData)
            statusOutput(f'Preprocessed {f}\n')
        except Exception as e:
            # If that one Xim failed, go on and process the next, it won't be included in the data products.
            statusOutput(f'Preprocessed {f} failed {e}.\n')
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
    return

if __name__ == '__main__':
    # preprocess_all_products()
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/017 EDS on Green phase/Before_1.ser', sessionId=314, statusOutput=print)
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/BundlizerData/20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser', sessionId=314, statusOutput=print)
    tempdir = '/Users/Zack/Desktop' # Mac
    # tempdir = '/home/zack/Rocket/WorkDir/BundlizerData' # Linux
    preprocess_all_products(os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer'), sessionId=314)
    # preprocess_one_product(fileName=os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser'), sessionId=314, statusOutput=print)
    # preprocess_one_product(fileName=os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0005.dm3'), sessionId=314, statusOutput=print)
    print ('Done')