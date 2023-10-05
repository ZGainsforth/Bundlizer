# Created 2023, Zack Gainsforth
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
import h5py
import datetime
import zipfile
import tifffile
from io import BytesIO

# For unique product ID's we start a counter.
productId = 0

def new_productId():
    global productId
    productId += 1
    return productId

# These file extensions indicate data types that we can convert 
raw_extensions = [
                  'im', # Cameca files
                  ]

def plot_tiffs_from_zip(zipFileName):
    # Read ZIP file
    with zipfile.ZipFile(zipFileName, 'r') as z:
        tiff_names = [name for name in z.namelist() if name.lower().endswith('.tif') or name.lower().endswith('.tiff')]
        
        # Calculate the number of subplots assuming a square shape.
        n = len(tiff_names)
        ncols = int(np.ceil(np.sqrt(n)))
        fig, ax = plt.subplots(nrows=ncols, ncols=ncols, figsize=(8, 8))
        ax = ax.ravel()

        for i, tiff_name in enumerate(tiff_names):
            with z.open(tiff_name) as tiff_file:
                tiff_data = tifffile.imread(BytesIO(tiff_file.read()))
                ax[i].imshow(tiff_data)
                ax[i].set_title(f"Image {i+1}")
                ax[i].axis('off')

        for i in range(n, ncols**2):
            ax[i].axis('off')

        plt.tight_layout()
    return fig  

def plot_im(fileName):

    return plot_tiffs_from_zip(fileName)

def preprocess_im(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    productId = new_productId()

    # im files are supposed to be in a directory with other files, and the directory gets turned into a collection.

    dataComponentType = 'NanoSIMSImageCollection'
    productName = f"{sessionId}_{dataComponentType}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    # Create and save the yaml file for this spectrum.
    yamlData = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(os.path.dirname(fileName)),
        "dataComponentType": dataComponentType,
    }
    yamlFileName = os.path.join(os.path.dirname(fileName), '..', f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    # Create a zip file for the directory
    shutil.make_archive(f'{os.path.join(os.path.dirname(fileName), "..", productName)}', 'zip', os.path.dirname(fileName))

    return

def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    # In the case of STXM, all products are pointed to by a hdr file.
    # Extract the file name.
    fullName, ext = os.path.splitext(fileName)
    baseName = os.path.basename(fullName) # This is the name of the file.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    # Get any samis data specific to this product (it could be in a subdir with additional samis info.)
    samisData = hf.load_samisdata(os.path.dirname(fileName), samisData, statusOutput)

    match ext:
        case '.im':
            preprocess_im(fileName, sessionId, statusOutput, samisData)
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

    # Load the base samisData.
    samisData = hf.load_samisdata(dirName, None, statusOutput)
    # # If there is a csv with additional metadata fields supplied by the user then load it.  Ususally this is used for descriptions.
    # try:
    #     samisData = pd.read_csv(os.path.join(dirName, 'samisdata.csv'))
    #     statusOutput('Loaded metadata from samisdata.csv.')
    # except:
    #     samisData = None
    #     statusOutput('There is no samisdata.csv.  Where are your descriptions going to come from?  Consider making a csv...')

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
    tempdir = '/home/zack/Dropbox/OSIRIS-REx/BundlizerData/NanoSIMS' # Linux
    preprocess_all_products(os.path.join(tempdir, 'Raw NanoSIMS'), sessionId='123')
    print ('Done')