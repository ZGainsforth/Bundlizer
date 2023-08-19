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
import xml.etree.ElementTree as ET
from yaml.representer import SafeRepresenter
import importlib
hf = importlib.import_module('helperfuncs', '..')
import json
import h5py
import datetime
# Use https://www.github.com/ZGainsforth/anasys-python-tools
import anasyspythontools as anasys

# For unique product ID's we start a counter.
productId = 0

def new_productId():
    global productId
    productId += 1
    return productId

# These file extensions indicate data types that we can convert 
raw_extensions = [
                  'irb', # background spectrum
                  'axz', # Data collection containing images and spectra.
                  ]

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

def preprocess_irb(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    productId = new_productId()

    productName = f"{sessionId}_nanoIRPointCollection_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

   # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    # The background spectrum file is stored as an xml
    tree = ET.ElementTree(file=fileName)
    root = tree.getroot()
    # Get all the single value nodes.
    nodes = {child.tag: child.text for child in root}
    # Get all the nexted nodes (arrays of numbers)
    data = {child.tag: [grandchild.text for grandchild in child] for child in root}
    # The intensity of the background spectrum.
    intensity = np.array(data['Table']).astype(float)

    # Calculate the energy axis.
    start = int(nodes['StartWavenumber'])
    end = int(nodes['EndWavenumber'])
    resolution = int(nodes['IRSweepResolution'])
    num_steps = int((end - start) / resolution) + 1
    energies = np.linspace(start, end, num_steps)

    # Create and save the yaml file for this background spectrum.
    yamlData = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": "nanoIRPointCollection",
        "headerRowCount":3,
        "countColumns":2,
        "countRows":len(energies),
        "columns": {
            "1": {
             "label":"cm-1",
             "fieldDescription":"Energy axis in wavenumbers",
             "fieldType":"decimal",
             "unitOfMeasure":"cm-1",
            },
            "2": {
             "label":nodes['Units'],
             "fieldDescription":"Intensity",
             "fieldType":"decimal",
             "unitOfMeasure":nodes['Units'],
            }
        },
    }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    # Create the CSV output.
    header =  f"#Laser Background Spectrum\n"
    header += f"#IRSweepSpeed={nodes['IRSweepSpeed']}\n"
    header += f"#cm-1,{nodes['Units']}"
    np.savetxt(os.path.join(os.path.dirname(fileName), f'{productName}.csv'), np.stack((energies, intensity), axis=1), delimiter=',', header=header)

    return

def preprocess_axz(fileName=None, sessionId=None, statusOutput=print, haadf=None, samisData=None):
    productId = new_productId()

    productName = f"{sessionId}_nanoIRMapCollection_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    file = anasys.read(fileName)

    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": 'STEMEDSCube',
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    return

def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    # In the case of NanoIR, all products can be identified from the file extension.
    fullName, ext = os.path.splitext(fileName)
    baseName = os.path.basename(fullName) # This is the name of the file.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    match ext:
        case '.irb':
            preprocess_irb(fileName, sessionId, statusOutput, samisData)
        case '.axz':
            preprocess_axz(fileName, sessionId, statusOutput, samisData)
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
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/BundlizerData/20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser', sessionId=314, statusOutput=print)
    # tempdir = '/Users/Zack/Dropbox/OSIRIS-REx/BundlizerData/NanoIR' # Mac
    tempdir = '/home/zack/Dropbox/OSIRIS-REx/BundlizerData/NanoIR' # Linux
    preprocess_all_products(os.path.join(tempdir, '20221205 - NanoIR Tapping - IOM - ORExSAT - Murchison B_1'), sessionId=314)
    print ('Done')