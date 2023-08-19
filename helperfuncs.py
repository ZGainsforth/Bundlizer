import streamlit as st
import os, re
import shutil
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import imwrite, TiffWriter, TiffFile
import yaml
from yaml.representer import SafeRepresenter
import importlib
import subprocess

'''--------------- GENERAL FUNCTIONS ---------------'''

# Convert (X,Y) format to two floats for getting fiducials.
def get_coordinates(coordinatesInput):
    try:
        coords = coordinatesInput.strip("()").split(",")
        x, y = float(coords[0]), float(coords[1])
        return x, y
    except:
        st.write(f"Unable to parse coordinates: '{coordinatesInput}'. Please enter coordinates in the format (X, Y)")
        return None, None

# Printing function which sends text into the void.
# Also can use print for sending to terminal, or st.write for streamlit output.
def nuke_text(args):
    return

@st.cache_data
def load_textfile(fileName):
    with open(fileName, 'r') as f:
        fileContents = f.read()
    return fileContents

def numpy_to_yaml(arr):
    with np.printoptions(linewidth=np.inf):
        s = np.array2string(arr, separator=', ', formatter={'float': lambda x: str(x)})
        # remove the brackets []
        s = s[1:-1]
    return s

def sanitize_dict_for_yaml(data):
    sanitized_data = {}
    for k, v in data.items():
        # Clean up the key name so it has no spaces or brackets.
        k = k.replace(' []', '') # If there are empty brackets at the end we just get rid of them.
        k = k.replace('[', '_').replace(']', '') # Filled brackets get changed with a preceding dunderscore.
        k = re.sub(r'\s+', '_', k) # And no whitespace.
        # Clean up the values.
        if isinstance(data, np.ndarray):
            sanitized_data[k] = numpy_to_yaml(v)
        elif isinstance(v, bytes):
            sanitized_data[k] = v.decode('utf-8')
        elif isinstance(v, dict):
            sanitized_data[k] = sanitize_dict_for_yaml(v)
        else:
            sanitized_data[k] = v
    return sanitized_data

# Thank you stackoverflow: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_dict(d, parent_key='', sep='.'):
    flattened_dict = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            flattened_dict.update(flatten_dict(v, new_key, sep=sep))
        else:
            flattened_dict[new_key] = v
    return flattened_dict

# add keys from dict 2 to dict 1 but don't overwrite keys already in dict1.
def union_dict_no_overwrite(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]

# OME-TIF stores image resolution as a unit, and each pixel is no many units across.
# TIF resolution tags use pixels/cm so we need to convert whatever units we have to px/cm.
def ome_to_resolution_cm(metadata):
    match metadata['PhysicalSizeXUnit']:
        case 'A' | 'Å' | '1/A' | '1/Å':
            scale = 1e8
        case 'nm' | '1/nm':
            scale = 1e7
        case 'um' | 'µm' | '1/um' | '1/µm':
            scale = 1e4
        case 'mm':
            scale = 10 
        case 'cm':
            scale = 1
        case 'm':
            scale = 0.01
    xval = scale/metadata['PhysicalSizeX']
    yval = scale/metadata['PhysicalSizeY']
    return (xval, yval)

def replace_greek_symbols(input_string):
    return input_string.replace("µ", "u").replace("Å", "A") 

# Look through the samisdata.csv file and find the row that matches this file if it is present.
def samis_dict_for_this_file(samisData=None, fileName=None, statusOutput=print):
    # If there is no info then we need to use an empty dict.
    if samisData is None:
        return {}

    # Use the .loc function to find rows where the "filename" column matches
    matchedRows = samisData.loc[samisData['filename'] == os.path.basename(fileName)]

    if matchedRows.empty:
        statusOutput(f"No metadata found in samisdata.csv for {fileName}")
        samisData = {}
    else:
        # Return the first matching row as a DataFrame
        samisData = matchedRows.iloc[0].to_dict()
    
    if len(matchedRows) > 1:
        statusOutput(f"Multiple rows in samisdata.csv matched {fileName}.  Using only first row.")
    
    # Turn it into a dictionary for the caller
    return samisData

def load_instrument_processor(bundleInfo=None):
    # Now we are going to import the appropriate python script to do raw data processing for this instrument.
    sanitizeRegex = r"\s+|-|\." # selects for any whitespace, dash or period.
    # Load the instrument processor for the BDD + instrument combo.
    try:
        instrumentModuleName = f"BundleProcessors.{re.sub(sanitizeRegex, '_', bundleInfo['analysisTechniqueIdentifier'])}.{re.sub(sanitizeRegex, '_', bundleInfo['instrumentName'])}" 
        instrumentProcessor = importlib.import_module(instrumentModuleName)
    except ModuleNotFoundError as e:
        # If we can't load the combo, then we fall back to using a generic BDD instrument processor.
        instrumentModuleName = f"BundleProcessors.{re.sub(sanitizeRegex, '_', bundleInfo['analysisTechniqueIdentifier'])}" 
        instrumentProcessor = importlib.import_module(instrumentModuleName)
        # If this doesn't work then it is an error.
        assert hasattr(instrumentProcessor,'preprocess_all_products'), f'No instrument processor for the BDD/instrument combo: {instrumentModuleName}, {instrumentProcessor}'
    st.session_state['instrumentProcessor'] = instrumentProcessor
    return instrumentProcessor

def zip_directory(rootDir, subDir):
    original_dir = os.getcwd()
    os.chdir(rootDir)
    command = f"7z a -tzip -mx=9 -mmt={os.cpu_count()} {subDir}.zip {subDir}"
    subprocess.run(command, shell=True)
    os.chdir(original_dir)


'''--------------- EMD FUNCTIONS ---------------'''

def create_emd(fileName=None):
    emd = h5py.File(emdFileName, 'w')
    emd.attrs['version_major'] = 0
    emd.attrs['version_minor'] = 2
    return emd

#  def emd_add_cube(emd=None, groupName=None, )
'''--------------- INIT FUNCTIONS ---------------'''

# We want to create directories for processing data.
def initialize_directories():
    if not os.path.exists('Raw'):
        os.makedirs('Raw')
    if not os.path.exists('Output'):
        os.makedirs('Output')

'''--------------- CLEANUP FUNCTIONS ---------------'''

# Make some code which will do cleanup if stuff fails.
cleanupActions = []

# We need to add things to the garbage collection queue.
# rmtree deletes a directory and subdirs.
def add_cleanup_action(cleanupType='rmtree', cleanupData=None):
    cleanupActions[cleanupType] = cleanupData
    return

# This does each cleanup step and finally ends with st.stop()
def do_cleanup():
    for cleanupType, cleanupData in cleanupActions.items():
        match cleanupType:
            case 'rmtree':
                shutil.rmtree(cleanupData)

    # Clear the cleanupActions dictionary after performing cleanup
    cleanupActions.clear()

def cleanup_and_stop():
    do_cleanup()
    st.stop()


# Make some code which will do cleanup if stuff fails.
cleanupActions = []

# We need to add things to the garbage collection queue.
# rmtree deletes a directory and subdirs.
def add_cleanup_action(cleanupType='rmtree', cleanupData=None):
    # Append the cleanup type and data as a list to cleanupActions
    cleanupPair = [cleanupType, cleanupData]
    if cleanupPair not in cleanupActions:
        cleanupActions.append(cleanupPair)


# This does each cleanup step and finally ends with st.stop()
def do_cleanup():
    for cleanupType, cleanupData in cleanupActions:
        match cleanupType:
            case 'rmtree':
                shutil.rmtree(cleanupData)

    # Clear the cleanupActions list after performing cleanup
    cleanupActions.clear()

def cleanup_and_stop():
    # Perform cleanup and stop the Streamlit application
    do_cleanup()
    st.stop()

'''--------------- PLOTTING FUNCTIONS ---------------'''

# We want to create directories for processing data.
def plot_array(img):
    fig = plt.figure()
    plt.gca().imshow(img)
    st.pyplot(fig)

# We want to create directories for processing data.
def plot_png(fileName):
    img = imread(fileName)
    fig = plt.figure()
    plt.gca().imshow(img)
    st.pyplot(fig)

'''--------------- BUNDLE PRODUCT WRITE FUNCTIONS ---------------'''

# We want to create directories for processing data.
def write_ome_tif_image(fileName, sessionId, productId, img, core_metadata, addl_metadata):

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"

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
        'ranges': [0.0,1.0],
        }
    # Add all the metadata here so it is all embedded in the OME XML header.
    ome_metadata.update(sanitize_dict_for_yaml(core_metadata))
    ome_metadata.update(sanitize_dict_for_yaml(addl_metadata))
    resolution = ome_to_resolution_cm(ome_metadata)

    with TiffWriter(os.path.join(os.path.dirname(fileName), f'{productName}.ome.tif')) as tif:
        mean = np.mean(img)
        std = np.std(img)
        minValTag = (280,   # 280=MinSampleValue TIFF tag.  See https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
                    11,     # dtype float32 
                    1,      # one value in the tag.
                    mean-std,    # What the value is.
                    False,   # Write it to the first page of the Tiff only.
                    )
        maxValTag = (281, 11, 1, mean+std, False) # MaxSampleValue TIFF tag.
        tif.write(img, photometric='minisblack', metadata=ome_metadata, resolution=resolution, resolutionunit='CENTIMETER', extratags=[minValTag, maxValTag])

    # Write the supplementary yaml+txt too with all the instrument data.
    # Hint: the txt file is a yaml, but the yaml pointing to the metadata has to be the BDD yaml only, no extra keys.
    yamlFileName = os.path.join(os.path.dirname(fileName), f"{sessionId}_instrumentMetadata_{productId:05d}.ome.txt")
    with open(yamlFileName, 'w') as f:
        yaml.dump(ome_metadata, f, default_flow_style=False, sort_keys=False)
    yamlFileName = os.path.join(os.path.dirname(fileName), f"{sessionId}_instrumentMetadata_{productId:05d}.ome.yaml")
    with open(yamlFileName, 'w') as f:
        suppYaml = {"description": core_metadata['description'],
                    "supDocType": 'instrumentMetadata',
                    # "supDocType": core_metadata['dataComponentType'],
                    "associatedFiles": [f'{productName}.ome.tif']}
                    # "associatedFiles": f'{sessionId}_instrumentMetadata_{productId:05d}.ome.txt'}
        yaml.dump(suppYaml, f, default_flow_style=False, sort_keys=False)


