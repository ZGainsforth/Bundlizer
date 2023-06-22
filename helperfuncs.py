import streamlit as st
import os, re
import shutil
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        
# Look through the samisdata.csv file and find the row that matches this file if it is present.
def samis_dict_for_this_file(samisData=None, fileName=None, statusOutput=print):
    # Use the .loc function to find rows where the "filename" column matches
    matchedRows = samisData.loc[samisData['filename'] == os.path.basename(fileName)]

    if matchedRows.empty:
        statusOutput(f"No metadata found in samisdata.csv for {f}")
        samisData = {}
    else:
        # Return the first matching row as a DataFrame
        samisData = matchedRows.iloc[0].to_dict()
    
    if len(matchedRows) > 1:
        statusOutput(f"Multiple rows in samisdata.csv matched {f}.  Using only first row.")
    
    # Turn it into a dictionary for the caller
    return samisData

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
def plot_png(fileName):
    img = imread(fileName)
    fig = plt.figure()
    plt.gca().imshow(img)
    st.pyplot(fig)
