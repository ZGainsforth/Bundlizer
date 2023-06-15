import streamlit as st
import os
import shutil
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

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
