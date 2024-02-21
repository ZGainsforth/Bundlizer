import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os,sys,shutil
import pandas as pd
from collections import OrderedDict
import yaml
import zipfile
from io import BytesIO
import importlib
import re
from helperfuncs import get_coordinates, nuke_text, add_cleanup_action, cleanup_and_stop, initialize_directories, plot_png, load_textfile
import glob2

st.markdown("# Upload data to process with Bundlizer")
# st.markdown("")

# Set up basic directory structure for later code.
initialize_directories()

if 'sessionId' in st.session_state:
    sessionId = st.session_state['sessionId']
else:
    sessionId = ''
sessionId = st.text_input("SAMIS Session ID (get one here https://samprod.lpl.arizona.edu/sada):", value=sessionId)

if not sessionId:
    st.stop()
else:
    st.session_state['sessionId'] = sessionId

# Make a local directory to host the bundle -- clean up any old directory that could be present from before.
bundleDir = os.path.join("Output", sessionId)
if os.path.exists(bundleDir):
    shutil.rmtree(bundleDir)
os.makedirs(bundleDir)

# Make a local directory to host the user's raw data that will go into the bundle.
rawDir = os.path.join("Raw", sessionId)
if not os.path.exists(rawDir):
    os.makedirs(rawDir)

# Get a zip file from the user with his raw files, and unzip them on the server.
uploadFile = st.file_uploader(f"Please upload a zip file containing instrument data.") #, type=["zip"])
# st.write(uploadFile)

if uploadFile is None:
    st.stop()

# Unzip the data we are going to process if this was just uploaded.
# if st.session_state['uploadFile'] is not None:
if 'uploadFile' not in st.session_state or st.session_state['uploadFile'] != uploadFile:
    # Clean out the raw dir (upload overwrites it)
    shutil.rmtree(rawDir)
    with zipfile.ZipFile(BytesIO(uploadFile.read()), 'r') as zip_ref:
        zip_ref.extractall(rawDir)  # extracts all files into the bundle directory
    # Note that this has been uploaded so that we don't try to unzip again on a page reprocess.
    st.session_state['uploadFile'] = uploadFile

# Check if there is already a bundle file.
bundleinfoFileName = os.path.join(rawDir, f'{sessionId}_bundleinfo.yaml')
if os.path.exists(bundleinfoFileName):
    st.write(f'{sessionId}_bundleinfo.yaml was found in the uploaded info.')
    if st.button('Edit bundle'):
        switch_page("edit bundle")
else:
    st.write(f'{sessionId}_bundleinfo.yaml was not found in the uploaded info.')
    if st.button('Create a new bundle from this data'):
        switch_page("create bundle")