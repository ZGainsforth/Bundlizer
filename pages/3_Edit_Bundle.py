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

st.markdown("# View/Edit bundle file and raw products")
st.markdown("The user can view a data bundle and add/remove products here.")

# Set up basic directory structure for later code.
initialize_directories()

if 'sessionId' in st.session_state:
    sessionId = st.session_state['sessionId']
else:
    sessionId = ''
sessionId = st.text_input("SAMIS Session ID (get one here https://samis.lpl.arizona.edu):", value=sessionId)

# Check that the local directory structure exists.  If not the files haven't been uploaded first.
bundleDir = os.path.join("Output", sessionId)
rawDir = os.path.join("Raw", sessionId)
if not os.path.exists(bundleDir):
    st.write('Local files do not exist for this session ID yet.  Please use the upload data page first.')
    if st.button('Go to upload data.'):
        switch_page("upload data")
    st.stop()

# Define the path for the output file
bundleinfoFileName = os.path.join(bundleDir, f'{sessionId}_bundleinfo.yaml')

# Check that the bundle file exists.  If not, then the user needs to make it first.
if not os.path.exists(bundleinfoFileName):
    st.write("Bundle info file doesn't exist yet.  The bundle should be created first.")
    if st.button('Go to create bundle.'):
        switch_page("create bundle")
    st.stop()

# Make a# Now collect all the data products into a dictionary with format
# {'productName': {'Include':True,          # whether to include this data product in the bundle.
#                  ['path\to\file.tif',     # all the files in the data product.
#                   'path\to\file.png'...
#                  ]}}
productsDict = {}

# Find all the yaml files.
yamlFiles = glob2.glob(os.path.join(rawDir, '**', '*.yaml'))

# Using the yamls find all the product files and add them to productsDict
for yamlFile in yamlFiles:
    fullName, _ = os.path.splitext(yamlFile)
    productId = os.path.basename(fullName) # This is the ID that matches the raw data source.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    # All files in the product are going to have the same name as the yaml but different extensions.
    matchingFiles = glob2.glob(os.path.join(pathName, productId) + '.*')
    # matchingFiles = [file for file in matchingFiles if file != yamlFile]

    productsDict[productId] = {'Include': True, 'Files': matchingFiles}

# Print the resulting dictionary
for productId, productInfo in productsDict.items():
    st.markdown('<HR>', unsafe_allow_html=True)
    st.write(f"Product: {productId}")
    productInfo['Include'] = st.checkbox('Include in bundle?', value=productInfo['Include'], key=f'include{productId}')
    # st.write(f"\tInclude in bundle?: {productInfo['Include']}")
    for f in productInfo['Files']:
        st.write(f"\t{f}")
        _, ext = os.path.splitext(f)
        if ext in ['.png', '.jpg', '.jpeg', '.gif']:
            st.image(f)
        if ext == '.yaml':
            productInfo['yamlContent'] = st.text_area(label=f'{f}:', value=load_textfile(f), key=f)
        shutil.copyfile(f, os.path.join(bundleDir, os.path.basename(f)))

# Zip the resulting raw data + bundle files up.
rawPlusBundleZip = os.path.join(rawDir, '..', f'{sessionId}')
shutil.make_archive(rawPlusBundleZip, 'zip', rawDir)
with open(rawPlusBundleZip+'.zip', 'rb') as f:
    st.download_button('Download raw data + bundle files', data=f, file_name=f'{sessionId}_plus_raw.zip')

# Zip the resulting bundle files only.
bundleZip = os.path.join(bundleDir, '..', f'{sessionId}')
shutil.make_archive(bundleZip, 'zip', bundleDir)
with open(bundleZip+'.zip', 'rb') as f:
    st.download_button('Download bundle file only', data=f, file_name=f'{sessionId}.zip')
bundleDir = os.path.join("Output", sessionId)
if os.path.exists(bundleDir):
    shutil.rmtree(bundleDir)
os.makedirs(bundleDir)
