import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_ace import st_ace
import os,sys,shutil
import pandas as pd
from collections import OrderedDict
import yaml
import zipfile
from io import BytesIO
import importlib
import re
from helperfuncs import get_coordinates, nuke_text, add_cleanup_action, cleanup_and_stop, initialize_directories, plot_png, load_textfile, load_instrument_processor
import glob2
import tifffile
import matplotlib.pyplot as plt
import numpy as np

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
bundleinfoFileName = os.path.join(rawDir, f'{sessionId}_bundleinfo.yaml')

# Check that the bundle file exists.  If not, then the user needs to make it first.
if not os.path.exists(bundleinfoFileName):
    st.write("Bundle info file doesn't exist yet.  The bundle should be created first.")
    if st.button('Go to create bundle.'):
        switch_page("create bundle")
    st.stop()

bundleYaml = st_ace(value=load_textfile(bundleinfoFileName), language='yaml')

# We may need the instrument processor to produce nice images and outputs for the files that have been bundlized.
instrumentProcessor = load_instrument_processor(yaml.safe_load(bundleYaml))

# Make a# Now collect all the data products into a dictionary with format
# {'productName': {'Include':True,          # whether to include this data product in the bundle.
#                  ['path\to\file.tif',     # all the files in the data product.
#                   'path\to\file.png'...
#                  ]}}
productsDict = {}

# Find all the yaml files.
yamlFiles = sorted(glob2.glob(os.path.join(rawDir, '**', '*.yaml')))
# In addition we need a list of all the files.
allFiles = glob2.glob(os.path.join(rawDir, '**', '*'))

# Using the yamls, find all the product files and add them to productsDict
for yamlFile in yamlFiles:
    fullName, _ = os.path.splitext(yamlFile)
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    # Get the sessionId and productId out of the yaml name.
    pattern = re.compile(r'(\d+)_.*_(\d+)')
    match = pattern.match(os.path.basename(fullName))
    if match:
        yamlsessionId, productId = match.groups()
    else:
        if os.path.basename(yamlFile) != f'{sessionId}_bundleinfo.yaml':
            st.write(f'cannot find sessionId, productId in yaml name: {yamlFile}')
        continue
    
    if yamlsessionId != sessionId:
        st.write(f'Invalid session ID from yaml name: sessionId={yamlsessionId}, fileName={yamlFile}')

    # Construct the regular expression pattern to find all the files in this product.
    # \d+ matches one or more digits, and .* matches any characters (except a newline).
    pattern = re.compile(rf"\d+_.*_{productId}\..*")

    matchingFiles = []

    for filename in allFiles :
        if pattern.match(os.path.basename(filename)):
            matchingFiles.append(filename)

    matchingFiles = sorted(matchingFiles)

    # Expander will point to the GUI element which displays all the info about this product on the page.
    # Include is whether this particular product will be included in the bundle.
    # Files are the files that are part of the product.
    # EditText is a dictionary containing a list of editable strings that are the text files in this product (yamls, txt, etc.)
    # EditText keys are the filenames so we can write the edited text back.
    productsDict[productId] = {'Expander': None, 'Include': True, 'Files': matchingFiles, 'EditText': {}}

productsDict = dict(sorted(productsDict.items()))

# We loop through all the products putting each on the screen.  Each product will be placed in an expander.
for productId, productInfo in productsDict.items():
    if productId == f'{sessionId}_bundleinfo.yaml':
        # The main bundle info file is a special case -- it already has an editor above.
        continue
    productInfo['Expander'] = st.expander(f'Product: {productId}', expanded=True)
    with productInfo['Expander'] as ex:
        productInfo['Include'] = st.checkbox('Include in bundle?', value=productInfo['Include'], key=f'include{productId}')
        for f in productInfo['Files']:
            # st.write(f"\t{f}")
            _, ext = os.path.splitext(f)
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                st.write(f"\t{os.path.basename(f)}")
                st.image(f)
            if ext == '.tif':
                st.write(f"\t{os.path.basename(f)}")
                img = tifffile.imread(f)
                fig = plt.figure()
                mean = np.mean(img)
                std = np.std(img)
                plt.imshow(img, cmap='gray', vmin=mean-2*std, vmax=mean+2*std)
                plt.colorbar()
                # plt.imshow(img)
                st.pyplot(fig)
            if ext == '.emd':
                fig = st.session_state['instrumentProcessor'].plot_emd(f)
                st.pyplot(fig)
            if ext == '.yaml':
                productInfo['EditText'][f] = productInfo['Expander'].text_area(label=f'{os.path.basename(f)}:', value=load_textfile(f), key=f)
            if ext == '.txt':
                productInfo['EditText'][f] = productInfo['Expander'].text_area(label=f'{os.path.basename(f)}:', value=load_textfile(f))
            shutil.copyfile(f, os.path.join(bundleDir, os.path.basename(f)))

if st.button('Prepare bundle files for download.'):
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
