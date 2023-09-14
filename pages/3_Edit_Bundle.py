import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_ace import st_ace
import os,sys,shutil
# import pandas as pd
import yaml
import zipfile
import re
from helperfuncs import get_coordinates, nuke_text, add_cleanup_action, cleanup_and_stop, initialize_directories, plot_png, load_textfile, load_instrument_processor
import glob2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import helperfuncs as hf
import hyperspy.api as hs

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

    # If this is the bundleinfo file, then it should be added as a product.
    if os.path.basename(yamlFile) == f'{sessionId}_bundleinfo.yaml':
        productsDict['bundleinfo'] = {'Expander': None, 'Include': True, 'Files': [yamlFile], 'EditText': {}, 
                               'sessionId': sessionId, 'productId': 'bundleinfo', 'dataComponentType': None}

    # Read the YAML file to check for the presence of the dataComponentType field
    with open(yamlFile, 'r') as file:
        yaml_content = yaml.safe_load(file)
        yamlDataComponentType = yaml_content.get('dataComponentType', None)

    # Get the sessionId and productId out of the yaml name.
    pattern = re.compile(rf'({re.escape(sessionId)})_(.*)_(\d+)')
    match = pattern.match(os.path.basename(fullName))
    if match:
        yamlsessionId, _, productId = match.groups()
    else:
        if os.path.basename(yamlFile) != f'{sessionId}_bundleinfo.yaml':
            st.write(f'cannot find sessionId, productId in yaml name: {yamlFile}')
        continue

    if yamlsessionId != sessionId:
        st.write(f'Invalid session ID from yaml name: sessionId={yamlsessionId}, fileName={yamlFile}')

    # st.write(f'Processing product {productId} from yaml: {os.path.basename(yamlFile)}')

    # Construct the regular expression pattern to find all the files in this product.
    # \d+ matches one or more digits, and .* matches any characters (except a newline).
    pattern = re.compile(rf"{yamlsessionId}_.*_{productId}.*")
    matchingFiles = []
    for filename in allFiles :
        if pattern.match(os.path.basename(filename)):
            matchingFiles.append(filename)
            # st.write(f'\tAdding {filename}.')
    matchingFiles = sorted(matchingFiles)

    # Expander will point to the GUI element which displays all the info about this product on the page.
    # Include is whether this particular product will be included in the bundle.
    # Files are the files that are part of the product.
    # EditText is a dictionary containing a list of editable strings that are the text files in this product (yamls, txt, etc.)
    # EditText keys are the filenames so we can write the edited text back.
    productsDict[productId] = {'Expander': None, 'Include': True, 'Files': matchingFiles, 'EditText': {}, 
                               'sessionId': sessionId, 'productId': productId, 'dataComponentType': yamlDataComponentType}

productsDict = dict(sorted(productsDict.items()))

@st.cache_data
def draw_msa(f):
    msa = hs.load(f)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(msa.axes_manager[0].axis, msa.data)
    ax.set_xlabel(msa.axes_manager[0].units)
    ax.set_ylabel(msa.metadata.Signal.quantity)
    st.pyplot(fig)

@st.cache_data
def draw_tiff(f):
    img = tifffile.imread(f)
    # If this is an image stack, then average into an image for display.
    if len(img.shape) > 2:
        img = np.mean(img, axis=0)
    fig = plt.figure()
    mean = np.mean(img)
    std = np.std(img)
    plt.imshow(img, cmap='gray', vmin=mean-2*std, vmax=mean+2*std)
    plt.colorbar()
    # plt.imshow(img)
    st.pyplot(fig)

@st.cache_data
def draw_image(f):
    st.image(f)

@st.cache_data
def draw_emd(f):
    fig = st.session_state['instrumentProcessor'].plot_emd(f)
    st.pyplot(fig)

def copy_file_to_output(f, bundleDir, productId, productInfo):
    # Most files in the bundle just get copied to the root of the bundle.
    # Collections are the exception.
    if 'Collection' not in f:
        shutil.copyfile(f, os.path.join(bundleDir, os.path.basename(f)))
        return

    # It is a collection.  There should be a yaml and a directory with files in it.
    # We will only process this if it is the yaml.
    RootName, ext = os.path.splitext(f)
    if (ext != '.yaml') and (ext != '.yml'):
        # It's not a yaml so move on.  This file will get collected when we handle the yaml.
        return
    # One more check: is there a subdirectory with the same name (there could be yamls inside the collection).
    if not os.path.isdir(RootName):
        # There isn't a directory with the same name so it isn't a yaml describing a collection.  It's a yaml *in* a collection.
        return

    # This is the yaml so we will zip up the directory with the same name.
    # RootName = os.path.abspath(RootName)
    shutil.make_archive(RootName, 'zip', RootName)
    # hf.zip_directory(os.path.join(RootName, '..'), RootName)
    # And we copy both the yaml and the zip to the output.
    shutil.copyfile(f, os.path.join(bundleDir, os.path.basename(f)))
    shutil.copyfile(RootName + '.zip', os.path.join(bundleDir, os.path.basename(RootName))+'.zip')

# We loop through all the products putting each on the screen.  Each product will be placed in an expander.
for productId, productInfo in productsDict.items():
    # if productId == 'bundleinfo':
    #     # The main bundle info file is a special case -- it already has an editor above.
    #     copy_file_to_output(f, bundleDir, productId, productInfo)
    #     continue
    productInfo['Expander'] = st.expander(f'Product: {productId}', expanded=True)
    with productInfo['Expander'] as ex:
        # Future option to include include checkbox.  For now probably not necessary.
        # productInfo['Include'] = st.checkbox('Include in bundle?', value=productInfo['Include'], key=f'include{productId}')
        for f in productInfo['Files']:
            # st.write(f"\t{f}")
            _, ext = os.path.splitext(f)
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                st.write(f"\t{os.path.basename(f)}")
                draw_image(f)
            if ext == '.tif':
                st.write(f"\t{os.path.basename(f)}")
                draw_tiff(f)
            if ext == '.msa':
                st.write(f"\t{os.path.basename(f)}")
                draw_msa(f)
            if ext == '.emd':
                st.write(f"\t{os.path.basename(f)}")
                draw_emd(f)
            if ext == '.yaml':
                if productId != 'bundleinfo':
                    # The main bundle info file is a special case -- it already has an editor above.
                    productInfo['EditText'][f] = productInfo['Expander'].text_area(label=f'{os.path.basename(f)}:', value=load_textfile(f), key=f)
            if ext == '.txt':
                productInfo['EditText'][f] = productInfo['Expander'].text_area(label=f'{os.path.basename(f)}:', value=load_textfile(f))
            # shutil.copyfile(f, os.path.join(bundleDir, os.path.basename(f)))
            copy_file_to_output(f, bundleDir, productId, productInfo)

# TODO: Update yamls and text files with edits the user made.

# def create_archive(input_dir, output_zip):
#     command = f"7z a -tzip -mx=9 -mmt={os.cpu_count()} {output_zip} {input_dir}"
#     subprocess.call(command, shell=True)

def create_archive(bundleDir, sessionId):
    for root, dirs, _ in os.walk(os.path.join(bundleDir, f'{sessionId}')):
        for dir in dirs:
            if dir != root:
                hf.zip_directory(root, dir)
                shutil.rmtree(os.path.join(root, dir))
    hf.zip_directory(bundleDir, f'{sessionId}')

if st.button('Prepare bundle file for download.'):
    # Zip the resulting bundle files only.
    bundleZip = os.path.join(bundleDir, '..', f'{sessionId}')
    # shutil.make_archive(bundleZip, 'zip', bundleDir)
    with st.spinner('Preparing... This may take a minute.'):
        # create_archive(bundleDir, bundleZip)
        create_archive(os.path.normpath(os.path.join(bundleDir,'..')), sessionId)
    with open(bundleZip+'.zip', 'rb') as f:
        st.download_button('Download bundle file', data=f, file_name=f'{sessionId}.zip')

# if st.button('Prepare raw data + bundle files for download.'):
#     st.write('Preparing... This may take a minute.')
#     # Zip the resulting raw data + bundle files up.
#     rawPlusBundleZip = os.path.join(rawDir, '..', f'{sessionId}')
#     # shutil.make_archive(rawPlusBundleZip, 'zip', rawDir)
#     create_archive(rawDir, rawPlusBundleZip)
#     with open(rawPlusBundleZip+'.zip', 'rb') as f:
#         st.download_button('Download raw data + bundle files', data=f, file_name=f'{sessionId}_plus_raw.zip')
