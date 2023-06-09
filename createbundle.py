import streamlit as st
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

st.markdown("# Bundlizer")
st.markdown("Convert raw instrument files to bundles that can be uploaded to SAMIS.")

# Set up basic directory structure for later code.
initialize_directories()

sessionId = None
sessionId = st.text_input("SAMIS Session ID (get one here https://samis.lpl.arizona.edu):")

# Select a BDD and instrument.
bdd = pd.read_csv(os.path.join('Config', 'BDD.csv'))
# Load people data
people = pd.read_csv(os.path.join('Config', 'People.csv'))

title = st.text_input("Title")
abstract = st.text_input("Abstract")

# BDD
bddName = st.selectbox("Select BDD", bdd['bddName'].unique())
bddSubset = bdd[bdd['bddName'] == bddName]
analysisTechniqueIdentifier = bddSubset['analysisTechniqueIdentifier'].iloc[0]

# Instrument
instrumentName = st.selectbox("Select instrument", bddSubset['instrument'])
bddVersion = bddSubset[bddSubset['instrument']==instrumentName]['bddVersion'].iloc[0]

defaultUser = people[people['name'] == 'None'].index[0]
dataBundleCreator = st.selectbox('Select Data Bundle Creator', people['name'], index=int(defaultUser))
instrumentOperator = st.selectbox('Select Instrument Operator', people['name'], index=int(defaultUser))
dataAnalyst = st.selectbox('Select Data Analyst', people['name'], index=int(defaultUser))

funding = st.text_input("Funding")

# Streamlit code to get user input in (X, Y) format
fiducialNorth = st.text_input("Enter North coordinates in (X, Y) format")
fiducialNorthX, fiducialNorthY = get_coordinates(fiducialNorth)
fiducialEast = st.text_input("Enter East coordinates in (X, Y) format")
fiducialEastX, fiducialEastY = get_coordinates(fiducialEast)
fiducialWest = st.text_input("Enter West coordinates in (X, Y) format")
fiducialWestX, fiducialWestY = get_coordinates(fiducialWest)
fiducialUnits = st.selectbox('Select fiducial units', ['um','mm', 'cm','m'])

# Now put all the data together to make a the bundle yaml
data = {
    "sessionId": sessionId,
    "analysisTechniqueIdentifier": analysisTechniqueIdentifier,
    "title": title,
    "abstract": abstract,
    "dataBundleCreator": [
        {
            "email": people.loc[people['name'] == dataBundleCreator, 'email'].iloc[0],
            "name": dataBundleCreator,
        }
    ],
    "instrumentOperator": [
        {
            "email": people.loc[people['name'] == instrumentOperator, 'email'].iloc[0],
            "name": instrumentOperator,
        }
    ],
    "dataAnalyst": [
        {
            "email": people.loc[people['name'] == dataAnalyst, 'email'].iloc[0],
            "name": dataAnalyst,
        }
    ],
    "bddVersion": bddVersion,
    "funding": funding,
    "fiducialNorthX": fiducialNorthX,
    "fiducialNorthY": fiducialNorthY,
    "fiducialEastX": fiducialEastX,
    "fiducialEastY": fiducialEastY,
    "fiducialWestX": fiducialWestX,
    "fiducialWestY": fiducialWestY,
    "fiducialUnits": fiducialUnits,
}

# Make a local directory to host the bundle.
bundleDir = os.path.join("Output", sessionId)
if not os.path.exists(bundleDir):
    os.makedirs(bundleDir)
add_cleanup_action('rmtree', bundleDir)

# Make a local directory to host the user's raw data that will go into the bundle.
rawDir = os.path.join("Raw", sessionId)
if not os.path.exists(rawDir):
    os.makedirs(rawDir)
add_cleanup_action('rmtree', rawDir)

# Define the path for the output file
bundleinfoFileName = os.path.join(bundleDir, f'{sessionId}_bundleinfo.yaml')

# Get a zip file from the user with his raw files, and unzip them on the server.
uploadFile = st.file_uploader(f"Please upload a zip file containing {analysisTechniqueIdentifier} data from {instrumentName}.") #, type=["zip"])

# Don't process the data until the user says to.
if not st.button('Process raw data now!'):
    cleanup_and_stop()

# Write the data to a YAML file
if sessionId == '':
    st.write('Cannot create bundle file without a sessionId.')
    cleanup_and_stop()
st.write(f'Generating bundle yaml file: {bundleinfoFileName}')
with open(bundleinfoFileName, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# Unzip the data we are going to process
if uploadFile is not None:
    with zipfile.ZipFile(BytesIO(uploadFile.read()), 'r') as zip_ref:
        zip_ref.extractall(rawDir)  # extracts all files into the bundle directory
else:
    cleanup_and_stop()

# Make a progress bar.  We will be updating this during the processing step since it takes a while.
instrumentProgress = st.progress(0.0)

# Now we are going to import the appropriate python script to do raw data processing for this instrument.
sanitizeRegex = r"\s+|-|\." # selects for any whitespace, dash or period.
instrumentModuleName = f"BundleProcessors.{re.sub(sanitizeRegex, '_', analysisTechniqueIdentifier)}.{re.sub(sanitizeRegex, '_', instrumentName)}" 
instrumentProcessor = importlib.import_module(instrumentModuleName)
instrumentProcessor = importlib.reload(instrumentProcessor)
_ = instrumentProcessor.preprocess_all_products(rawDir, sessionId=sessionId, statusOutput=nuke_text, statusProgress=instrumentProgress)

# Now collect all the data products into a dictionary with format
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
        if ext == '.png':
            plot_png(f)
        if ext == '.yaml':
            productInfo['yamlContent'] = st.text_area(label=f'{f}:', value=load_textfile(f), key=f)