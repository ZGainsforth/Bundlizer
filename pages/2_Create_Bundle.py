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
from helperfuncs import get_coordinates, nuke_text, add_cleanup_action, cleanup_and_stop, initialize_directories, plot_png, load_textfile, load_instrument_processor
import glob2

st.markdown("# Create a bundle from raw data")
st.markdown("Convert raw instrument files to bundles that can be uploaded to SAMIS.")

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

# Select a BDD and instrument.
bdd = pd.read_csv(os.path.join('Config', 'BDD.csv'))
# Load people data
people = pd.read_csv(os.path.join('Config', 'People.csv'))

title = st.text_input("Title")
abstract = st.text_area("Abstract")

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
    "instrumentName": instrumentName,
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
}

if fiducialNorthX is not None:
    data.update({
        "fiducialNorthX": fiducialNorthX,
        "fiducialNorthY": fiducialNorthY,
        "fiducialEastX": fiducialEastX,
        "fiducialEastY": fiducialEastY,
        "fiducialWestX": fiducialWestX,
        "fiducialWestY": fiducialWestY,
        "fiducialUnits": fiducialUnits,
    })

# Define the path for the output file
bundleinfoFileName = os.path.join(rawDir, f'{sessionId}_bundleinfo.yaml')

# Don't process the data until the user says to.
if not st.button('Process raw data now!'):
    st.stop()

# Write the data to a YAML file
if sessionId == '':
    st.write('Cannot create bundle file without a sessionId.')
    st.stop()
st.write(f'Generating bundle yaml file: {bundleinfoFileName}')
with open(bundleinfoFileName, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# Make a progress bar.  We will be updating this during the processing step since it takes a while.
instrumentProgress = st.progress(0.0)

# Preprocess the data in the uploaded zip to produce the bundle file products.
instrumentProcessor  = load_instrument_processor(data)
instrumentProcessor.preprocess_all_products(rawDir, sessionId=sessionId, statusOutput=nuke_text, statusProgress=instrumentProgress)

# Now move to the next page to view the results.
switch_page('edit bundle')