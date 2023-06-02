import streamlit as st
import os,sys,shutil
import pandas as pd
from collections import OrderedDict
import yaml

st.markdown("# Bundlizer")
st.markdown("Convert raw instrument files to bundles that can be uploaded to SAMIS.")

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

# Convert (X,Y) format to two floats for getting fiducials.
def get_coordinates(coordinatesInput):
    try:
        coords = coordinatesInput.strip("()").split(",")
        x, y = float(coords[0]), float(coords[1])
        return x, y
    except:
        st.write(f"Unable to parse coordinates: '{coordinatesInput}'. Please enter coordinates in the format (X, Y)")
        return None, None

# Streamlit code to get user input in (X, Y) format
fiducialNorth = st.text_input("Enter North coordinates in (X, Y) format")
fiducialNorthX, fiducialNorthY = get_coordinates(fiducialNorth)
fiducialEast = st.text_input("Enter East coordinates in (X, Y) format")
fiducialEastX, fiducialEastY = get_coordinates(fiducialEast)
fiducialWest = st.text_input("Enter West coordinates in (X, Y) format")
fiducialWestX, fiducialWestY = get_coordinates(fiducialWest)
fiducialUnits = st.selectbox('Select fiducial units', ['um','mm', 'cm','m'])

# Now put all the data together to make a the bundle 
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

bundleDir = os.path.join("Output", sessionId)

# Check if the session directory exists, if not, create it
if not os.path.exists(bundleDir):
    os.makedirs(bundleDir)

# Define the path for the output file
bundleinfoFileName = os.path.join(bundleDir, f'{sessionId}_bundleinfo.yaml')

# Write the data to a YAML file
st.write(f'Generating yaml file: {bundleinfoFileName}')
with open(bundleinfoFileName, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
