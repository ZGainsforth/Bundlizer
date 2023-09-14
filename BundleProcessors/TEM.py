# Created 2023, Zack Gainsforth
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import hyperspy.api as hs
import pandas as pd
import re, os, io, gc
import shutil
import glob2
import yaml
from yaml.representer import SafeRepresenter
from ncempy.io import dm, ser # Reader for dm3/dm4 files and ser (TIA) files.
import importlib
hf = importlib.import_module('helperfuncs', '..')
import json
import h5py
import datetime

# For unique product ID's we start a counter.
productId = 0

def new_productId():
    global productId
    productId += 1
    return productId

# These file extensions indicate data types that we can convert 
raw_extensions = [
                  'dm3', # Gatan files
                  'dm4',
                  'ser', # TIA files.
                  'bcf', # Bruker EDS files.
                  'h5oina', # Oxford EDS files.
                  'msa', # MSA/EMSA files.
                  'emsa', # MSA/EMSA files.
                  ]

# Figure out if this is a TEM image, a STEM image or what -- for ser files.
def get_image_type(metadata=None):
    match str(metadata['Mode []']):
        case value if 'STEM' in value:
            return 'STEM'
        case _:
            raise ValueError('Cannot guess image type from metadata.  Please expand case statement.')

def plot_emd(fileName):
    with h5py.File(fileName, 'r') as emd:
        # Load EDS data
        eds_data = np.array(emd['data']['EDS']['EDS'])

        # Get axes information
        dim1 = emd['data']['EDS']['dim1']
        dim2 = emd['data']['EDS']['dim2']
        dim3 = emd['data']['EDS']['dim3']
        # dim1 = np.array(emd['data']['EDS']['dim1'])[:, 0]
        # dim2 = np.array(emd['data']['EDS']['dim2'])[:, 0]
        # dim3 = np.array(emd['data']['EDS']['dim3'])[:, 0]
        
        for n,d in {0:dim1, 1:dim2, 2:dim3}.items():
            match d.attrs['name'].decode('utf-8'):
                case 'Width':
                    x = n; xdim = d
                case 'Height':
                    y = n; ydim = d
                case 'Energy':
                    e = n; edim = d
                case _:
                    raise ValueError(f"Invalid dimension name {d.attrs['name'].decode('utf-8')}.")

        # Create sum over energy axis (2D image)
        sum_image = np.sum(eds_data, axis=e)

        # Create mean and max over spatial dimensions (spectrum)
        spectrum = np.mean(eds_data, axis=(x,y))
        spectrum_max = np.max(eds_data, axis=(x,y))

        # Create the figure and axes using plt.subplots
        fig, axs = plt.subplots(2, 1, figsize=(6, 9)) 

        # Plot 2D image
        im = axs[0].imshow(sum_image, cmap='gray', extent=[ydim[0][0], ydim[-1][0], xdim[0][0], xdim[-1][0]])
        axs[0].set_title('EDS stack sum image')
        axs[0].set_ylabel(ydim.attrs['name'].decode() + ' (' + ydim.attrs['units'].decode() + ')')
        axs[0].set_xlabel(xdim.attrs['name'].decode() + ' (' + xdim.attrs['units'].decode() + ')')

        # Plot spectrum
        axs[1].plot(edim, spectrum)
        axs[1].plot(edim, spectrum_max)
        axs[1].set_title('EDS stack spectrum')
        axs[1].set_xlabel(edim.attrs['name'].decode() + ' (' + edim.attrs['units'].decode() + ')')
        axs[1].set_ylabel('Counts')
        axs[1].legend(['Mean value', 'Max value'])

        plt.tight_layout()

    return fig

def write_TEM_image(fileName=None, sessionId=None, statusOutput=print, img=None, core_metadata=None, addl_metadata=None):
    productId = new_productId()

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        "pixelScaleX": core_metadata['PhysicalSizeX'],
        "pixelScaleY": core_metadata['PhysicalSizeY'],
        "pixelUnits": core_metadata['PhysicalSizeXUnit'],
        "channel1": "Intensity"
        }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.ome.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    # Write an OME-TIF with all the metadata, pixel scales, etc.
    hf.write_ome_tif_image(fileName, sessionId, productId, img, core_metadata, addl_metadata)

    # If this is an electron diffraction pattern, then there is supposed to be a PDF for the calibration.
    if core_metadata['dataComponentType'] == 'TEMPatternsImage':
        calibrationFileName = os.path.splitext(fileName)[0] + ".pdf"
        try:
            shutil.copyfile(calibrationFileName, f"{sessionId}_calibrationFile_{productId:05d}.ome.pdf")
        except Exception as e:
            statusOutput(f'Could not find calibration file {calibrationFileName}.')

    return

def preprocess_ser(fileName=None, sessionId=None, statusOutput=print, file=None, samisData=None):
    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')
    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": 'STEMImage',
        "PhysicalSizeX": float(file['pixelSize'][0]),
        "PhysicalSizeXUnit": hf.replace_greek_symbols(str(file['pixelUnit'][1])),
        "PhysicalSizeY": float(file['pixelSize'][1]),
        "PhysicalSizeYUnit": hf.replace_greek_symbols(str(file['pixelUnit'][0])),
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    write_TEM_image(fileName=fileName, sessionId=sessionId, statusOutput=statusOutput, img=file['data'][:,:].astype('float32'), core_metadata=core_metadata, addl_metadata=file['metadata'])
    return

def preprocess_dm(fileName=None, sessionId=None, statusOutput=print, file=None, samisData=None):
    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    if '/' in file.axes_manager['y'].units:
        dataComponentType = 'TEMPatternsImage'
    else:
        dataComponentType = 'TEMImage'

    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": dataComponentType,
        "PhysicalSizeX": float(file.axes_manager['x'].scale),
        "PhysicalSizeXUnit": hf.replace_greek_symbols(file.axes_manager['x'].units),
        "PhysicalSizeY": float(file.axes_manager['y'].scale),
        "PhysicalSizeYUnit": hf.replace_greek_symbols(file.axes_manager['y'].units),
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    write_TEM_image(fileName=fileName, sessionId=sessionId, statusOutput=statusOutput, img=file.data.astype('float32'), core_metadata=core_metadata, addl_metadata=file.metadata.as_dictionary())
    return

# Preprocess Bruker EDS cube.
def preprocess_bcf(fileName=None, sessionId=None, statusOutput=print, haadf=None, eds=None, samisData=None):
    productId = new_productId()

    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    core_metadata = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": 'STEMEDSCube',
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        "dimensions": [] # This will be populated as we go.
        }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    # We don't actually save the yaml yet, because it needs some info from the bcf populated into it.

    # Create emd file from bcf.
    emdFileName = os.path.join(os.path.dirname(fileName), f'{productName}.emd')
    emd = h5py.File(emdFileName, 'w')
    emd.attrs['version_major'] = 0
    emd.attrs['version_minor'] = 2
    emd_data = emd.create_group('data')

    emd_eds = emd_data.create_group('EDS')
    emd_eds.attrs['emd_group_type'] = 1
    eds_data = emd_eds.create_dataset('EDS', eds.data.shape, dtype='float', compression='gzip', compression_opts=7)
    eds_data[:] = eds.data

    dim = emd_eds.create_dataset(f'dim1', (eds.data.shape[0],1))
    dim[:,0] = eds.axes_manager['height'].axis
    dim.attrs['name'] = np.string_('Height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['height'].units))
    yamlData['dimensions'].append({'fieldDescription':dim.attrs['name'].decode('utf-8'), 'unitOfMeasure': dim.attrs['units'].decode('utf-8')})

    dim = emd_eds.create_dataset(f'dim2', (eds.data.shape[1],1))
    dim[:,0] = eds.axes_manager['width'].axis
    dim.attrs['name'] = np.string_('Width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['width'].units))
    yamlData['dimensions'].append({'fieldDescription':dim.attrs['name'].decode('utf-8'), 'unitOfMeasure': dim.attrs['units'].decode('utf-8')})

    dim = emd_eds.create_dataset(f'dim3', (eds.data.shape[2],1))
    dim[:,0] = eds.axes_manager['Energy'].axis
    dim.attrs['name'] = np.string_('Energy')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['Energy'].units))
    yamlData['dimensions'].append({'fieldDescription':dim.attrs['name'].decode('utf-8'), 'unitOfMeasure': dim.attrs['units'].decode('utf-8')})

    eds_metadata = emd_eds.create_group('microscope') # metadata for haadf image
    for k, v in core_metadata.items():
        eds_metadata.attrs[k] = v
    for k, v in hf.flatten_dict(eds.metadata.as_dictionary()).items():
        if type(v) in [bool, str, int, float]:
            eds_metadata.attrs[k] = v

    emd_haadf = emd_data.create_group('HAADF')
    emd_haadf.attrs['emd_group_type'] = 1
    haadf_data = emd_haadf.create_dataset('HAADF', haadf.data.shape, dtype='float', compression='gzip', compression_opts=7)
    haadf_data[:] = haadf.data

    dim = emd_haadf.create_dataset(f'dim1', (haadf.data.shape[0],1))
    dim[:,0] = haadf.axes_manager['Height'].axis
    dim.attrs['name'] = np.string_('Height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(haadf.axes_manager['height'].units))

    dim = emd_haadf.create_dataset(f'dim2', (haadf.data.shape[1],1))
    dim[:,0] = haadf.axes_manager['Width'].axis
    dim.attrs['name'] = np.string_('width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(haadf.axes_manager['width'].units))

    haadf_metadata = emd_haadf.create_group('microscope') # metadata for haadf image
    for k, v in core_metadata.items():
        haadf_metadata.attrs[k] = v
    for k, v in hf.flatten_dict(haadf.metadata.as_dictionary()).items():
        if type(v) in [bool, str, int, float]:
            haadf_metadata.attrs[k] = v

    emd.close()

    # Save the yaml now that we have all the data.
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    return

# Converts all the datasets in the current node (non-recursively) into a dictionary.
def h5_node_to_dict(node):
    nodeDict = {}
    for key in node.keys():
        # Don't recurse.
        if isinstance(node[key], h5py.Group):
            nodeDict[key] = h5_node_to_dict(node[key])
            continue

        # Add this dataset into the dictinary
        if node[key].dtype == object:
            nodeDict[key] = node[key][0].decode('utf-8')
        else:
            nodeDict[key] = np.array(node[key])[0].item()
    return hf.flatten_dict(nodeDict)

# Process Oxford EDS info.
def preprocess_h5oina(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    productId = new_productId()
    baseName = os.path.basename(fileName)
    fileNameNoExt, _ = os.path.splitext(fileName)

    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    core_metadata = { 
        "description": f"{baseName}: {description}" if description else baseName,
        "dataComponentType": 'STEMEDSCube',
        }
    # Add any metadata from samisData for this product.
    hf.union_dict_no_overwrite(core_metadata, hf.sanitize_dict_for_yaml(samisDict))

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        "dimensions": [] # This will be populated as we go.
        }
    # We don't actually save the yaml yet, because it needs some info populated into it.

    # Load the data
    rpl = hs.load(f'{fileNameNoExt}.rpl')

    # Open the h5oina file which we need for metadata.
    h5oina = h5py.File(fileName, 'r+')

    # Extract metadata from the .h5oina file
    edsHeader = h5oina['1']['EDS']['Header']
    # edsMetadata = {key: edsHeader[key][()] for key in edsHeader.keys()}
    edsMetadata = h5_node_to_dict(edsHeader)
    imageHeader = h5oina['1']['Electron Image']['Header']
    imageMetadata = h5_node_to_dict(imageHeader)
    # imageMetadata = {key: imageHeader[key][()] for key in imageHeader.keys()}

    # Create emd file from bcf.
    emdFileName = os.path.join(os.path.dirname(fileName), f'{productName}.emd')
    emd = h5py.File(emdFileName, 'w')
    emd.attrs['version_major'] = 0
    emd.attrs['version_minor'] = 2
    emd_data = emd.create_group('data')

    emd_eds = emd_data.create_group('EDS')
    emd_eds.attrs['emd_group_type'] = 1
    eds_data = emd_eds.create_dataset('EDS', rpl.data.shape, dtype='float', compression='gzip', compression_opts=7)
    eds_data[:] = rpl.data

    # Can I just say that it is shocking that Oxford does not export the units for their data cubes?  I sure hope they are always um x ux x eV...
    dim = emd_eds.create_dataset(f'dim1', (rpl.data.shape[0],1))
    dim[:,0] = [edsMetadata['Start Channel'] + i * edsMetadata['Channel Width'] for i in range(edsMetadata['Number Channels'])]
    dim.attrs['name'] = np.string_('Energy')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols('eV'))
    yamlData['dimensions'].append({'fieldDescription':dim.attrs['name'].decode('utf-8'), 'unitOfMeasure': dim.attrs['units'].decode('utf-8')})

    dim = emd_eds.create_dataset(f'dim2', (rpl.data.shape[1],1))
    dim[:,0] = np.linspace(0, edsMetadata['Y Step'], edsMetadata['Y Cells'])
    # dim[:,0] = np.linspace(0, np.array(h5oina['1']['EDS']['Header']['Y Step'][0]), np.array(h5oina['1']['EDS']['Header']['Y Cells'][0]))
    dim.attrs['name'] = np.string_('Height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols('um'))
    yamlData['dimensions'].append({'fieldDescription':dim.attrs['name'].decode('utf-8'), 'unitOfMeasure': dim.attrs['units'].decode('utf-8')})

    dim = emd_eds.create_dataset(f'dim3', (rpl.data.shape[2],1))
    dim[:,0] = np.linspace(0, edsMetadata['X Step'], edsMetadata['X Cells'])
    dim.attrs['name'] = np.string_('Width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols('um'))
    yamlData['dimensions'].append({'fieldDescription':dim.attrs['name'].decode('utf-8'), 'unitOfMeasure': dim.attrs['units'].decode('utf-8')})

    emd_edsMetadata = emd_eds.create_group('microscope') # metadata for eds
    for k, v in core_metadata.items():
        emd_edsMetadata.attrs[k] = v
    for k, v in edsMetadata.items():
        if type(v) in [bool, str, int, float]:
            emd_edsMetadata.attrs[k] = v

    emd.close()

    # Now we can save the yaml.    
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    return

def preprocess_msa(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    productId = new_productId()

    # We need to know if this msa file is an EDS spectrum or an EELS spectrum.
    with open(fileName, 'r') as f:
        msaText = f.read()
        if 'EDS' in msaText:
            dataComponentType = 'STEMEDSTabular'
        elif 'EELS' in msaText:
            dataComponentType = 'STEMEELSTabular'
        else:
            raise ValueError(f'Neither "EDS" nor "EELS" were found in {fileName}.')

        # Determine the number of header lines (excluding the #ENDOFDATA line)
        headerLines = re.findall(r'^#.*', msaText, re.MULTILINE)
        headerRowCount = len(headerLines) - 1 if '#ENDOFDATA   :' in msaText else len(headerLines)

        # Determine if the data section has one or two columns by inspecting the first line of actual data
        dataSection = re.findall(r'^-?\d+\.\d+,\s?-?\d*\.?\d*', msaText, re.MULTILINE)
        countColumns = len(dataSection[0].split(',')) if dataSection else 0
        if countColumns != 2:
            raise ValueError(f'{fileName} must have two columns of data to be valid.')

    productName = f"{sessionId}_{dataComponentType}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Get any information in the user's csv for SAMIS.
    samisDict = hf.samis_dict_for_this_file(samisData, fileName, statusOutput)
    description = samisDict.get('description', '')

    # In order to get the metadata for the yaml we need to read the msa.
    msa = hs.load(fileName)

    # Create and save the yaml file for this spectrum.
    yamlData = { 
        "description": f"{os.path.basename(fileName)}: {description}" if description else os.path.basename(fileName),
        "dataComponentType": dataComponentType,
        "headerRowCount":headerRowCount,
        "countColumns":countColumns,
        "countRows":msa.axes_manager[0].size,
        "columns": {
            "1": {
             "label":msa.axes_manager[0].units,
             "fieldDescription":"Energy axis",
             "fieldType":"decimal",
             "unitOfMeasure":msa.axes_manager[0].units,
            },
            "2": {
             "label":msa.metadata.Signal.quantity,
             "fieldDescription":"Intensity",
             "fieldType":"decimal",
             "unitOfMeasure":msa.metadata.Signal.quantity,
            }
        },
    }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    # Copy the msa file to the correct file name for packaging.
    shutil.copy(fileName, os.path.join(os.path.dirname(fileName), f'{productName}.msa'))

    return


def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print, samisData=None):
    # In the case of STXM, all products are pointed to by a hdr file.
    # Extract the file name.
    fullName, ext = os.path.splitext(fileName)
    baseName = os.path.basename(fullName) # This is the name of the file.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    match ext:
        case '.dm3' | '.dm4':
            file = hs.load(fileName)
            preprocess_dm(fileName, sessionId, statusOutput, file, samisData)
        case '.ser':
            file = ser.serReader(fileName)
            preprocess_ser(fileName, sessionId, statusOutput, file, samisData)
        case '.bcf':
            haadf, eds = hs.load(fileName)
            statusOutput(f'Stack dimensions are: {eds.data.shape}.')
            preprocess_bcf(fileName, sessionId, statusOutput, haadf, eds, samisData)
        case '.h5oina':
            preprocess_h5oina(fileName, sessionId, statusOutput, samisData)
        case '.msa' | '.emsa':
            preprocess_msa(fileName, sessionId, statusOutput, samisData)
        case _:
            raise ValueError(f"{ext} is an invalid data product type.")

    return

def preprocess_all_products(dirName=None, sessionId=None, statusOutput=print, statusProgress=None):
    # The user is telling us a directory which contains raw products from this instrument.
    if dirName is None:
        dirName =  os.getcwd()
    # Create a list of all files with extensions that we can process.
    rawFiles = []
    for ext in raw_extensions:
        rawFiles.extend(glob2.glob(os.path.join(dirName, '**', f'*.{ext}')))
    rawFiles.sort()

    # If there is a csv with additional metadata fields supplied by the user then load it.  Ususally this is used for descriptions.
    try:
        samisData = pd.read_csv(os.path.join(dirName, 'samisdata.csv'))
        statusOutput('Loaded metadata from samisdata.csv.')
    except:
        samisData = None
        statusOutput('There is no samisdata.csv.  Where are your descriptions going to come from?  Consider making a csv...')

    for i, f in enumerate(rawFiles):
        try:
            preprocess_one_product(f, sessionId=sessionId, statusOutput=statusOutput, samisData=samisData)
            statusOutput(f'Preprocessed {f}\n')
        except Exception as e:
            # If that one Xim failed, go on and process the next, it won't be included in the data products.
            statusOutput(f'Preprocessed {f} failed {e}.\n')
            pass
        # We are tearing through RAM with these stacks.  Sometimes, the garbage collection can't keep up and we run out of memory.
        # Or we force it to clean up after each stack.
        gc.collect()
        # Update the status in the calling page.
        if statusProgress is not None:
            statusProgress.progress(i/len(rawFiles), text='Processing raw data.  This may take a minute.')
    # And always leave a final status of 100%
    if statusProgress is not None:
        statusProgress.progress(1.0, text='Raw data processing complete.')
    return

if __name__ == '__main__':
    # preprocess_all_products()
    # preprocess_one_product(fileName='/hoe/zack/Rocket/WorkDir/017 EDS on Green phase/Before_1.ser', sessionId=314, statusOutput=print)
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/BundlizerData/20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser', sessionId=314, statusOutput=print)
    # tempdir = '/Users/Zack/Desktop' # Mac
    tempdir = '/home/zack/Dropbox/OSIRIS-REx/BundlizerData/TEM' # Linux
    preprocess_all_products(os.path.join(tempdir, '20230113 - TitanX - Sutter IOM Bullet 1 Grid A1 Section1 SAMISTest'), sessionId='20230818_TEM_LBNL_TEST-800206-1_1')
    # preprocess_all_products(os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer'), sessionId=314)
    # preprocess_one_product(fileName=os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser'), sessionId=314, statusOutput=print)
    # preprocess_one_product(fileName=os.path.join(tempdir, '20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0005.dm3'), sessionId=314, statusOutput=print)
    print ('Done')