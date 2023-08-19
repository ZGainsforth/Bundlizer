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

        # Create sum over energy axis (2D image)
        sum_image = np.sum(eds_data, axis=2)

        # Create mean and max over spatial dimensions (spectrum)
        spectrum = np.mean(eds_data, axis=(0,1))
        spectrum_max = np.max(eds_data, axis=(0,1))

        # Get axes information
        dim1 = np.array(emd['data']['EDS']['dim1'])[:, 0]
        dim2 = np.array(emd['data']['EDS']['dim2'])[:, 0]
        dim3 = np.array(emd['data']['EDS']['dim3'])[:, 0]

        # Create the figure and axes using plt.subplots
        fig, axs = plt.subplots(2, 1, figsize=(6, 9)) 

        # Plot 2D image
        im = axs[0].imshow(sum_image, cmap='gray', extent=[dim2[0], dim2[-1], dim1[0], dim1[-1]])
        axs[0].set_title('EDS stack sum image')
        axs[0].set_ylabel(emd['data']['EDS']['dim1'].attrs['name'].decode() + ' (' + emd['data']['EDS']['dim1'].attrs['units'].decode() + ')')
        axs[0].set_xlabel(emd['data']['EDS']['dim2'].attrs['name'].decode() + ' (' + emd['data']['EDS']['dim2'].attrs['units'].decode() + ')')

        # Plot spectrum
        axs[1].plot(dim3, spectrum)
        axs[1].plot(dim3, spectrum_max)
        axs[1].set_title('EDS stack spectrum')
        axs[1].set_xlabel(emd['data']['EDS']['dim3'].attrs['name'].decode() + ' (' + emd['data']['EDS']['dim3'].attrs['units'].decode() + ')')
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
        }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

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
    dim.attrs['name'] = np.string_('height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['height'].units))

    dim = emd_eds.create_dataset(f'dim2', (eds.data.shape[1],1))
    dim[:,0] = eds.axes_manager['width'].axis
    dim.attrs['name'] = np.string_('width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['width'].units))

    dim = emd_eds.create_dataset(f'dim3', (eds.data.shape[2],1))
    dim[:,0] = eds.axes_manager['Energy'].axis
    dim.attrs['name'] = np.string_('Energy')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['Energy'].units))

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
    dim[:,0] = haadf.axes_manager['height'].axis
    dim.attrs['name'] = np.string_('height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(haadf.axes_manager['height'].units))

    dim = emd_haadf.create_dataset(f'dim2', (haadf.data.shape[1],1))
    dim[:,0] = haadf.axes_manager['width'].axis
    dim.attrs['name'] = np.string_('width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(haadf.axes_manager['width'].units))

    haadf_metadata = emd_haadf.create_group('microscope') # metadata for haadf image
    for k, v in core_metadata.items():
        haadf_metadata.attrs[k] = v
    for k, v in hf.flatten_dict(haadf.metadata.as_dictionary()).items():
        if type(v) in [bool, str, int, float]:
            haadf_metadata.attrs[k] = v

    emd.close()

    return

# Process Oxford EDS info.
def preprocess_h5oina(fileName=None, sessionId=None, statusOutput=print, samisData=None):
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

    # Load the h5 file.
    h5 = h5py.File(fileName, 'r')

    productName = f"{sessionId}_{core_metadata['dataComponentType']}_{productId:05d}"
    statusOutput(f'Producing data product {productName}.')

    # Make a yaml describing this data product.
    yamlData = { 
        "description": core_metadata['description'],
        "dataComponentType": core_metadata['dataComponentType'],
        }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

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
    dim.attrs['name'] = np.string_('height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['height'].units))

    dim = emd_eds.create_dataset(f'dim2', (eds.data.shape[1],1))
    dim[:,0] = eds.axes_manager['width'].axis
    dim.attrs['name'] = np.string_('width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['width'].units))

    dim = emd_eds.create_dataset(f'dim3', (eds.data.shape[2],1))
    dim[:,0] = eds.axes_manager['Energy'].axis
    dim.attrs['name'] = np.string_('Energy')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(eds.axes_manager['Energy'].units))

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
    dim[:,0] = haadf.axes_manager['height'].axis
    dim.attrs['name'] = np.string_('height')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(haadf.axes_manager['height'].units))

    dim = emd_haadf.create_dataset(f'dim2', (haadf.data.shape[1],1))
    dim[:,0] = haadf.axes_manager['width'].axis
    dim.attrs['name'] = np.string_('width')
    dim.attrs['units'] = np.string_(hf.replace_greek_symbols(haadf.axes_manager['width'].units))

    haadf_metadata = emd_haadf.create_group('microscope') # metadata for haadf image
    for k, v in core_metadata.items():
        haadf_metadata.attrs[k] = v
    for k, v in hf.flatten_dict(haadf.metadata.as_dictionary()).items():
        if type(v) in [bool, str, int, float]:
            haadf_metadata.attrs[k] = v

    emd.close()

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