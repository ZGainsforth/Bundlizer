# Created 2023, Zack Gainsforth
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import hyperspy.api as hs
import pandas as pd
import re, os, io, gc
import glob2
import yaml
from yaml.representer import SafeRepresenter
from ncempy.io import dm, ser # Reader for dm3/dm4 files and ser (TIA) files.
import importlib
hf = importlib.import_module('helperfuncs', '..')
from tifffile import imwrite, TiffWriter
# from tifffile import imsave as tiffsave
import json

# For unique product ID's we start a counter.
productId = 0

# These file extensions indicate data types that we can convert 
raw_extensions = ['dm3', # Gatan files
                  'dm4',
                  'ser', # TIA files.
                  'bcf', # Bruker EDS files.
                 ]

# Figure out if this is a TEM image, a STEM image or what.
def get_image_type(metadata=None):
    match str(metadata['Mode []']):
        case value if 'STEM' in value:
            return 'STEM'
        case _:
            raise ValueError('Cannot guess image type from metadata.  Please expand case statement.')

# Given this is a STEM image, is this a HAADF image, BF-STEM, etc?
def get_STEM_type(metadata=None):
    # For now, only HAADF.
    return 'HAADF'

def sanitize_dict_for_yaml(data):
    sanitized_data = {}
    for k, v in data.items():
        if isinstance(data, np.ndarray):
            sanitized_data[k] = hf.numpy_to_yaml(v)
        elif isinstance(v, bytes):
            sanitized_data[k] = v.decode('utf-8')
        else:
            sanitized_data[k] = v
    return sanitized_data

def preprocess_STEM(fileName=None, sessionId=None, statusOutput=print, file=None):
    dataComponentType = 'STEMImage'
    global productId
    productId += 1

    # Make a yaml describing this data product.
    productName = f'{sessionId}_{dataComponentType}_{productId:05d}'
    yamlData = {
        "description": "default description",
        "dataComponentType": dataComponentType,
        "channel": get_STEM_type(metadata=file['metadata']),
        "pixelScaleX": float(file['pixelSize'][0]),
        "pixelScaleY": float(file['pixelSize'][1]),
        "pixelUnits": str(file['pixelUnit'][0]),
    }
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{productName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)

    # Write the supplementary yaml too with all the instrument data.
    yamlFileName = os.path.join(os.path.dirname(fileName), f'{sessionId}_instrumentMetadata_{productId:05d}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(sanitize_dict_for_yaml(file['metadata']), f, default_flow_style=False, sort_keys=False)

    metadata={
        'axes': 'YX',
        'PixelType': 'float32',
        'BigEndian': False,
        'SizeX': file['data'].shape[0],
        'SizeY': file['data'].shape[1],
        'PhysicalSizeX': 0.1,
        'PhysicalSizeXUnit': 'nm',
        # 'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': 0.1,
        'PhysicalSizeYUnit': 'nm',
        # 'PhysicalSizeYUnit': 'µm',
    }
    metadata.update(sanitize_dict_for_yaml(file['metadata']))

    with TiffWriter(os.path.join(os.path.dirname(fileName), f'{productName}.ome.tif')) as tif:
        tif.save(file['data'][:,:].astype('float32'), photometric='minisblack', metadata=metadata)
        # tif.save(file['data'][np.newaxis,:,:].astype('float32'), photometric='minisblack', metadata=metadata)
        # tiffsave(os.path.join(os.path.dirname(fileName), f'{productName}.tif'), file['data'], metadata=sanitize_dict_for_yaml(file['metadata']))

    return

def preprocess_TEM(fileName=None, sessionId=None, statusOutput=print, file=None):
    return

def preprocess_EDSCube(fileName=None, sessionId=None, statusOutput=print, file=None):
    return

def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print):
    # In the case of STXM, all products are pointed to by a hdr file.
    # Extract the file name.
    fullName, ext = os.path.splitext(fileName)
    baseName = os.path.basename(fullName) # This is the name of the file.
    pathName = os.path.dirname(fullName) # This is the directory containing the file.

    match ext:
        case '.dm3' | '.dm4':
            file = dm.dmReader(fileName)
            fileType = get_image_type(file['metadata'])
        case '.ser':
            file = ser.serReader(fileName)
            fileType = get_image_type(file['metadata'])
        case '.bcf':
            HAADF, EDS = hs.load(fileName)
            fileType = 'EDSCube'
        case _:
            raise ValueError(f"{ext} is an invalid data product type.")

    match fileType:
        case 'STEM':
            preprocess_STEM(fileName, sessionId, statusOutput, file)
        case 'TEM':
            preprocess_TEM(fileName, sessionId, statusOutput, file)
        case 'EDSCube':
            preprocess_EDSCube(fileName, sessionId, statusOutput, file)

    # # the first file is a yaml describing the data collection.
    # ProductName = f'{sessionId}_{dataComponentType}_{ProductID:05d}'
    # yamlData = {
    #     "description": "default description",
    #     "dataComponentType": dataComponentType,
    # }
    # yamlFileName = os.path.join(PathName, f'{ProductName}.yaml')
    # with open(yamlFileName, 'w') as f:
    #     yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)
    # ProductDict[ProductName] = yamlFileName

    # for n in range(1, Xim['NumRegions']+1):
    #     RegionName = 'Region%d'%n
    #     SubProductName = f'{ProductName}_{RegionName}'
    #     ProductFiles = [] # We don't know all the files yet.  Append as we go.

    #     # In stacks there is a region name since they often are multi-region.
    #     # if Xim['Type'] == 'Image':
    #     #     PlotName = 'Plot'
    #     # else:
    #     PlotName = f'Region{n}_Plot'

    #     # Save a user friendly plot image.
    #     FriendlyPlotFileName = os.path.join(PathName, f'{SubProductName}.png')
    #     Xim[PlotName].save(FriendlyPlotFileName)
    #     ProductFiles.append(FriendlyPlotFileName)

    #     # Save the raw data as tif.
    #     TifFileName = os.path.join(PathName, f'{SubProductName}.tif')
    #     SaveTifStack(TifFileName, Xim[RegionName])
    #     ProductFiles.append(TifFileName)

    #     # # Save the energy axis.
    #     # EnergyFileName = os.path.join(PathName, f'{SubProductName}.txt')
    #     # np.savetxt(EnergyFileName, Xim['Energies'])
    #     # ProductFiles.append(EnergyFileName)

    #     # Now put all the data together to make a yaml with the metadata 
    #     yamlData = {
    #         "description": "default description",
    #         "dataComponentType": dataComponentType,
    #         "dimensions": [
    #             {
    #                 "dimension": "X",
    #                 "fieldDescription": NumpyToYaml(Xim[f'Region{n}_X']),
    #                 "unitOfMeasure": "um",
    #             },
    #             {
    #                 "dimension": "Y",
    #                 "fieldDescription": NumpyToYaml(Xim[f'Region{n}_Y']),
    #                 "unitOfMeasure": "um",
    #             },
    #             {
    #                 "dimension": "Z",
    #                 "fieldDescription": NumpyToYaml(Xim['Energies']),
    #                 "unitOfMeasure": "eV",
    #             },
    #         ]
    #     }
    #     yamlFileName = os.path.join(PathName, f'{SubProductName}.yaml')
    #     # statusOutput(f'Generating yaml file: {yamlFileName}')
    #     with open(yamlFileName, 'w') as f:
    #         yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)
    #     ProductFiles.append(yamlFileName)

    #     ProductDict.update({SubProductName: ProductFiles})

    # return ProductDict

def preprocess_all_products(dirName=None, sessionId=None, statusOutput=print, statusProgress=None):
    # The user is telling us a directory which contains raw products from this instrument.
    if dirName is None:
        dirName =  os.getcwd()
    # On STXM, all data stacks of all types are identified by .hdr files.
    rawFiles = glob2.glob(os.path.join(dirName, '**', '*.hdr'))
    productsList = {}
    for i, f in enumerate(rawFiles):
        try:
            dsd = preprocess_one_product(f, sessionId=sessionId, statusOutput=statusOutput)
            productsList.update(dsd)
            statusOutput(f'Preprocessed {f} -> {dsd}.')
        except Exception as e:
            # If that one Xim failed, go on and process the next, it won't be included in the data products.
            statusOutput(f'Preprocessed {f} -> failed {e}.')
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
    return productsList

if __name__ == '__main__':
    #preprocess_all_products('/Users/Zack/Desktop/STXM Example/Track 220 W7 STXM 210620')
    # preprocess_all_products()
    # preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/017 EDS on Green phase/Before_1.ser', sessionId=314, statusOutput=print)
    preprocess_one_product(fileName='/home/zack/Rocket/WorkDir/BundlizerData/20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/0001_0000_1.ser', sessionId=314, statusOutput=print)
    print ('Done')