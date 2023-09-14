# Created 2014, Zack Gainsforth
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import io
import gc
from PIL import Image
import pandas as pd
import glob2
import yaml
import importlib
hf = importlib.import_module('helperfuncs', '..')

def SaveTifStack(FileName, Stack):
    try:
        # We can only write multiframe tiffs if the user has downloaded the tifffile module.
        from tifffile import imsave as tiffsave
    except:
        return

    # # tifffile will save it using the order: frames, height, width.  So we have to redimension the array.
    # # Currently the array is height, width, frames.
    # if len(Stack.shape) == 3:
    #     Stack = Stack.swapaxes(1,2)
    #     Stack = Stack.swapaxes(0,1)
    # In the case shape is 2D then it will just write it correctly.

    # Now save it.
    tiffsave(FileName, Stack)


def ReadXim(XimName):
    # ReadXim reads a Xim header, and then reads in the xim files to produce an image, map or stack.
    #
    # XimName is the name of the header.
    # Returns a dictionary: Xim with the following entries:
    # NumRegions: How many regions in the image.  Only supported for stacks.
    # Energies: How many energies.  1 means image.  2 means map, 3 or more is a stack.
    # Image: A single frame representation.  For an image, (1 energy) it is just the image. For the map,
    # it is the PostEdge - PreEdge.  For a stack, it is a sum along the E axis.
    # Plot: Image plotted with the info about it in the title.  This can be saved to a Tif or jpg.
    # RegionN: For each region this contains the stack as a numpy cube.

    # Extract the file name for the header.
    BaseName, _ = os.path.splitext(XimName)
    AxisName = os.path.basename(BaseName)
    hdrName = BaseName + '.hdr'

    # Open the header file for the xim.
    with open(hdrName, 'r') as f:
        hdr = f.read()

    # Find out how many regions there are.
    RegionsMatch = re.search('Regions = \(([0-9]*),', hdr)
    try:
        NumRegions = int(RegionsMatch.group(1))
    except:
        print ('could not get number of regions from hdr file.')
        return None

    # Find out the energ(ies).  Look in the StackAxis section for the entry Points = (numbers);
    # Pull out numbers.
    EnergyMatch = re.search('StackAxis.*?Points = \((.*?)\);', hdr, re.S)

    # Convert the text numbers into energies.
    try:
        Energies = np.fromstring(EnergyMatch.group(1), dtype = float, sep = ', ')
    except:
        print ('Could not get energy axis for image/stack')
        return None

    # Now the header is too nice.  It tells us how many energies there are in the first entry.  Test it and then
    # discard it.
    if Energies[0] != len(Energies)-1:
        print ("Hdr file corrupted.  Energy axis length doesn't match the first entry (which gives the length of the axis).")
        return None

    Energies = np.delete(Energies, 0)

    # Initialize list for storing x and y axis points for each region
    xAxisRegions = []
    yAxisRegions = []

    # Extract the x and y axis points for each region
    pattern = r'PAxis.*?Points = \((.*?)\);.*?QAxis.*?Points = \((.*?)\);'
    matches = re.findall(pattern, hdr, re.S)

    # Check if the number of matches is the same as the number of regions
    if len(matches) != NumRegions:
        print('Mismatch in number of regions detected in header file.')
        return None

    for match in matches:
        xAxisPoints = np.fromstring(match[0], dtype=float, sep=', ')
        yAxisPoints = np.fromstring(match[1], dtype=float, sep=', ')

        # Remove the first element which is the count of points
        xAxisPoints = np.delete(xAxisPoints, 0)
        yAxisPoints = np.delete(yAxisPoints, 0)

        xAxisRegions.append(xAxisPoints)
        yAxisRegions.append(yAxisPoints)

    # Put the data into the XimHeader
    Xim = {}
    Xim['Energies'] = Energies
    Xim['NumRegions'] = NumRegions

    # # While Axis probably does it, we don't support multiple regions for anything other than a stack.
    # # We can add this in the future.
    # if len(Energies) <= 1 and NumRegions > 1:
    #     print ("Multi-region images are not currently supported.")
    #     return None

    # For single region stacks, the numbers go 000, 001, 002, ...
    if NumRegions == 1:
        NumberIncrement = 1
        ExtensionStr = '_a%03d.xim'
    # For multi region stacks, region 1 goes 0000, 0010, 0020, ... and region 2 is 0001, 0011, 0021, ...
    else:
        NumberIncrement = 10
        ExtensionStr = '_a%04d.xim'

    # Three options based on how many energies we have:
    # 1: It's just an image.
    # 2: It's a map.
    # >2: It's a stack.
    if len(Energies) == 1:
        # For images, we just load the image.  The stack is rather dull, just one frame.
        Xim['Type'] = 'Image'
        for n in range(1, NumRegions+1):
            if NumRegions > 1:
                Xim[f'Region{n}'] = np.loadtxt(BaseName + '_a%01d.xim'%(n-1), dtype='uint16')
            # For single region
            else:
                Xim[f'Region{n}'] = np.loadtxt(BaseName + '_a.xim', dtype='uint16')
            Xim[f'Region{n}_Image'] = Xim[f'Region{n}'] 
            Xim[f'Region{n}_X'] = xAxisRegions[n-1]
            Xim[f'Region{n}_Y'] = yAxisRegions[n-1]
            Xim[f'Region{n}_Plot'] = PlotImage(Xim[f'Region{n}'], AxisName + f'_Region{n}: %0.2f eV'%Xim['Energies'][0])
    elif len(Energies) == 2:
        Xim['Type'] = 'Map'
        # Load each region into Xim.
        for n in range(1, NumRegions+1):
            PreEdge = np.loadtxt(BaseName + ExtensionStr%(n-1), dtype='uint16')
            PostEdge = np.loadtxt(BaseName + ExtensionStr%(n-1+NumberIncrement), dtype='uint16')
            Xim[f'Region{n}'] = np.array([PreEdge, PostEdge])
            Xim[f'Region{n}_X'] = xAxisRegions[n-1]
            Xim[f'Region{n}_Y'] = yAxisRegions[n-1]
            Xim[f'Region{n}_Image'], Xim[f'Region{n}_Plot'] = PlotMap(PreEdge, PostEdge, Xim['Energies'], AxisName)
    else:
        Xim['Type'] = 'Stack'

        # Load each region into Xim.
        for n in range(1, NumRegions+1):
            # Name this one.
            # RegionName = 'Region%d'%n
            FirstFrame = np.loadtxt(BaseName + ExtensionStr%(n-1), dtype='uint16')

            # Allocate the numpy array.
            ThisRegion = np.zeros((len(Xim['Energies']), FirstFrame.shape[0], FirstFrame.shape[1]), dtype='uint16')
            # Put in the one frame we already loaded.
            ThisRegion[0] = FirstFrame
            # Loop for each of the remaining frames.
            for i in range(1, len(Xim['Energies'])):
                # Read in one frame and store it.
                try:
                    # We're doing this with pandas read_csv because it is so fast.
                    # n-1 gives us the region number, i-1 gives us the frame number.  NumberIncrement gives us the jumps between frames.
                    # e.g. region 1, frame 1 = 0, region 1, frame 2 = 10, region 2 frame 2 = 11, region 2 frame 3 = 21, etc.
                    t = pd.read_csv(BaseName + ExtensionStr%((n-1)+(i-1)*NumberIncrement), sep='\t', header=None)
                    # But Tolek has a \t at the end of the line and Pandas reads in a last column of NaNs.  So ditch
                    # the last col.
                    ThisRegion[i] = t.values[:,:-1]
                except IOError:
                    # It is common that stacks will be truncated.  In this case, we will fail at reading some file.
                    # Lets trunctate the stack here.
                    Xim['Energies'] = Xim['Energies'][:i]
                    ThisRegion = ThisRegion[:i]
                    # Done.  Go to next region.
                    break

            # Store it.
            # Xim[RegionName] = ThisRegion
            Xim[f'Region{n}'] = ThisRegion
            Xim[f'Region{n}_X'] = xAxisRegions[n-1]
            Xim[f'Region{n}_Y'] = yAxisRegions[n-1]
            Xim[f'Region{n}_Image'], Xim[f'Region{n}_Plot'] = PlotStack(Xim[f'Region{n}'], Xim['Energies'], AxisName+f'_Region{n}')

            # # And generate "pretty images" i.e. thumbnails.
            # ThisAxisName = AxisName
            # if NumRegions > 1:
            #     ThisAxisName = AxisName + '_Region%d'%n
            # Xim['Image'+RegionName], Xim['Plot'+RegionName] = PlotStack(Xim[RegionName], Xim['Energies'], AxisName)

    return Xim

def PlotImage(ImRaw, title):
    plt.clf()
    # Get the min and max intensity.
    vmin = np.min(ImRaw)
    vmax = np.max(ImRaw)
    # Plot it.
    ax = plt.imshow(ImRaw, interpolation='none', cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    # And save that plot as an image in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)

    return im

def PlotMap(PreEdge, PostEdge, Energies, AxisName):
    # Compute the OD image.
    Map = -np.log(PostEdge.astype(float)/PreEdge.astype(float))

    # Make a plot in memory with a title.  Png format will be default.
    im = PlotImage(Map, AxisName + ': ln(%0.2f/%0.2f) eV'%(Energies[1], Energies[0]))

    # And pass it back out.
    return Map, im

def PlotStack(Region, Energies, AxisName):
    # Compute the sum image
    Sum = np.sum(Region.astype(float), axis=0)

    # Make a plot in memory with a title.  Png format will be default.
    im = PlotImage(Sum, AxisName + ': sum between %0.2f to %0.2f eV'%(Energies[0], Energies[-1]))

    # And pass it back out.
    return Sum, im

def preprocess_one_product(fileName=None, sessionId=None, statusOutput=print):
    # In the case of STXM, all products are pointed to by a hdr file.
    # Extract the file name.
    FullName, _ = os.path.splitext(fileName)
    BaseName = os.path.basename(FullName) # This is the name of the file.
    _,ProductID = BaseName.split("_") # Get rid of the beamline prefix -- SAMIS requires just a number.
    PathName = os.path.dirname(FullName) # This is the directory containing the file.

    Xim = ReadXim(FullName)

    match Xim['Type']:
        case 'Map':
            dataComponentType = 'XANESCollection'
        case 'Stack':
            dataComponentType = 'XANESCollection'
        case 'Image':
            dataComponentType = 'XANESImage'
        # These are todo.
        # case 'LinescanRaw':
        #     dataComponentType = 'XANESRawTabular'
        # case 'LinescanProcessed':
        #     dataComponentType = 'XANESProcessedTabular'
        case _:
            raise ValueError(f"{Xim['Type']} is an invalid data product type.")
    ProductDict = {}

    ProductName = f'{sessionId}_{dataComponentType}_{ProductID}'

    # Data collections need to go into a subdirectory so they can be zipped up later.
    if dataComponentType == 'XANESCollection':
        DataPath = os.path.join(PathName, ProductName)
        os.makedirs(DataPath , exist_ok=True)  # Create the subdirectory if it doesn't exist
    else:
        DataPath  = PathName

    # the first file is a yaml describing the data collection.
    yamlData = {
        "description": "default description",
        "dataComponentType": dataComponentType,
    }
    yamlFileName = os.path.join(PathName, f'{ProductName}.yaml')
    with open(yamlFileName, 'w') as f:
        yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)
    ProductDict[ProductName] = yamlFileName

    for n in range(1, Xim['NumRegions']+1):
        RegionName = 'Region%d'%n
        SubProductName = f'{ProductName}_{RegionName}'
        ProductFiles = [] # We don't know all the files yet.  Append as we go.

        PlotName = f'Region{n}_Plot'

        # Save a user friendly plot image.
        FriendlyPlotFileName = os.path.join(DataPath , f'{SubProductName}.png')
        Xim[PlotName].save(FriendlyPlotFileName)
        ProductFiles.append(FriendlyPlotFileName)

        # Save the raw data as tif.
        TifFileName = os.path.join(DataPath , f'{SubProductName}.tif')
        SaveTifStack(TifFileName, Xim[RegionName])
        ProductFiles.append(TifFileName)

        # Now put all the data together to make a yaml with the metadata 
        yamlData = {
            "description": "default description",
            "dataComponentType": dataComponentType,
            "dimensions": [
                {
                    "dimension": "X",
                    "fieldDescription": hf.NumpyToYaml(Xim[f'Region{n}_X']),
                    "unitOfMeasure": "um",
                },
                {
                    "dimension": "Y",
                    "fieldDescription": hf.NumpyToYaml(Xim[f'Region{n}_Y']),
                    "unitOfMeasure": "um",
                },
                {
                    "dimension": "Z",
                    "fieldDescription": hf.NumpyToYaml(Xim['Energies']),
                    "unitOfMeasure": "eV",
                },
            ]
        }
        yamlFileName = os.path.join(DataPath , f'{SubProductName}.yaml')
        # statusOutput(f'Generating yaml file: {yamlFileName}')
        with open(yamlFileName, 'w') as f:
            yaml.dump(yamlData, f, default_flow_style=False, sort_keys=False)
        ProductFiles.append(yamlFileName)

        ProductDict.update({SubProductName: ProductFiles})

    return ProductDict

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
    # tempdir = '/Users/Zack/Dropbox/OSIRIS-REx/BundlizerData/NanoIR' # Mac
    tempdir = '/home/zack/Dropbox/OSIRIS-REx/BundlizerData/STXM' # Linux
    preprocess_all_products(os.path.join(tempdir, 'Track 220 W7 STXM 210620'), sessionId=314)
    print ('Done')