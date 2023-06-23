import tifffile

fname = '/Users/Zack/Desktop/20230503 - TitanX - Tagish Lake Stub 3 Lamella 1 bundlizer/314_TEMImage_00001.ome.tif'
   
with tifffile.TiffFile(fname) as tif:
    tif_tags = {}
    for tag in tif.pages[0].tags.values():
        name, value = tag.name, tag.value
        tif_tags[name] = value
        print(name,value)
    image = tif.pages[0].asarray()

