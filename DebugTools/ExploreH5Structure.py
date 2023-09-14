import h5py

def explore_h5_group(group, path=''):
    metadata_entries = []
    for key in group.keys():
        new_path = path + f"['{key}']"
        if isinstance(group[key], h5py.Dataset):
            data_size = group[key].size
            data_value = group[key][()] if data_size < 1000 else "(too large)"
            metadata_entries.append((new_path, type(data_value), data_value))
        elif isinstance(group[key], h5py.Group):
            metadata_entries.extend(explore_h5_group(group[key], new_path))
    return metadata_entries

# Open the .h5oina file to explore the structure
h5oina_file_path = '../Raw/abc/FullTEMTest/STO Specimen 1 Site 8 Map Data 9.h5oina'
h5oina_file = h5py.File(h5oina_file_path, 'r')

# Get all metadata entries from the h5 file
h5oina_metadata_entries = explore_h5_group(h5oina_file)

# Close the h5 file
h5oina_file.close()

# Print all metadata entries
for i, entry in enumerate(h5oina_metadata_entries):
    path, data_type, data_value = entry
    print(f"{i+1}. Path: {path}")
    print(f"   Data Type: {data_type}")
    print(f"   Value: {data_value}\n")
