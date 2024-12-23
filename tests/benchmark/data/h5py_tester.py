import h5py
DATASET = 'data_wikipediaInt8'
hdf5_output_file_name = "%s.hdf5" %DATASET
with h5py.File(hdf5_output_file_name,"r") as f:
    print(len(f['train']))
    print(f['test'].astype('i8')[:])