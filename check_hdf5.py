import h5py
import os
import sys

data_dir = ''
count = 0
with h5py.File(os.path.join(data_dir, 'data1.hdf5'), 'r') as hdf5_fh:
    def print_name(name):
        global count  
        print(name)
        count += 1
    hdf5_fh.visit(print_name)
print(count)
