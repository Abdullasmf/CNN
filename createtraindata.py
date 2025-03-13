from utils2 import creat_image, create_filled_geom, create_weighted_stress_field
import time
import pandas as pd
import os
import h5py
project_dir = os.getcwd()
tensor_file = project_dir + '/TrainingData_whole+edge_refine_mesh.h5'
def create_tensors():
    with h5py.File('edge_worst_PS_list_refine_mesh.h5', 'r') as hf:
        # List all datasets
        dataset_names = list(hf.keys())
        #print("Datasets in the file:", dataset_names)
        print("Number of datasets:", len(dataset_names))    
        # Read each dataset
        PS_list_edge = [hf[dataset][:] for dataset in dataset_names]

    with h5py.File('worst_PS_list_refine_mesh.h5', 'r') as hf:
        # List all datasets
        dataset_names = list(hf.keys())
        #print("Datasets in the file:", dataset_names)
        print("Number of datasets:", len(dataset_names))    
        # Read each dataset
        PS_list_whole = [hf[dataset][:] for dataset in dataset_names]
    stress_list=[]
    geom_list=[]
    for i in range(len(PS_list_edge)):
        stress_image, geom_image=creat_image(PS_list_whole[i],40,240,0,185,3,3)
        edgestress_img, edgegeom_img = creat_image(PS_list_edge[i],40, 240, 0, 185, 3, 3)
        filled_img = create_filled_geom(edgegeom_img,30)
        interpolated_stress = create_weighted_stress_field(stress_image, radius=30, decay_factor=2)
        masked_interpolated = np.where(filled_img == 0, 0, interpolated_stress)
        geom_list.append(filled_img)
        stress_list.append(masked_interpolated)
        print(i)
        
    stress_array=np.array(stress_list)
    geom_array=np.array(geom_list)

    with h5py.File(tensor_file, 'w') as f:
    # Convert PyTorch tensors to numpy arrays before saving
        f.create_dataset('geom', data=geom_array)
        f.create_dataset('stress', data=stress_array)
    return geom_tensor, stress_tensor

if not os.path.exists(tensor_file):
    print('Creating tensors...')
    geom_tensor, stress_tensor = create_tensors()
    
    print('Tensors created and saved to disk.')
else:
    print('lul')
    print('Tensors loaded from disk.')