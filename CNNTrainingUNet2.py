from utils import get_device, find_principal_stresses_3d, creat_image, create_filled_geom, create_weighted_stress_field
# from torch.utils.data import random_split
# from torch.utils.data import Dataset, DataLoader, random_split
# import torch.optim as optim
import time
# import torch
import matplotlib.pyplot as plt
# import torchvision
import numpy as np
# import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
# from models import CNNDataset
from scipy.spatial import KDTree
# from torch.utils.data import random_split
# from sklearn.model_selection import train_test_split
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader, Dataset
from scipy.spatial import cKDTree
# from models5 import StressNet, StressConcentrationLoss
import time
device = get_device()
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
    stress_tensor=torch.tensor(stress_array).float()
    geom_tensor=torch.tensor(geom_array).float()
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
    with h5py.File(tensor_file, 'r') as f:
        geom_tensor = torch.from_numpy(f['geom'][:])
        stress_tensor = torch.from_numpy(f['stress'][:])
    print('Tensors loaded from disk.')

pause = input("Press enter to continue...")

outscaler = MinMaxScaler(feature_range=(0, 1))
stress_tensor_scaled = outscaler.fit_transform(stress_tensor.reshape(-1, 1)).reshape(stress_tensor.shape)
stress_tensor_scaled = torch.tensor(stress_tensor_scaled).float()
print("scaled tensor created")

Dataset = CNNDataset(geom_tensor, stress_tensor_scaled)
train_size = int(0.85 * len(Dataset))
test_size = len(Dataset) - train_size
# Dataset = CNNDataset(geom_tensor[0:2], stress_tensor_scaled[0:2])
# print(f"Dataset length: {len(Dataset)}")
# train_size = 1
# test_size = 1
batch_size = 20
train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")


for batch_idx, (input_images, target_images) in enumerate(train_loader):
    print(f"Input shape: {input_images.shape}")  # Should be [batch_size, 1, 350, 500]
    print(f"Target shape: {target_images.shape}")  # Should be [batch_size, 1, 350, 500]
    break  # Only check the first batch




BestModelPath = project_dir + '/StressNetrefinedMesh.pth'
if os.path.exists(BestModelPath):
    print("Loading existing model...")
    model = StressNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    checkpoint = torch.load(BestModelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Existing model loaded. Best validation loss: {best_val_loss}")
    scheduler = ReduceLROnPlateau(optimizer, 'min',  patience=5, factor=0.5)
    criterion = StressConcentrationLoss(alpha=0.9, gamma=1.5, threshold=0.3)
    num_epochs = 30000
else:
    print("No existing model found. Creating a new model...")
    model = StressNet().to(device)
    criterion = StressConcentrationLoss(alpha=0.9, gamma=1.5, threshold=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min',  patience=5, factor=0.5)
    best_val_loss = float('inf')
    
    print("Starting Training")
    # Training loop
    num_epochs = 30000

variance = stress_tensor_scaled.var()
for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    for batch_idx, (input_images, target_images) in enumerate(train_loader):
        input_images = input_images.to(device)
        optimizer.zero_grad()
        outputs = model(input_images).to(device)
        target_images = target_images.to(device)
        loss = criterion(outputs, target_images)
        print(outputs.shape, target_images.shape, input_images.shape, loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        
        
    
    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        for batch_idx, (input_images, target_images) in enumerate(test_loader):
            input_images = input_images.to(device)
            outputs = model(input_images).to(device)
            target_images = target_images.to(device)
            loss = criterion(outputs, target_images)
            #print(f"Epoch {epoch}, Batch {batch_idx}. Loss: {loss.item()}")
            validation_loss += loss.item()
    scheduler.step(validation_loss)
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_loss': training_loss,
        'val_loss': validation_loss
        }, BestModelPath)
        print(f"New best model saved. Val loss: {best_val_loss} time: {time.time()}")
    if epoch % 50 == 0:
        print(f"Epoch {epoch}. Train: {training_loss}. Val loss: {validation_loss} time: {time.time()}")

print("Finished Training")