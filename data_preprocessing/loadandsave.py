import os
import numpy as np
import torch


def saveDataloaders(source_folder, dataloaders):
    source_path = os.path.join('./', source_folder)
    if not os.path.exists(source_path):
        os.makedirs(source_path)
    
    for phase, dataloader in dataloaders.items():
        file_name = 'dataloader_'+phase+'.pt'
        path = os.path.join(source_path, file_name)
        torch.save(dataloader, path)
        print(f'saved in {path}')


def loadNumpy(source_path, file_name):
    path = os.path.join(source_path, file_name)
    print(f'loading from {path}')
    
    try:
        with open(path, 'rb') as f:
            numpy_array = np.load(f)
            
        return numpy_array
    except FileExistsError:
        print(f'loading numpy failed !')
        
    
    