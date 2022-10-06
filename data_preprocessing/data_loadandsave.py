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


def saveNumpy(numpy, source_path, file_name):
    path = os.path.join(source_path, file_name)
    print(f'saveing to {path}')
    
    try:
        with open(path, 'wb') as f:
            np.save(f, numpy)
            
        return numpy_array
    except FileExistsError:
        print(f'saving numpy failed !')
    

def splitNumpy(numpy, ratio_train_data, ratio_val_data,
               source_path, file_name):
    
    dataset_size = numpy.shape[0]
    train_size = int(dataset_size * ratio_train_data)
    val_size = int(dataset_size * ratio_val_data)
    test_size = dataset_size - train_size - val_size
    