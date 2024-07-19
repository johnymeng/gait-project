import secrets

import easydict
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.utils.data as data
from tqdm import tqdm
import pandas as pd
import numpy as np


class mHealth(data.Dataset):
    
    def __init__(self, dataframe, raw_data):
        """
        Args:
            csv_file (string): path to csv file with data
        """
        self.dataset = dataframe
        self.raw_data = raw_data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): index
            
        Returns:
            (sequence, label)
            returns the time series data then label value for data
        """
        
        # label = self.raw_data['label'].iloc[index]
        
        # sequence = np.array()
        # print(self.dataset['chest acceleration X-axis'].iloc[index:index+32].type)
        # sequence = [self.dataset['chest acceleration X-axis'].iloc[index:index+32],
        #             self.dataset['chest acceleration Y-axis'].iloc[index:index+32],
        #             self.dataset['chest acceleration Z-axis'].iloc[index:index+32],]
        
        # sequence = [self.dataset['Chest X-axis'].iloc[index],
        #             self.dataset['Chest Y-axis'].iloc[index],
        #             self.dataset['Chest Z-axis'].iloc[index],]
        
        sequence = torch.tensor(self.dataset[index:index+16]).type(torch.float32)
        target = torch.tensor(self.dataset[index+17:index+32]).type(torch.float32)
        
        # print(sequence.type)
        return sequence, target

#create new dataframe with each index holding the 32 values in the future  