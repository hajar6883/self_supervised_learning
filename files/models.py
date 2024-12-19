import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same")  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2)  
        self.dp1 = nn.Dropout(p=0.25) 
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=512)  
        self.dp2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10) 

    def forward(self, x):
        x = F.relu(self.conv1(x))     
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        x = self.dp1(x)                 # Dropout1
        x = torch.flatten(x, 1)        
        x = F.relu(self.fc1(x))         
        x = self.dp2(x)                 
        x = self.fc2(x) #logits                
        return x
    
class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16* 16 * 16, out_features=10)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = torch.flatten(x, 1)               
        x = self.fc1(x)                       
        return x
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(32*32, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x