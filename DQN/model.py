import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        # input is (100, 80)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        y_output, x_output = self.get_output_size((input_shape[0], input_shape[1]), 8, 4)
        y_output, x_output = self.get_output_size((y_output, x_output), 4, 2)
        y_output, x_output = self.get_output_size((y_output, x_output), 3, 1)
        #print((y_output, x_output))
        self.num_features = 64 * (y_output * x_output)
        self.fc1 = nn.Linear(self.num_features, 100)
        self.fc2 = nn.Linear(100, num_actions)
    
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_features)
        
        #print(x.shape)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def get_output_size(self, input_shape, field_size, stride):
        y, x = input_shape
        return int(((y - field_size) / stride) + 1), int(((x - field_size) / stride) + 1)