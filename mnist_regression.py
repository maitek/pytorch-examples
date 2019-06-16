from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import scipy.spatial

class TargetCenters:
    # Translate target index into point coordinate
    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    def __init__(self,num_classes):
        D = np.ones((num_classes,num_classes), dtype=np.float32) - np.eye(num_classes, dtype=np.float32)
        U, S, V = np.linalg.svd(D)
        self.X = U*np.sqrt(S)

    def __call__(self,idx_or_arr):
        return self.X[idx_or_arr]


# Define Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # target centers
        self.target_centers = TargetCenters(10)

    def get_target_center(self, target):
        return torch.from_numpy(self.target_centers(target))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def predict(self,x):
        coord = self.forward(x).detach()
        dist = scipy.spatial.distance_matrix(coord, self.target_centers.X)
        return np.argmin(dist, 1)

def main():

    # Initilize hyper parameters
    learning_rate = 1e-4
    num_epochs = 1000
    batch_size = 64
    log_interval = 100

    # Setup preprocess transforms
    image_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Setup train data loader
    train_data = datasets.MNIST('./data', train=True, download=True, transform=image_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **gpu_args)

    # Setup test data loader
    test_data = datasets.MNIST('./data', train=False, download=True, transform=image_transforms)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **gpu_args)
    
    # Initialize model on device
    model = Net().to(device)

    
    

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(0, num_epochs):
        
        # Train loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #import pdb; pdb.set_trace()
            
            optimizer.zero_grad()
            output = model(data)

            # Regression towards target class centers
            target_c = model.get_target_center(target)

            target_c_other = target_c

            loss = F.mse_loss(output, target_c)
            loss.backward()
            optimizer.step()
    
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            pred = torch.tensor(model.predict(data))
            
        # Test loop
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad(): # disable gradients for speed 
            for data, target in test_loader:

                # Predict using model
                data, target = data.to(device), target.to(device)
                output = model(data)

                target_c = model.get_target_center(target)
                pred = torch.tensor(model.predict(data))
                test_loss += F.mse_loss(output, target_c, reduction='sum').item() # sum up batch loss
                
                 # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()