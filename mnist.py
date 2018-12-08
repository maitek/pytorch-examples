from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Define Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **gpu_args)
    
    # Initialize model on device
    model = Net().to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(0, num_epochs):
        
        # Train loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # Classification Loss (Negative Log Likelihood)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # Test loop
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad(): # disable gradients for speed 
            for data, target in test_loader:

                # Predict using model
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
