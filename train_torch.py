import os
import cv2
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms.v2 as transforms

from dataset import load_dataset_torch

base_dir = './datasets'

train_dir = os.path.join(base_dir, 'train/')
validation_dir = os.path.join(base_dir, 'val/')
test_dir = os.path.join(base_dir, 'test/')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_test = "./datasets/train/NORMAL/IM-0115-0001.jpeg"
image = cv2.imread(image_test)

mean = torch.mean(torch.from_numpy(image[:, :, 0].astype('float64'))), torch.mean(torch.from_numpy(image[:, :, 1].astype('float64'))), torch.mean(torch.from_numpy(image[:, :, 2].astype('float64')))
std = torch.std(torch.from_numpy(image[:, :, 0].astype('float64'))), torch.std(torch.from_numpy(image[:, :, 1].astype('float64'))), torch.std(torch.from_numpy(image[:, :, 2].astype('float64')))

print("MEAN channels: ", mean)
print("STD channels: ", std)

EPOCHS = 10
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean, std)
])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
validation_data = torchvision.datasets.ImageFolder(root=validation_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

classes = ( 'NORMAL', 'PNEUMONIA' )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256 * 256 * 3, 128) # size of the image 256 * 256 * 3
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = x.view(-1, 256 * 256 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        # return F.log_softmax(x, dim=1)
    
model = Net().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# save model
PATH = './model_torch.pth'
torch.save(model.state_dict(), PATH)

# load model
# model = Net()
# model.load_state_dict(torch.load(PATH))
# outputs = model(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
