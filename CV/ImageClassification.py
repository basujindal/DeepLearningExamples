import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision
from torchvision import  transforms
import torch.nn.functional as F


# Data augmentation and normalization for trainings
# Normalization for validation
num_classes = 33
batch_size = 64
trans_train = transforms.Compose([ torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
trans_test = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])


train_data_dir = '/content/data/train' # put path of training dataset
val_data_dir = '/content/data/val' # put path of validation dataset
test_data_dir = '/content/data/test' # put path of test dataset

trainset = torchvision.datasets.ImageFolder(root= train_data_dir, transform=trans_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageFolder(root= val_data_dir, transform=trans_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                         shuffle=False, num_workers=2)

testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=trans_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
class_names = trainset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(class_names)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,10))
    plt.imshow(inp)
    
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Base Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,128, 3)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3)

        self.fc1 = nn.Linear(256* 3* 3, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 33)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = self.bn5(self.pool(F.relu(self.conv5(x))))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def val(loader):
  corrects = 0
  totals = 0
  with torch.no_grad():
      net.eval()
      for data in loader:
          images, labels = data
          images = images.to(device)
          labels = labels.to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          totals += labels.size(0)
          corrects += (predicted == labels).sum().item()

  # print('Test Accuracy: %d %%' % (100 * corrects / totals))
  return (100*(corrects / totals))

net = Net().to(device)
print(net)

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
running_loss = 0.0
num_steps = 50
ep = 0
best_accu = 0
vals, trains, losses=  [], [], []


#Training
for epoch in range(70):  # loop over the dataset multiple times
    if epoch%3 == 0:
      trainset = torchvision.datasets.ImageFolder(root= train_data_dir, transform=trans_train)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True)
    correct, total = 0, 0
    for data in trainloader:
        ep+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if ep % num_steps == 0: 
            val_accu = val(valloader)
            print("epoch {0} | loss: {1:.2f} | Val Accuracy: {2:.2f} %".format(epoch, running_loss/num_steps, val_accu ))
            running_loss = 0.0
            if val_accu > best_accu:
              best_accu= val_accu 
              torch.save(net.state_dict(), 'net_val.pth')
              print("Saving")
            net.train()
    print("Train accuracy: {0:.2f} %".format(100 * correct / total))

print('Finished Training')

net = Net().to(device)
net.load_state_dict(torch.load("/content/net_val.pth"))

testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=trans_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True)


class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
with torch.no_grad():
    net.eval()
    for data in testloader:
          images, labels = data
          outputs = net(images.to(device))
          _, predicted = torch.max(outputs, 1)
          c = (predicted.cpu() == labels).squeeze()
          for i in range(len(data[0])):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1

full = []
for i in range(num_classes):
    accu  = 100 * class_correct[i] / class_total[i]
    full.append(accu)
    print('Accuracy of %5s : %2d %%' % (class_names[i], accu))
print("Test Accuracy {0:.2f}".format(np.mean(np.array(full))))

# Checking for Wrong predictions.
li = []
with torch.no_grad():
    net.eval()
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted.cpu() != labels).squeeze()
        print(c)
        
        try:
          out = torchvision.utils.make_grid([images[i] for i in range(len(c)) if c[i]])
          imshow(out, title=[str(class_names[predicted[i]])+ ' '+ str(class_names[labels[i]]) for i in range(len(c)) if c[i]])
          break
        except:
          pass

print([(n, i) for n, i in enumerate(class_names)])
