
import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        #  I don't know why I have to manually do this and not
        #  [to_device(x, device) for x in data] but this is the
        #  only way it would work?
        return_list = []
        for x in data:
            return_list.append(x.to(device))

        return return_list
    return data.to(device, non_blocking=True)


# Use standard FashionMNIST dataset
train_data = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#  Visualize?
# for image, label in train_data:
#     print(labels_map[label])
#     plt.imshow(image[0, :])
#     plt.show()



#  Setup our data
val_size = 5000
train_size = len(train_data) - val_size
train_ds, val_ds = random_split(train_data, [train_size, val_size])

batch_size = 256
train_loader = DataLoader(train_ds, batch_size)
val_loader = DataLoader(val_ds, batch_size)


#  Setup our model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 14 x 14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 7 x 7

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, batch):
        images, labels = batch
        output = self.model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def validate(self, batch):
        images, labels = batch
        outputs = self.model(images)

        _, predictions = torch.max(outputs, dim=1)
        accuracy = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

        loss = F.cross_entropy(outputs, labels)

        return accuracy, loss

    def forward(self, image):
        return self.model(image)

    def reset(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 14 x 14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 7 x 7

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))


device = get_default_device()
#  Move to GPU
model = CNN()
model = to_device(model, device)

#  Initial guesses pretraining
acc_avg = []
loss_avg = []
for batch in val_loader:
    batch = to_device(batch, device)
    accuracy, loss = model.validate(batch)
    acc_avg.append(accuracy)
    loss_avg.append(loss)
first_accuracy = sum(acc_avg)/len(acc_avg)
first_loss = sum(loss_avg)/len(loss_avg)
print("Epoch:", 0, ", Accuracy:", first_accuracy.item(), ", Loss:", first_loss.item())

#  Beginning training our data and checking with validation
epochs = 5
accuracies = []
losses = []
for i in range(epochs):
    #  Training
    for batch in train_loader:
        batch = to_device(batch, device)
        model.train(batch)

    #  Validation
    acc_avg = []
    loss_avg = []
    for batch in val_loader:
        batch = to_device(batch, device)
        accuracy, loss = model.validate(batch)
        acc_avg.append(accuracy)
        loss_avg.append(loss)

    accuracy = sum(acc_avg)/len(acc_avg)
    loss = sum(loss_avg)/len(loss_avg)

    acc_avg.clear()
    loss_avg.clear()

    accuracies.append(accuracy)
    losses.append(loss)

    print("Epoch:", i + 1, ", Accuracy:", accuracy.item(), ", Loss:", loss.item())

#  Visualize how our epochs did
losses.insert(0, first_loss)
accuracies.insert(0, first_accuracy)
losses = to_device(losses, 'cpu')
accuracies = to_device(accuracies, 'cpu')

plt.plot(losses, label="Loss")
plt.plot(accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Percentage")
plt.legend()
plt.show()

#  Visualize results with test data
test_data = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# model = to_device(model, 'cpu')
count = 0
for image, label in test_data:
    showImage = image
    image = to_device(image, device)
    print("Index:", count)
    print("Label:", labels_map[label])
    # in_image = image.unsqueeze(1)
    output = model(image[None, ...])
    _, prediction = torch.max(output, dim=1)
    print("Prediction:", labels_map[prediction.item()])
    plt.imshow(showImage[0, :])
    title = plt.title("Label: " + labels_map[label] + ", Prediction: " + labels_map[prediction.item()])
    if label == prediction.item():
        plt.setp(title, color='k')
    else:
        plt.setp(title, color='r')
    plt.show()
    print()
    count += 1





















