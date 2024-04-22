import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score

transform_tensor_to_pil = ToPILImage()
transform_pil_to_tensor = ToTensor()

def load_data():

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data

training_data, test_data = load_data()
#mod_train_data = ModifiedDataset(training_data)
#mod_test_data = ModifiedDataset(test_data)

print (training_data[0][0].shape)
#print (mod_train_data[0][0].shape)

def create_dataloaders(training_data, test_data, batch_size=64):

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    return train_dataloader, test_dataloader

print (len(set([y for x,y in training_data])))

train_loader, test_loader = create_dataloaders(training_data, test_data, batch_size = 32)

class cs21m001(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Softmax(dim =1)
        self.fc1 = nn.Linear(28*28*1, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.m(x)
        return x

y = (len(set([y for x,y in training_data])))
model = cs21m001()

#train the network
def train_network(train_loader, optimizer,criteria, e):
  for epoch in range(e):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        #print(outputs.shape, labels.shape)
        tmp = torch.nn.functional.one_hot(labels, num_classes= 10)
        loss = criteria(outputs, tmp)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

  print('Finished Training')

#cross entropy
def loss_fun(y_pred, y_ground):
  v = -(y_ground * torch.log(y_pred + 0.0001))
  v = torch.sum(v)
  return v

x,y = training_data[0]
model = cs21m001()
y_pred = model(x)
print(y_pred.shape)
print(y_pred)
print(torch.sum(y_pred))
#cross_entropy(10,y_pred)

y_ground = y
loss_val = loss_fun(y_pred, y_ground)
print(loss_val)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_network(train_loader,optimizer,loss_fun,10)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

!pip install torchmetrics

from torchmetrics import Precision, Recall, F1Score, Accuracy

from torchmetrics.classification import accuracy
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #precision_recall_fscore_support(y_ground, y_pred, average='macro')
    accuracy1 = Accuracy(task="multiclass",num_classes=10)
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(task="multiclass",average = 'macro', num_classes = 10)
    print('precision :', precision(pred,y))

    recall = Recall(task="multiclass",average = 'macro', num_classes = 10)
    print('recall :', recall(pred,y))
    f1_score = F1Score(task="multiclass",average = 'macro', num_classes = 10)
    print('f1_score :', f1_score(pred,y))
    return accuracy1,precision, recall, f1_score

test(test_loader, model, loss_fun)

#write the get model
def get_model(train_loader,e = 10):
	model = cs21m001()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	criteria = loss_fun
	train_network(train_loader, optimizer,criteria,e)
	return model

