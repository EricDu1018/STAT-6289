import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
    os.system("unzip train.zip")

DATA_DIR = os.getcwd() + "/train/"
x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (400, 400)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read()
    y.append(label.split("\n"))

binary_one_hot = MultiLabelBinarizer(classes=['red blood cell', 'difficult', 'gametocyte', 'trophozoite', 'ring', 'schizont', 'leukocyte'])
y = binary_one_hot.fit_transform(y)
x, y = np.array(x), np.array(y)
np.save("x_train_400.npy", x)
np.save("y_train_400.npy", y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 5e-3
N_EPOCHS = 100
BATCH_SIZE = 150
DROPOUT = 0.2
# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = np.load("x_train_400.npy"), np.load("y_train_400.npy")

x = x.transpose(0,3,1,2)/255
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2)

trainset=TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
train_loader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
testset=TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))
test_loader=DataLoader(testset,batch_size=BATCH_SIZE//4,shuffle=True)
x_test,y_test = torch.tensor(x_test).float().to(device),torch.tensor(y_test).to(device)
# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=2)  # output (n_examples, 16, 400,400)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 200,200)

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)  # output (n_examples, 32, 200,200)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 100,100)

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)  # output (n_examples, 64, 100,100)
        self.convnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 50,50)

        self.conv4 = nn.Conv2d(64, 32, (3, 3), padding=1)  # output (n_examples, 32, 50,50)
        self.convnorm4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 25,25)

        self.linear1 = nn.Linear(32*25*25, 30)  # input will be flattened to (n_examples, 32 * 25 * 25)
        self.linear1_bn = nn.BatchNorm1d(30)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(30, 7)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
        x = self.pool4(self.convnorm4(self.act(self.conv4(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
criterion=nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    model.train()
    loss_train = 0
    for step_train, (batch_x_train, batch_y_train) in enumerate(train_loader):
        y = Variable(batch_y_train).cuda()
        x = Variable(batch_x_train).cuda()
        pred_y = torch.sigmoid(model(x))
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    loss_test=0
    model.eval()
    for step_test, (batch_x_test, batch_y_test) in enumerate(test_loader):
        y = Variable(batch_y_test).cuda()
        x = Variable(batch_x_test).cuda()
        y_test_pred = torch.sigmoid(model(x))
        loss = criterion(y_test_pred, y)
        loss_test += loss.item()
    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, loss_train / BATCH_SIZE, loss_test / BATCH_SIZE))

torch.save(model.state_dict(), "model_bdu16.pt")



RESIZE_TO_x, RESIZE_TO_y = 400, 400
DROPOUT=0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(x):
    images = []
    for num in range(len(x)):
        images.append(cv2.resize(cv2.imread(x[num]), (RESIZE_TO_x, RESIZE_TO_y)))
    x = torch.FloatTensor(np.array(images))
    x=x/255

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2)  # output (n_examples, 16, 120,160) 400
            self.convnorm1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 60,80) 200

            self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)  # output (n_examples, 32, 60,80) 200
            self.convnorm2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 30,40) 100

            self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)  # output (n_examples, 64, 30,40) 100
            self.convnorm3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 15,20) 50

            self.conv4 = nn.Conv2d(64, 32, (3, 3), padding=1)  # output (n_examples, 32, 15,20) 50
            self.convnorm4 = nn.BatchNorm2d(32)
            self.pool4 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 7,10) 25

            self.linear1 = nn.Linear(32 * 25 * 25, 30)  # input will be flattened to (n_examples, 32 * 25 * 25)
            self.linear1_bn = nn.BatchNorm1d(30)
            self.drop = nn.Dropout(DROPOUT)
            self.linear2 = nn.Linear(30, 7)
            self.act = torch.relu

        def forward(self, x):
            x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
            x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
            x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
            x = self.pool4(self.convnorm4(self.act(self.conv4(x))))
            x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
            return self.linear2(x)

    model = CNN().to(device)
    model.load_state_dict(torch.load("model_bdu16_day4.pt"))
    model.eval()
    x = Variable(x).cuda()
    y_pred = model(x)
    y_pred = (y_pred.data.cpu()>0).float()







