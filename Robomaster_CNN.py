import cv2
import os
import torch
import torch.nn as nn
import torchvision
import argparse
import numpy as np
import copy
from torch.autograd import Variable

pa = argparse.ArgumentParser()
pa.add_argument("--input_row", type=int, default=0)
pa.add_argument("--input_col", type=int, default=0)
pa.add_argument("--input_channel", type=int, default=0)

opt = pa.parse_args()
opt.input_row = 24
opt.input_col = 32
opt.input_channel = 1
print(opt)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("++++++++++Your CUDA is available!++++++++++" if torch.cuda.is_available() else "----------No CUDA available!----------")


def readDataL(num):
    paths = os.getcwd() + "/RobomasterNumData/" + str(num) + "L/"
    filelist = os.listdir(paths)
    AllDataSub = torch.rand(len(filelist), 1, 24, 32)
    for i in range(len(filelist)):
        I = cv2.imread(paths + filelist[i])
        I = cv2.resize(I, (opt.input_col, opt.input_row))
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)  # 原始图竟然是四通道的！！
        temp = torch.from_numpy(I)
        AllDataSub[i, 0, :, :] = temp
        # cv2.imshow("1", I)
        # cv2.waitKey(1)
    return AllDataSub


def readDataS(num):
    paths = os.getcwd() + "/RobomasterNumData/" + str(num) + "S/"
    filelist = os.listdir(paths)
    AllDataSub = torch.rand(len(filelist), 1, 24, 32)
    for i in range(len(filelist)):
        I = cv2.imread(paths + filelist[i])
        I = cv2.resize(I, (opt.input_col, opt.input_row))
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)  # 原始图竟然是四通道的！！
        temp = torch.from_numpy(I)
        AllDataSub[i, 0, :, :] = temp
        # cv2.imshow("1", I)
        # cv2.waitKey(1)
    return AllDataSub


class Num_Reg(nn.Module):
    def __init__(self):
        super(Num_Reg, self).__init__()
        self.convA = nn.Sequential(
            nn.Conv2d(
                in_channels=opt.input_channel,
                out_channels=10,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.convB = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=30,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.f1 = nn.Linear(8 * 6 * 1 * 30, 200)
        self.f2 = nn.Linear(200, 16)

    def forward(self, x):
        x = self.convA(x)
        x = self.convB(x)
        x = x.view(x.size(0), -1)  # why it's -1
        x = self.f1(x)
        x = self.f2(x)
        return x


Recognizer = Num_Reg()
print("Your Model's structure: ", Recognizer)

# prepare the data
AllData1 = readDataL(1)
AllData2 = readDataL(2)
AllData3 = readDataL(3)
AllData4 = readDataL(4)
AllData5 = readDataL(5)
AllData6 = readDataL(6)
AllData7 = readDataL(7)
AllData8 = readDataL(8)

AllData1s = readDataS(1)
AllData2s = readDataS(2)
AllData3s = readDataS(3)
AllData4s = readDataS(4)
AllData5s = readDataS(5)
AllData6s = readDataS(6)
AllData7s = readDataS(7)
AllData8s = readDataS(8)
AllData = torch.cat((AllData1, AllData2, AllData3, AllData4, AllData5, AllData6, AllData7, AllData8,
                     AllData1s, AllData2s, AllData3s, AllData4s, AllData5s, AllData6s, AllData7s, AllData8s), 0)

# prepare the label
label = np.zeros((50 * 16), int)
k = 0
for i in range(16):
    for j in range(50):
        label[k] = i
        k = k + 1
        # print(k, "  ", label[k])
label = torch.from_numpy(label)

# test your label and your data
for j in range(800):
    cv2.imwrite("./TestImage/" + str(j) + "-" + str(label.numpy()[j]) + ".jpg", AllData[j, 0, :, :].numpy())
print("Here is your data's shape", AllData.shape)
print("Here is your label's shape", label.shape)

# exit(128)
# shuffle it
shuffle_kernel = np.arange(0, 800, 1)
np.random.shuffle(shuffle_kernel)

labeltemp = label.numpy()
labeltemp1 = copy.deepcopy(labeltemp)

AllDataTemp = copy.deepcopy(AllData)
for n in range(800):
    AllDataTemp[n, :, :, :] = AllData[shuffle_kernel[n], :, :, :]
    labeltemp1[n] = labeltemp[shuffle_kernel[n]]

label = torch.from_numpy(labeltemp1)
for n in range(800):
    cv2.imwrite("./TestImage2/" + str(n) + "-" + str(label.numpy()[n]) + ".jpg", AllDataTemp[n, 0, :, :].numpy())
AllData = copy.deepcopy(AllDataTemp)
# print("labeltemp1", labeltemp1)
# print("label", label)
# exit(255)

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
optimizer = torch.optim.Adam(Recognizer.parameters(), lr=0.001)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
loss_func.cuda(device)
Recognizer.cuda(device)
for num in range(750):
    xx = AllData[num, :, :, :]
    x = torch.rand(1, 1, 24, 32)
    x[0, 0, :, :] = xx
    x = x / 255.0
    x = x.cuda(device)
    # print(type(x), x.is_floating_point(), x.dtype)

    label = label.cuda(device)
    y = Recognizer(x)[0]
    y = y.reshape(1, 16)
    y = y.cuda(device)
    # print(type(y), y.is_floating_point(), y.dtype)

    labelv = Variable(label[num])
    labelv = labelv.reshape(1)
    # labelv = labelv.to(torch.int64)
    labelv = labelv.cuda(device)
    # print(type(labelv), labelv.is_floating_point(), labelv.dtype)

    # print(y.shape, labelv)
    loss = loss_func(y, labelv)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    print("num: ", num)
print("OVER! ")

for rr in range(50):
    rrr = rr+750
    xx = AllData[rrr, :, :, :]
    x = torch.rand(1, 1, 24, 32)
    x[0, 0, :, :] = xx
    x = x/255.0
    # print(x.shape)
    x = x.cuda()
    test_output = Recognizer(x)[0]

    test_output = test_output.cpu()
    test_output = test_output.reshape(1, 16)
    # print(test_output)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    label = label.cpu()
    # print(rr, 'prediction number', pred_y, 'real number', label[rrr], test_output, pred_y == label[rrr].numpy())
    print(rr, 'prediction number', pred_y, 'real number', label[rrr], pred_y == label[rrr].numpy())

torch.save(Recognizer, "./RoboMaterData.t7")
print("OVER! ")
