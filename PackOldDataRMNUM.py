import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
# import torch
import cv2
import copy
# from torch.autograd import Variable
from tensorflow.python.platform import gfile

batch_size = 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def readDataL(num):
    paths = "/home/nvidia/RobomasterNumData_Color/"+str(num)+"L/"
    filelist = os.listdir(paths)
    AllDataSub = np.zeros((len(filelist), 24, 32, 1))
    for i in range(len(filelist)):
        # print(paths + filelist[i])
        I = cv2.imread(paths + filelist[i])
        I = cv2.resize(I, (32, 24))
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)  # 原始图竟然是四通道的！！
        I = I.reshape(24, 32, 1)
        AllDataSub[i, :, :, :] = I
        # print(I)
        # cv2.imshow("1", I)
        # cv2.waitKey(1)
    return AllDataSub


def readDataS(num):
    paths = "/home/nvidia/RobomasterNumData_Color/"+str(num)+"S/"
    filelist = os.listdir(paths)
    AllDataSub = np.zeros((len(filelist), 24, 32, 1))
    for i in range(len(filelist)):
        I = cv2.imread(paths + filelist[i])
        I = cv2.resize(I, (32, 24))
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)  # 原始图竟然是四通道的！！
        # print("Before:", I.shape)
        I = I.reshape(24, 32, 1)
        # print("After", I.shape)
        AllDataSub[i, :, :, :] = I
        # cv2.imshow("1", I)
        # cv2.waitKey(1)
    return AllDataSub


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
AllData = np.concatenate((AllData1, AllData2, AllData3, AllData4, AllData5, AllData6, AllData7, AllData8,
                     AllData1s, AllData2s, AllData3s, AllData4s, AllData5s, AllData6s, AllData7s, AllData8s), 0)

print(AllData.shape)
AllDataTemp = copy.deepcopy(AllData)

# prepare the label
label = np.zeros((800), int)
k = 0
for i in range(16):
    for j in range(50):
        label[k] = i
        print(k, "  ", label[k])
        k = k + 1

for i in range(800):
    print("./TempData/"+str(i)+str(label[i])+".png")
    cv2.imwrite("./TempData/"+str(i)+str(label[i])+".png", AllData[i, :, :, :])
