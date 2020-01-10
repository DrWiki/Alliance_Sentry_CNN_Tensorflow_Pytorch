import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
# import torch
import cv2
import copy
# from torch.autograd import Variable
from tensorflow.python.platform import gfile


def readData(num):
    paths = os.getcwd() + "/TempData/" + str(num) + "/"
    filelist = os.listdir(paths)
    AllDataSub = np.zeros((len(filelist), 24, 32, 1))
    for i in range(len(filelist)):
        print(paths + filelist[i])
        I = cv2.imread(paths + filelist[i])
        I = cv2.resize(I, (32, 24))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        I = I*0.6
        I = I.reshape(24, 32, 1)
        AllDataSub[i, :, :, :] = I
        # print(I)
        cv2.imshow("1", I)
        cv2.waitKey(1)
    return AllDataSub


# prepare the data
AllData0 = readData(0)
AllData1 = readData(1)
AllData2 = readData(2)
AllData3 = readData(3)
AllData4 = readData(4)
AllData5 = readData(5)
AllDataX = np.concatenate((AllData0, AllData1, AllData2, AllData3, AllData4, AllData5), 0)

for i in range(AllDataX.shape[0]):
    print("./TempData/"+"D"+str(i)+".png")
    cv2.imwrite("./TempData/"+"D"+str(i) + ".png", AllDataX[i, :, :, :])
