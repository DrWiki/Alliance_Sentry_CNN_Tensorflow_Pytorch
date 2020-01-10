import cv2
import numpy as np
import torch
import torch.nn
import torchvision

#

I1 = cv2.imread("./1.png")
cv2.imshow("I1", I1)
print(type(I1))
print(I1.shape)
I2 = I1.reshape(320, 960, 3)
cv2.imshow("I2", I2)

I2t = torch.from_numpy(I2)
print(I2t)
print(I2t.shape)
print(I2t.size())

I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
cv2.imshow("I1g", I1)

data = np.arange(0, 10, 1)
i = np.arange(0, 10, 1)
np.random.shuffle(i)
data = data[i]
print(data)
print(i)
print(type(i))
print(i.shape)
cv2.waitKey(0)


