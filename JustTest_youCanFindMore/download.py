import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL import Image

# super parameters
DOWNLOAD = True

# ¶šÒåÊýŸÝ±ä»»
transform1 = transforms.ToTensor()  # ¿ÉÒÔ°ÑÏÂÔØµœµÄÊýŸÝ×ª»¯³ÉÕÅÁ¿žñÊœ

# transforms.Compose()¶šÒå¶àÖØÊýŸÝ±ä»¯
transform2 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # ¹éÒ»»¯[-1,1]

mT_trainset = dsets.MNIST(root='./MNIST/Tensor/training', train=True, transform=transform1, download=DOWNLOAD)
mT_testset = dsets.MNIST(root='./MNIST/Tensor/test', train=False, transform=transform1, download=DOWNLOAD)

cT_trainset = dsets.CIFAR10(root='./CIFAR10/Tensor/training', train=True, transform=transform1, download=DOWNLOAD)
cT_testset = dsets.CIFAR10(root='./CIFAR10/Tensor/test', train=False, transform=transform1, download=DOWNLOAD)

mN_trainset = dsets.MNIST(root='./MNIST/Normal/training', train=True, transform=transform2, download=DOWNLOAD)
mN_testset = dsets.MNIST(root='./MNIST/Normal/test', train=False, transform=transform2, download=DOWNLOAD)

cN_trainset = dsets.CIFAR10(root='./CIFAR10/Normal/training', train=True, transform=transform2, download=DOWNLOAD)
cN_testset = dsets.CIFAR10(root='./CIFAR10/Normal/test', train=False, transform=transform2, download=DOWNLOAD)


