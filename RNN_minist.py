import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# 超参数
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step  / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.01
DOWNLOWD_MNIST = True  # 如果没有下载好MNIST数据，设置为True

# 下载数据
# 训练数据
train_data = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOWD_MNIST)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# print(train_data.train_data.shape) 60000*28*28

# 测试数据
test_data = datasets.MNIST(root='./mnist', train=False)
# print(test_data.test_data.shape)大小10000*28*28
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000] / 255.  # size 2000*28*28

test_y = test_data.test_labels.numpy()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # rnn 运行的结果出了每层的输出之外，还有该层要传入下一层进行辅助分析的hidden state,
        # lstm 的hidden state相比于 RNN，其分成了主线h_n,分线h_c
        r_out, (h_n, h_c) = self.rnn(x, None)  # x shape ( batch, step, input_size), None 之前的hidden state（没有则填None）
        # print(r_out.shape) 64* 28*64
        # print(h_c.shape)  1*64*64
        # print(h_n.shape)   1*64 *64
        # print(r_out[:, -1, :].shape)
        out = self.out(r_out[:, -1, :])  # 选取最后一个时刻的output，进行最终的类别判断
        # print(out.shape)
        return out


rnn = RNN()
# print(rnn)

# 优化器
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# 误差函数
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # X的size 64*1*28*28
        b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch, time_step, input_size)
        # b_x size 64*28*28
        b_y = Variable(y)
        # b_y size 64
        # print(b_y)
        output = rnn(b_x)
        # print(output)
        # print(output.shape) 64*10
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = np.squeeze(torch.max(test_output, 1)[1].data.numpy())
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, ' | train loss: %.4f' % loss.data.numpy(), ' | test accuracy: %.2f' % accuracy)
            print(step)

# 输出前10个测试数据的测试值
test_output = rnn(test_x[: 10].view(-1, 28, 28))
pred_y = np.squeeze(torch.max(test_output, 1)[1].data.numpy())
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
