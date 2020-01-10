import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
# import torch
import cv2
import copy
import Mymodule.m1
# from torch.autograd import Variable
from tensorflow.python.platform import gfile

batch_size = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
picNum = 500
Class = 6

def readDataT(num):
    paths = os.getcwd() + "/RmDataSet/SentryData/" + str(num) + "/"
    filelist = os.listdir(paths)
    AllDataSub = np.zeros((len(filelist), 24, 32, 1))
    for i in range(len(filelist)):
        print(i, range(len(filelist)), paths + filelist[i])
        I = cv2.imread(paths + filelist[i])
        #cv2.imshow("1", I)
        #cv2.waitKey(1)
        print(I.size, I.shape)
        I = cv2.resize(I, (32, 24))
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)  # 原始图竟然是四通道的！！
        I = I.reshape(24, 32, 1)
        AllDataSub[i, :, :, :] = I
        # print(I)

    return AllDataSub


def readDataL(num):
    paths = os.getcwd() + "/RmDataSet/SentryData/" + str(num) + "L/"
    filelist = os.listdir(paths)
    AllDataSub = np.zeros((len(filelist), 24, 32, 1))
    for i in range(len(filelist)):
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
    paths = os.getcwd() + "/RobomasterNumData/" + str(num) + "S/"
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


# 初始化权重与定义网络结构。
# 这里，我们将要构建一个拥有3个卷积层和3个池化层，随后接1个全连接层和1个输出层的卷积神经网络
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# prepare the data
# AllData1 = readDataL(1)
# AllData2 = readDataL(2)
# AllData3 = readDataL(3)
# AllData4 = readDataL(4)
# AllData5 = readDataL(5)
# AllData6 = readDataL(6)
# AllData7 = readDataL(7)
# AllData8 = readDataL(8)
#
# AllData1s = readDataS(1)
# AllData2s = readDataS(2)
# AllData3s = readDataS(3)
# AllData4s = readDataS(4)
# AllData5s = readDataS(5)
# AllData6s = readDataS(6)
# AllData7s = readDataS(7)
# AllData8s = readDataS(8)
# AllData = np.concatenate((AllData1, AllData2, AllData3, AllData4, AllData5, AllData6, AllData7, AllData8,
#                      AllData1s, AllData2s, AllData3s, AllData4s, AllData5s, AllData6s, AllData7s, AllData8s), 0)

AllData0 = readDataT(0)
AllData1 = readDataT(1)
AllData2 = readDataT(2)
AllData3 = readDataT(3)
AllData4 = readDataT(4)
AllData5 = readDataT(5)
AllData = np.concatenate((AllData0, AllData1, AllData2, AllData3, AllData4, AllData5), 0)

print(AllData.shape)
AllDataTemp = copy.deepcopy(AllData)

# prepare the label
label = np.zeros((picNum * Class), int)
k = 0
for i in range(Class):
    for j in range(picNum):
        label[k] = i
        print(k, "  ", label[k])
        k = k + 1

# for i in range(6*500):
#     cv2.imwrite("./TempData/"+str(i)+str(label[i])+".png", AllData[i, :, :, :])
#     print("./TempData/"+str(i)+str(label[i])+".png")

trY = np.zeros((Class*picNum, Class))
for n in range(Class*picNum):
    for nn in range(Class):
        if nn == label[n]:
            trY[n, nn] = 1

shuffle_kernel = np.arange(0, Class*picNum, 1)
np.random.shuffle(shuffle_kernel)
labeltemp1 = copy.deepcopy(trY)
labeltemp = copy.deepcopy(trY)
AllDataTemp = copy.deepcopy(AllData)
for n in range(Class*picNum):
    AllDataTemp[n, :, :, :] = AllData[shuffle_kernel[n], :, :, :]
    labeltemp1[n, :] = labeltemp[shuffle_kernel[n], :]
    print(labeltemp1[n, :])

trY = labeltemp1
trX = AllDataTemp/255.0
print(trX.shape, trY.shape)

# 在一个会话中启动图，开始训练和评估
# Launch the graph in a session


with tf.Session(graph=tf.Graph()) as sess:
    # you need to initialize all variables
    X = tf.placeholder("float", [None, 24, 32, 1], 'X')
    wConv1 = init_weights([5, 5, 1, 32])
    wConv2 = init_weights([5, 5, 32, 64])
    wConv3 = init_weights([5, 5, 60, 120])
    wlinear4 = init_weights([64 * 6 * 8, 200])
    wlinear5 = init_weights([500, 300])
    wlinear6 = init_weights([200, Class])
    Y = tf.placeholder("float", [None, Class])


    def model(X, Conv1, Conv2, Conv3, linear4, linear5, linear6):
        l1 = tf.nn.relu(tf.nn.conv2d(X, Conv1, strides=[1, 1, 1, 1], padding='SAME'), name='l1r')
        l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='l1p')

        l2 = tf.nn.relu(tf.nn.conv2d(l1, Conv2, strides=[1, 1, 1, 1], padding='SAME'), name='l2r')
        l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='l2p')

        #
        # l3 = tf.nn.relu(tf.nn.conv2d(l2, Conv3, strides=[1, 1, 1, 1], padding='SAME'), name='l3r')
        # l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='l2p')

        l3 = tf.reshape(l2, [-1, linear4.get_shape().as_list()[0]], name='l2l')  # reshape to (?, 2048)


        # 全连接层，最后dropout一些神经元
        l4 = tf.matmul(l3, linear4, name='matmul4')
        # l5 = tf.matmul(l4, linear5, name='matmul5')
        # 输出层
        pyx = tf.matmul(l4, linear6, name='Y_')
        return pyx


    # 我们定义dropout的占位符——keep_conv，它表示在一层中有多少比例的神经元被保留下来。生成网络模型，得到预测值
    py_x = model(X, wConv1, wConv2, wConv3, wlinear4, wlinear5, wlinear6)  # 得到预测值

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    tf.global_variables_initializer().run()

    for i in range(10):
        training_batch = zip(range(0, len(trX)-200, batch_size), range(batch_size, len(trX) + 1-200, batch_size))
        for start, end in training_batch:
            # print(start, end)
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        for n in range(Class*picNum-200, Class*picNum):
            k = sess.run(predict_op, feed_dict={X: trX[n:n + 1]})
            print(i, np.argmax(trY[n], axis=0), k, np.argmax(trY[n], axis=0) == k)

    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['X', 'Y_'])
    with tf.gfile.FastGFile("./TrainedModel/Sentry_New.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())

