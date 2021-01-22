from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import cv2
import numpy as np
import threading as th
import sys
import glob


class Myui(QWidget):
    def __init__(self,parent = None):
        super().__init__()

        self.lable = QLabel()
        # self.lable1 = QLabel()
        self.lablejg = QLabel()
        self.lablejd = QLabel()
        self.lableacc = QLabel()
        self.lableapach = QLabel()
        self.Linefile = QLineEdit()
        self.Linefile2 = QLineEdit()

        # self.lable.setText("暂无数据")
        self.lablejd.setText("模型训练进度:")
        self.lableacc.setText("训练准确率:  ")
        self.lableapach.setText("测试图片路径:")
        self.lablejg.setText("检测结果:")

        self.toolBtn = QPushButton("...")
        self.strBtn = QPushButton("开启训练")
        self.endBtn = QPushButton("返回")
        self.layout = QGridLayout()
        self.setWindowTitle("数字识别系统")
        # self.lable.setText("创久")
        # self.imagesz = QPixmap()
        # self.imagesz.
        self.initimage()
        self.progjd = QProgressBar()
        self.progjacc = QProgressBar()

        # self.layout.addWidget(self.lable,0,0,1,0)
        self.layout.addWidget(self.lable,0,0,1,1)
        self.layout.addWidget(self.lablejg,0,1,1,1)
        self.layout.addWidget(self.Linefile2, 0, 2, 1, 2)
        self.layout.addWidget(self.lableapach, 1,0, 1, 1)
        self.layout.addWidget(self.Linefile, 1, 1,1,3)
        self.layout.addWidget(self.toolBtn,1,4,1,1)
        self.layout.addWidget(self.lablejd,2,0,1,1)
        self.layout.addWidget(self.progjd, 2, 1, 1, 4)
        self.layout.addWidget(self.lableacc, 3, 0, 1, 1)
        self.layout.addWidget(self.progjacc, 3, 1, 1, 4)
        self.layout.addWidget(self.strBtn, 4, 0, 1, 2)
        self.layout.addWidget(self.endBtn, 4, 3, 1, 2)
        # self.layout.addWidget(self.toolBtn, 0, 4, 1, 1)

        # self.layout.addWidget(self.toolBtn, 2, 3, 2, 2)
        # self.layout.addWidget(self.prog,2,2)

        self.setLayout(self.layout)
        self.toolBtn.clicked.connect(self.fileshow)
        self.strBtn.clicked.connect(self.Learn)
        self.endBtn.clicked.connect(self.endFun)
        self.setGeometry(400,500,600,400)
        self.show()

    def initimage(self):
        #print("66666")
        self.imagezw = QPixmap("./zanwu.jpg")
        self.imagezw1 = self.imagezw.scaled(56, 56)
        #print("666677")
        self.lable.setPixmap(self.imagezw1)

    def fileshow(self):
        self.fliename = QFileDialog.getOpenFileName()
        self.im = self.fliename[0]
        self.Linefile.setText(self.im)
        self.imagesz = QPixmap(self.im)
        self.imagesz1 = self.imagesz.scaled(56,56)
        self.lable.setPixmap(self.imagesz1)

    windowList = []
    def endFun(self):
        myshuzi = Myshuzi()
        self.windowList.append(myshuzi)
        self.close()

    def Learn(self):
        if self.Linefile.text() == '':
            QMessageBox.information(self,"提示","测试图片路径为空！")
        else:
            self.Linefile2.setText("")
            thre = th.Thread(target=self.covn)
            thre.start()

        # thre = th.Thread(target=self.covn)
        # thre.start()

    # #########################################
    def read_image(self,path):

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # 1.图片必须为正方形
        # 2.必须为28*28*1
        # 3.必须灰色

        image1 = cv2.resize(image, dsize=(28, 28))
        image1 = np.resize(image1, new_shape=(1, 784))

        return image, image1

    def w_rand(self,shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return w

    # 随机生成偏移b:
    def b_rand(self,shape):
        b = tf.Variable(tf.random_normal(shape=shape))
        return b

    def model(self):
        #生成x与预测值y_dome:
        with tf.variable_scope("data"):
            x_true = tf.placeholder(tf.float32,[None,784])
            y_true = tf.placeholder(tf.int32, [None, 10])

        # 卷积1
        with tf.variable_scope("covn1"):
            #改变x: [None,784]------>[-1,28,28,1]
            x_reshape = tf.reshape(x_true,[-1,28,28,1])

            #接收第一层卷积权重和偏移:
            w_conv1 = self.w_rand(shape = [5,5,1,32])
            b_conv1 =self.b_rand(shape = [32])
            #第一层卷积:
            x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape,w_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1)
             #池化
            x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        # 卷积2
        with tf.variable_scope("covn2"):
            # 接收第二层卷积权重和偏移:
            w_conv2 = self.w_rand([5,5,32,64])
            b_conv2 = self.b_rand([64])

            # 第二层卷积:
            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
             # 池化
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("covn3"):

            w =self.w_rand([7 * 7 * 64, 10])
            b = self.b_rand([10])
            x = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

            y_pre = tf.matmul(x , w) + b

        return x_true,y_true,y_pre

    def covn(self):

        mnist_data = input_data.read_data_sets("data\mnist\input_data", one_hot=True)
        x, y_true, y_pre = self.model()

        # 交叉熵
        with tf.variable_scope("soft_loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pre))

        # 梯度下降
        with tf.variable_scope("Optimzer"):
            train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
            # train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # 计算准确率
        with tf.variable_scope("acc"):
            equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pre, 1))
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        init = tf.global_variables_initializer()

        # saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(500):
                epoch_loss = 0
                # for e in range(10):
                x_data, y_data = mnist_data.train.next_batch(50)
                # print("22222222222222222", x_data)
                iter_loss, _ = sess.run([loss, train_op], feed_dict={x: x_data, y_true: y_data})
                epoch_loss += iter_loss
                # self.prog.setValue((i+1)*10)
                self.progjd.setValue(((i+1)/500)*100)

                print("epoch", i, ":  ", epoch_loss)

                acc = sess.run(accuracy, feed_dict={x: x_data, y_true: y_data})

                self.progjacc.setValue(acc*100)

                print("  准确率为: ", acc)
                # project_image("1.jpg",sess,y_pre)

            image, image1 = self.read_image(self.Linefile.text())
            # print("666666666666666", image1)
            res = sess.run(y_pre, feed_dict={x: image1})
            # print(res)
            res = np.argmax(res, 1)

            # print(res)
            shuzi = "这个数字是: "+str(res[0])
            self.Linefile2.setText(shuzi)

class Myyzm(QWidget):
    def __init__(self,parent = None):
        super().__init__()

        self.lable = QLabel()
        # self.lable1 = QLabel()
        self.lablejg = QLabel()
        self.lablejd = QLabel()
        self.lableacc = QLabel()
        self.lableapach = QLabel()
        self.Linefile = QLineEdit()
        self.Linefile2 = QLineEdit()

        # self.lable.setText("暂无数据")
        self.lablejd.setText("模型训练进度:")
        self.lableacc.setText("训练准确率:  ")
        self.lableapach.setText("测试图片路径:")
        self.lablejg.setText("检测结果:")

        self.toolBtn = QPushButton("...")
        self.strBtn = QPushButton("开启训练")
        self.endBtn = QPushButton("返回")
        self.layout = QGridLayout()
        self.setWindowTitle("验证码识别系统")
        # self.lable.setText("创久")
        # self.imagesz = QPixmap()
        # self.imagesz.
        self.initimage()
        self.progjd = QProgressBar()
        self.progjacc = QProgressBar()

        # self.layout.addWidget(self.lable,0,0,1,0)
        self.layout.addWidget(self.lable, 0, 0, 1, 1)
        self.layout.addWidget(self.lablejg, 0, 1, 1, 1)
        self.layout.addWidget(self.Linefile2, 0, 2, 1, 2)
        self.layout.addWidget(self.lableapach, 1, 0, 1, 1)
        self.layout.addWidget(self.Linefile, 1, 1, 1, 3)
        self.layout.addWidget(self.toolBtn, 1, 4, 1, 1)
        self.layout.addWidget(self.lablejd, 2, 0, 1, 1)
        self.layout.addWidget(self.progjd, 2, 1, 1, 4)
        self.layout.addWidget(self.lableacc, 3, 0, 1, 1)
        self.layout.addWidget(self.progjacc, 3, 1, 1, 4)
        self.layout.addWidget(self.strBtn, 4, 0, 1, 2)
        self.layout.addWidget(self.endBtn, 4, 3, 1, 2)
        # self.layout.addWidget(self.toolBtn, 0, 4, 1, 1)

        # self.layout.addWidget(self.toolBtn, 2, 3, 2, 2)
        # self.layout.addWidget(self.prog,2,2)

        self.setLayout(self.layout)

        self.strBtn.clicked.connect(self.Learn)
        self.toolBtn.clicked.connect(self.fileshow)
        self.endBtn.clicked.connect(self.endFun)
        self.setGeometry(400, 500, 600, 400)
        self.show()

    def initimage(self):
        print("66666")
        self.imagezw = QPixmap("./zanwu.jpg")
        self.imagezw1 = self.imagezw.scaled(110, 27)
        print("666677")
        self.lable.setPixmap(self.imagezw1)

    def fileshow(self):
        self.fliename = QFileDialog.getOpenFileName()
        self.im = self.fliename[0]
        self.Linefile.setText(self.im)
        self.imagesz = QPixmap(self.im)
        self.imagesz1 = self.imagesz.scaled(110,27)
        self.lable.setPixmap(self.imagesz1)
        return self.im

    windowList = []
    def endFun(self):
        myshuzi = Myshuzi()
        self.windowList.append(myshuzi)
        self.close()


    def Learn(self):
        if self.Linefile.text() == '':
            QMessageBox.information(self,"提示","测试图片路径为空！")
        else:
            self.Linefile2.setText("")
            thre = th.Thread(target=self.open_session)
            thre.start()


    def read_image(self):

        # 1.构建文件名队列：
        # 构建文件名列表：
        file_name = glob.glob("./GenPics//*.jpg")  ##################################
        # print("file_name: ",file_name)

        # 构建文件名队列：
        file_queue = tf.train.string_input_producer(file_name)

        # 2.读取与编码
        # 读取阶段:
        reader = tf.WholeFileReader()
        filename, image = reader.read(file_queue)

        # 解码阶段:
        decoded = tf.image.decode_jpeg(image)
        decoded.set_shape([20, 80, 3])  # 确定图片形状，[高，宽，通道]
        # print(decoded)
        image_cast = tf.cast(decoded, tf.float32)

        # 批处理
        filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=100, num_threads=1,
                                                     capacity=200)

        return filename_batch, image_batch

    def parse_csv(self):

        csv_data = pd.read_csv("./GenPics/labels.csv", names=["file_num", "chars"], index_col="file_num")

        lables = []
        for lable in csv_data["chars"]:
            letter = []
            for word in lable:
                letter.append(ord(word) - ord('A'))
            lables.append(letter)

        csv_data["lables"] = lables

        return csv_data

    def filename2lable(self,filename, csv_data):
        # 将特征值与目标值一一对应
        labels = []
        # 提取filename中的数字 例如GenPics/666.jpg中的666数字
        for file_name in filename:
            file_num = "".join(list(filter(str.isdigit, str(file_name))))
            target = csv_data.loc[int(file_num), "lables"]
            labels.append(target)
        # print(labels)
        # print(csv_data)
        # print(labels)

        return np.array(labels)

    def w_rand(self,shape):
        w = tf.Variable(tf.random_normal(shape=shape, stddev=0.01))
        return w

    # 随机生成偏移b:
    def b_rand(self,shape):
        b = tf.Variable(tf.random_normal(shape=shape, stddev=0.01))
        return b

    def create_cnn(self):
        with tf.variable_scope("data"):
            x = tf.placeholder(tf.float32, shape=[None, 20, 80, 3])
            y_true = tf.placeholder(tf.float32, shape=[None, 4 * 26])
            # y_true = tf.placeholder(tf.int32, shape=[None, 4 * 26])
        with tf.variable_scope("covn1"):
            # 接收第一层卷积权重和偏移:
            w_conv1 = self.w_rand([5, 5, 3, 32])
            b_conv1 = self.b_rand([32])
            # 第一层卷积:
            x_relu1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
            # 池化
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 卷积2
        with tf.variable_scope("covn2"):
            # 接收第二层卷积权重和偏移:
            w_conv2 = self.w_rand([5, 5, 32, 64])
            b_conv2 = self.b_rand([64])

            # 第二层卷积:
            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
            # 池化
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        with tf.variable_scope("fc"):
            w_fc = self.w_rand([5 * 20 * 64, 4 * 26])
            b_fc = self.b_rand([4 * 26])
            x_fc = tf.reshape(x_pool2, [-1, 5 * 20 * 64])

            y_pre = tf.matmul(x_fc, w_fc) + b_fc

        return x,y_true,y_pre

    def read_image0(self,path):

        image = cv2.imread(path)

        # 1.图片必须为80*20(宽*高)
        # 2.必须为[1,20,80,3]

        image1 = cv2.resize(image, dsize=(80, 20))
        image1 = np.resize(image1, new_shape=(1, 20, 80, 3))

        return image, image1

    def showyzm(self,sess,y_pre,x):
        im = self.Linefile.text()
        image0, image1 = self.read_image0(im)
        # print("6666666",image1)

        res = sess.run(y_pre, feed_dict={x: image1})
        # print(res)
        res = tf.reshape(res, shape=[4, 26])
        # print(sess.run(res))
        res1 = []
        res2 = []
        res3 = []
        res4 = []

        res1 = res[0]
        res2 = res[1]
        res3 = res[2]
        res4 = res[3]

        res1 = tf.reshape(res1, shape=[1, 26])
        res2 = tf.reshape(res2, shape=[1, 26])
        res3 = tf.reshape(res3, shape=[1, 26])
        res4 = tf.reshape(res4, shape=[1, 26])

        keyword1 = sess.run(tf.argmax(res1, -1))
        keyword2 = sess.run(tf.argmax(res2, -1))
        keyword3 = sess.run(tf.argmax(res3, -1))
        keyword4 = sess.run(tf.argmax(res4, -1))
        yzm1 = chr(int(keyword1) + 65)
        yzm2 = chr(int(keyword2) + 65)
        yzm3 = chr(int(keyword3) + 65)
        yzm4 = chr(int(keyword4) + 65)

        yzm = '这个验证码是:'+yzm1+' '+yzm2+' '+yzm3+' '+yzm4

        self.Linefile2.setText(str(yzm))
        # print(chr(int(keyword1) + 65), end="")
        # print(chr(int(keyword2) + 65), end="")
        # print(chr(int(keyword3) + 65), end="")
        # print(chr(int(keyword4) + 65))


    def open_session(self):

        filename, image = self.read_image()
        csv_data = self.parse_csv()
        x,y_true,y_pre = self.create_cnn()


        with tf.variable_scope("soft_loss"):
            # tf.cast(y_true, tf.float32)
            loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pre)
            # loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pre)
            loss = tf.reduce_mean(loss_list)

        with tf.variable_scope("optimizer"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        with tf.variable_scope("acc"):
            y_pre_shape = tf.reshape(y_pre, shape=[-1, 4, 26])
            y_true_shape = tf.reshape(y_true, shape=[-1, 4, 26])
            # tf.argmax(y_pre_shape,axis=2)
            # tf.argmax(y_true_shape,axis=2)
            bool_equ = tf.equal(tf.argmax(y_pre_shape, axis=2), tf.argmax(y_true_shape, axis=2))
            equal_list = tf.reduce_all(bool_equ, axis=1)
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))



        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # 创建线程
            sess.run(init)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)

            # 开启会话
            for i in range(150):
                filename_new, image_new = sess.run([filename, image])
                # print("filename: ",filename_new)
                # print("image: ",image_new)  #9
                labels = self.filename2lable(filename_new, csv_data)
                # print(image_new)

                # 将标签转换为one_hot
                labels_value = tf.reshape(tf.one_hot(labels, depth=26), [-1, 4 * 26]).eval()

                _, error, accuracy_value = sess.run([optimizer, loss, accuracy],
                                                    feed_dict={x: image_new, y_true: labels_value})
                self.progjd.setValue(((i + 1) /150)* 100)

                print("第%d次训练后损失为%f，准确率为%f" % (i + 1, error, accuracy_value))
                self.progjacc.setValue(accuracy_value*100)

            self.showyzm(sess,y_pre,x)

            # 释放线程
            coord.request_stop()
            coord.join()


class Myshuzi(QWidget):
    def __init__(self,parent = None):
        super().__init__()
        self.setFixedSize(230,230)
        self.setWindowTitle("目标检测")
        self.Lablebt = QLabel()
        self.Lablebt.setText("根据需求进入下列不同的界面")

        self.shuziBtn = QPushButton("进入手写数字识别")
        self.yzmBtn = QPushButton("进入验证码识别")


        self.layout = QGridLayout()
        self.layout.addWidget(self.Lablebt, 0, 1, 1, 2)
        self.layout.addWidget(self.shuziBtn,1,1,1,2)
        self.layout.addWidget(self.yzmBtn, 2, 1, 1, 2)

        self.setLayout(self.layout)

        self.shuziBtn.clicked.connect(self.shuzi)
        self.yzmBtn.clicked.connect(self.yzm)
        self.show()


    windowList = []
    def shuzi(self):
        myui = Myui()
        self.windowList.append(myui)
        self.close()

    windowList1 = []
    def yzm(self):
        myyzm = Myyzm()
        self.windowList1.append(myyzm)
        self.close()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    foin = Myshuzi()
    sys.exit(app.exec_())
