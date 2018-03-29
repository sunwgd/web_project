from django.shortcuts import render
from django import forms
from django.http import HttpResponse
from tests.models import Images
from tests import models
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from PIL import Image





def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # 初始化权重
        self.weights = []
        # 权重必须从输出层的后一层相乘weight，从1层开始
        for i in range(1, len(layers) - 1):
            # 第一次相乘weight(行数是第一层的的个数，列数是二层的列数(行数为神经元数))，+1是加一个偏置
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    # 开始正向推算和反向传播更新weight
    def fit(self, X, y, learning_rate=0.2, epochs=2000):
        self.weights[0]=np.loadtxt('./static/weights0.txt',delimiter=' ')
        self.weights[1]=np.loadtxt('./static/weights1.txt',delimiter=' ')
        # # 首先确定X的行数是否为两行，因为数据一行不能更新weight
        # X = np.atleast_2d(X)
        # # 解决x的偏指加一列
        # temp = np.ones([X.shape[0], X.shape[1] + 1])
        # temp[:, 0:-1] = X
        # X = temp
        # # 确定y是一个矩阵
        # y = np.array(y)
        # # 开始循环更新weights
        # for k in range(epochs):
        #     # 随机从x的行数中取一行
        #     i = np.random.randint(X.shape[0])
        #     a = [X[i]]
        #     # 进入神经元进行正向乘权重
        #     # 得到两层的a 存入数组a
        #     for l in range(len(self.weights)):
        #         a.append(self.activation(np.dot(a[l], self.weights[l])))
        #     # 计算最后一组a的误差
        #     error = y[i] - a[-1]
        #     # 计算deal
        #     dealta = [error * self.activation_deriv(a[-1])]
        #     # 开始反向传播带入误差dealtas的公式求出前一项的误差最后求到第一层
        #     # 反向循环，求出的dealta储存
        #     for l in range(len(a) - 2, 0, -1):
        #         dealta.append(dealta[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
        #     # 反转dealta
        #     dealta.reverse()
        #     # 循环更新权重 ， 根据weights数组的长度
        #     for i in range(len(self.weights)):
        #         # 判断每一层的 a 和 dealta 是否为两行
        #         layer = np.atleast_2d(a[i])
        #         dealtas = np.atleast_2d(dealta[i])
        #         # 带入更新dealta的公式
        #         self.weights[i] += learning_rate * layer.T.dot(dealtas)
        # for i in range(len(self.weights)):
        #     np.savetxt('./static/weights' + str(i) + '.txt', self.weights[i])
    # 结果预测
    def predicr(self, x):
        # 判断x是否为矩阵
        x = np.array(x)
        # 得到一个x行的一个1矩阵，将x数组加入偏执 1

        temp = np.ones(x.shape[0] + 1)
        #        print(temp[0:-1])
        #        print(x)
        #        temp[0:-1]=x
        a = np.column_stack((temp[0:-1], x))
        #        print(a)
        # 循环计算每一组的预测值
        for i in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[i]))
        return a
class PictureProcessing:
    def __init__(self,filename):
        self.filename=filename
    def ImageToMatrix(self,im=''):
        if self.filename!=None:
            im = Image.open(self.filename)
            im=im.resize((93,70))
        width,height = im.size
        im = im.convert("L")
        im=im.rotate(-90)
        data = im.getdata()
        data = np.matrix(data,dtype='float')/255.0
        new_data = np.reshape(data,(width,height))
        return new_data
    def readMarir(self):
        data = self.ImageToMatrix(self.filename)
        data=data.reshape(1,93*70)
        return data


class NeuralText:
    def __init__(self,datafile,filename):
        self.datafile=datafile
        self.filename=filename
    def result(self):
        data=np.loadtxt(self.datafile,delimiter=',')
        nn=PictureProcessing(self.filename)
        data1=nn.readMarir()
        test_x=data1
        test_x-=test_x.min()
        test_x/=test_x.max()
        x=data[:,:-1]
        y=data[:,-1]
        x-=x.min()
        x/=x.max()
        nn=NeuralNetwork([6510,200,5],'logistic')
        lable_tran=LabelBinarizer().fit_transform(y)
        nn.fit(x,lable_tran)
        result=[]
        for i in range(test_x.shape[0]):
            o=nn.predicr(test_x[i])
            result.append(np.argmax(o))
        return result
def index(request):
    return render(request,'tests/index.html')
class ImageUploadForm(forms.Form):
    img=forms.FileField()

def predict(request):
    if request.method == 'POST':
        # name=request.POST.get('name')
        # type=request.POST.get('type')
        myFile = request.FILES.get("image", None)
        if os.path.exists('./static')==False:
            os.makedirs('./static')
        if not myFile:
            returnHttpResponse("no files for upload!")
        destination = open(os.path.join("./static", myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()
        img = "./static/" + myFile.name
        models.Images.objects.create(img=img)
        datafile = './data.txt'
        filename = img
        nn = NeuralText(datafile, filename)
        result = nn.result()
        print('result',result)
        # print('result_type',type(result))
        if result==[0]:
            s='上'
        elif result ==[1]:
            s='右上'
        elif result ==[2]:
            s='右'
        elif result==[3]:
            s='右下'
        else :
            s='下'
        # return HttpResponse(s)

    return render(request, 'tests/predict.html',{'s': s,'img':img})
