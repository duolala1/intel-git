#########################################
# Name : ProtecTooth 护牙系统图像识别功能
# Intro : 使用Alexnet进行图像识别
# 调用 caffe 的 python 接口
# 在 django 中结合该脚本，实现在线口腔图像识别功能
# author : Ft
# date : 2017.3
#########################################
import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys

# 设置图像显示参数（调试相关）
plt.rcParams['figure.figsize'] = (10, 10)        # 显示图像的最大范围
plt.rcParams['image.interpolation'] = 'nearest'  # 差值方式
plt.rcParams['image.cmap'] = 'gray'  # 灰度空间

# caffe 框架加载
caffe_root='../../'
sys.path.insert(0,caffe_root+'python')

# 加载口腔图片识别模型
import os
if os.path.isfile(caffe_root+'models/protectoothcaffenet.caffemodel'):
    print 'net model found.'
else:
    print 'route error'
    # 加载模型（使用cpu进行计算）
    caffe.set_mode_cpu()

    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'  # 网络结构定义文件
    model_weights = caffe_root + 'models/bvlc_reference_caffenet/protectoothcaffenet.caffemodel'  # 加载了caffe的预训练模型

    # 测试模式下进行图片分析
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # 预处理
    mu = np.load(caffe_root + 'python/caffe/pt/pics.npy')
    mu = mu.mean(1).mean(1)  # 计算像素的平均值
    print 'mean-subtracted values:', zip('BGR', mu)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    # 归一化
    transformer.set_transpose('data', (2, 0, 1))  # 分离图像的通道
    transformer.set_mean('data', mu)  # 减去平均像素值
    transformer.set_raw_scale('data', 255)  # 将0-1空间变成0-255空间
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换RGB空间到BGR空间

    net.blobs['data'].reshape(50, 3, 227, 227)  # batchsize=50,三通道，图像大小是227*227

    # 加载图像
    image = caffe.io.load_image(caffe_root + 'test/evalue.jpg')
    traformed_image = transformer.preprocess('data', image)
    plt.imshow(image)

    # 将图像数据拷贝到内存中并分配给网络net
    net.blobs['data'].data[...] = traformed_image

    # 分类，这里用的是caffemodel，这是在imagenet上的预训练模型，有1000类
    output = net.forward()

    output_prob = output['prob'][0]
    print 'predicted class is:', output_prob.argmax()

    # 加载标签
    labels_file = caffe_root + 'data/protectooth/pt_labels.txt'
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    print 'output label:', labels[output_prob.argmax()]

    # 输出概率较大的
    top_inds = output_prob.argsort()[::-1][:1]

    print 'probailities and labels:'
    zip(output_prob[top_inds], labels[top_inds])