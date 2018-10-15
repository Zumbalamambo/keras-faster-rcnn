# -*- coding:utf8 -*-

import sys
import time
import random
import pprint
import pickle
import numpy as np
from optparse import OptionParser


import keras
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils

import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses

# python中递归深度是有限制的,默认为999
sys.setrecursionlimit(40000)

parser = OptionParser()

# 从命令行输入的训练参数
parser.add_option()
parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.", default="data/")
parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)
parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).", default="config/config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='model/model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
(options, args) = parser.parse_args()

# 若没有指定文件名
if not options.train_path:
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")



# 通过命令行传入配置信息, 并保存到config对象中
C = config.Config()
# bool()是int的子类, 将给定参数转换为布尔类型，如果没有参数，返回 False
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.model_path = options.output_weight_path
C.rot_90 = int(options.rot_90)

# 判断使用的网络
if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
if options.network == 'resnet50':
    C.network = 'resnet50'
    from keras_frcnn import resnet as nn
else:
    print('No valid model')
    raise ValueError

# 检查权重文件是否来自命令行
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    C.base_net_weights = nn.get_weight_path()

# 读取voc数据集的所有图片，类别统计，类的映射
all_imgs, classes_count, class_mapping = get_data('')
# 若没有背景类，向里面添加背景类
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    # 背景所在的类别映射是最大的
    class_mapping = len(classes_count)

C.class_mapping = class_mapping
# 以列表返回可遍历的(键, 值) 元组数组
inv_class_mapping = {v: k for k, v in class_mapping.items()}
print('Training_imgs per class:')
pprint.pprint(classes_count)
print('Num classes(bg) = {}'.format(len(classes_count)))

# open打开文件，默认是只读模式，现在是写模式
# pickle.dump：将对象obj保存到文件file中去
# 测试时pickle.load()加载python对象,将文件重新存储为类
config_output_filename = options.output_filename
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded '
          'when testing to ensure correct results'.
          format(config_output_filename))

# 打乱、随机选取图片
num_imgs = len(all_imgs)
random.shuffle(all_imgs)
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train_imgs: {}'.format(len(train_imgs)))
print('Num val_imgs: {}'.format(len(val_imgs)))

# 得到每一个anchor的训练数据，供RPN网络训练使用
data_gen_train = data_generators.get_anchor_gt(
    train_imgs, classes_count, C, nn.get_img_output_length(), K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(
    val_imgs, classes_count, C, nn.get_img_output_length(),K.image_dim_ordering(), mode='val')
#
if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (3, None, None)

#-----------------------------------------------------------------------------------------------
# 1、定义网络的输入和输出
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))
# define base network
shared_layers = nn.nn_base(img_input, trainable=True)
# define RPN built on base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
# 定义网络最后的 分类和框回归
classifier = nn.classifier(shared_layers, roi_input, C.num_rois,
                           nb_classes=len(classes_count), trainable=True)


# 2、构建网络，加载参数
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)
# 将RPN、classifier倆个list合并，方便后面权重的加载
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# 尝试加载权重,错误则抛出异常
try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras '
          'application folder https://github.com/fchollet/keras/tree/master/keras/applications')


# 3、训练之前的编译
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(len(classes_count))],
                  metrics={'dense.class_{}'.format(len(classes_count)): 'accuracy'})
# metrics 给这一层增加一个accuracy参数,
model_all.compile(optimizer='sgd', loss='mae')


# 4、设定训练的参数
epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []  # 记录每一次训练正框的个数
rpn_accuracy_for_epoch = []
start_time = time.time()
best_loss = np.Inf  # inf表示一个无限大的正数

class_mapping_inv = {v:k for k, v in class_mapping.items()}
print('Start training')


vis = True
for epoch_num in range(num_epochs):
    # keras的工具类，generic_utils.Progbar，创建一个类，要指定进度条长度
    # 进度条,动态显示每个epoch的训练情况
    progbar = generic_utils.Progbar(epoch_length)
    print('epoch {}/{}'.format(epoch_num+1, num_epochs))

    while True:
        # 循环中加入try..catch..模块，以便异常时抛出异常并进入下一次循环，防止多次循环中出错
        try:
            # 用来监督每一次epoch的平均正回归框的个数
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                # 显示每个epoch的平均正框个数
                mean_overlapping_bboxes = len(rpn_accuracy_rpn_monitor) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'
                    .format(mean_overlapping_bboxes, epoch_length))
                # 正框为0时，抛出异常
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. '
                          'Check RPN settings or keep training.')

            # -------------------------训练RPN网络、Classifier网络-------------------------------------
            # RPN和Classifier这两个网络交替训练，

            X, Y, img_data = next(data_gen_train)  # X是图片、Y是对应类别和回归梯度
            loss_rpn = model_rpn.train_on_batch(X, Y)
            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(),
                                       use_regr=True, overlap_thresh=0.7, max_boxes=300)
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
            if X2 == None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue






            # update(当前次数,[(名称, 数值), (名称, 数值)]),显示当前的训练状态
            # losses数组，是每个epoch前n次已经训练后的平均loss，并非当次训练的loss
            progbar.update(iter_num,[('rpn_cls', np.mean(losses[:iter_num, 0])),
                                     ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                     ('detector_cls', np.mean(losses[:iter_num, 2])),
                                     ('detector_regr', np.mean(losses[:iter_num, 3]))])



