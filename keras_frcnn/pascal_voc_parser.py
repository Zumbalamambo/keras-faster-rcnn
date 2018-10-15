import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np


# 输入：input_path，给定VOC所在文件夹路径，不需要具体到版本
def get_data(input_path):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    visualise = False

    # ----------------------------1、基本信息设置--------------------------------------------

    # 依次读取VOC2007、VOC2012，若只使用一个删除另一个即可
    data_paths = [os.path.join(input_path, s) for s in ['VOC2007', 'VOC2012']]

    print('Parsing annotation files')

    for data_path in data_paths:

        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        # 训练集、测试集中图片编号的集合
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        trainval_files = []
        test_files = []

        # 两个try..except..模块是为了得到训练、测试集图片文件的名称，便于以后判断图片是属于哪个集

        # try..except..模块，若try内函数异常，则执行except内函数，而不是报错
        try:
            # open默认为只读模式，其实 f 是一个'_io.TextIOWrapper'类
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')
        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        # --------------------------2、读取XML文件----------------------------------------

        # os.listdir 是所有文件名称 的集合
        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        idx = 0
        for annot in annots:
            try:
                idx += 1
                # ET 专门的包来处理xml文件
                et = ET.parse(annot)
                # 得到xml的根，所有根包含的属性都可以从中得到
                element = et.getroot()
                # 取出xml文件中所有的object，find方法得到xml的相应标签
                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    # 图片的路径、宽度、高度、框、所属训练集
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename),
                                       'width': element_width, 'height': element_height, 'bboxes': []}
                    # 判断一个名称，是否包含在一个列表里
                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    # 类别若不在列表里，加至表中，计数+1
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        # 类别在列表里，计数+1
                        classes_count[class_name] += 1
                    # 类别映射若不在表中，加至表中，映射为表的长度
                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    # 向bboxes添加的字典信息包括：类别、左上角与右下角坐标、难度
                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                all_imgs.append(annotation_data)

                # 是否需要读一张图片显示一张图片
                if visualise:
                    img = cv2.imread(annotation_data['filepath'])
                    for bbox in annotation_data['bboxes']:
                        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[ 'x2'], bbox['y2']), (0, 0, 255))
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            except Exception as e:
                print(e)
                continue
    # 函数的输出
    return all_imgs, classes_count, class_mapping


""" 
1、all_imgs：
数据格式为list,每一条信息是以字典形式存储包含了一张图片的所有信息
(1) 字典包含：图片的高度，宽度，路径，所处训练集, bboxes。
(2) bboxes: list,每条信息是以字典存储了一个box的所有信息,包括上下两点坐标、类别、难度

2、classes_count:
是一个字典，其存储类别 和 其对应的总个数

3、class_mapping:
是一个字典，存储类别 和 每一个类别对应的编号
"""
