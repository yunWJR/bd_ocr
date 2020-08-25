# -*- coding: utf-8 -*-
# @Time    : 2019/6/24
# @Author  : 椰汁雪饼
# @Email   : chenxiaobing12@139.com

# 基于 Advanced EAST 的文本检测

import numpy as np
from PIL import Image, ImageDraw
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import cfg
from network import East
from nms import nms
from preprocess import resize_image


# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# AdvancedEAST 模型
def east_detect():
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    return east_detect


# 基于 Advanced EAST 的文本检测
# 输入：AdvancedEAST模型，图片路径，像素分类阈值
# 返回：检测后文本框的位置信息
def text_detect(east_detect, img_path, pixel_threshold=0.9):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')

    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    bboxes = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            bboxes.append(rescaled_geo_list)
        # if np.amin(score) == 0:
        #     rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
        #     rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
        #     bboxes.append(rescaled_geo_list)

    return bboxes


if __name__ == '__main__':

    # 图片路径
    img_path = '/Users/yun/Downloads/ocr.jpg'
    # img_path = 'demo/012.png'

    # 加载模型
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    # 文本检测
    bboxes = text_detect(east_detect, img_path, pixel_threshold=0.8)

    # 绘制边框
    img = Image.open(img_path)
    line_width, line_color = 3, 'green'
    draw = ImageDraw.Draw(img)
    for box in bboxes:
        draw.line([(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7]), (box[0], box[1])],
                  width=line_width, fill=line_color)

    img.show()
