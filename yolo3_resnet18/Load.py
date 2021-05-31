# -*- coding: utf-8 -*-
from skimage import io, color
import numpy as np
import re
import cv2

pos_ = {'identification': 0, 'disc': 1, 'vertebra': 1}  # 这个元组用来处理txt文件里给的标记顺序不一样的情况，从txt里读出是什么器官，用这个来量化


def HPsorted(Y):
    for i in range(len(Y)):
        for j in range(len(Y) - i - 1):
            if Y[j+1][1] > Y[j][1]:
                tmp = Y[j]
                Y[j] = Y[j+1]
                Y[j+1] = tmp
    return Y


def DataLoad(Path):
    # 初始图片load进来
    files = sorted(io.ImageCollection(Path + '/*.jpg').files)
    X = []
    Labels = []
    Y = []
    for i in files:
        tStr = list(i)
        tStr[-3:] = ['t', 'x', 't']
        tStr = ''.join(tStr)  # tStr（textString）用于找到对应图片的txt文件
        with open(tStr) as f:
            lines = f.readlines()
            X.append(i)
            Labels.append(lines)
    for txtf in Labels:
        tmpy = []
        for line in txtf:  # 把txt中的一行分成三段处理，处理各段的代码已经用空格隔开
            p1 = re.search(r",{'", line).span()  # 第一段：句首到右大括号
            tmp = line[:p1[0]]  # 截取得到坐标
            dot = re.search(r',', tmp).span()
            y = [0,0,0,0]
            y[0] = int(tmp[:dot[0]])
            y[1] = int(tmp[dot[1]:])  # 处理第一段的代码，主要获取标记的坐标

            line = line[p1[-1]:-3].split("', '")
            line = [a.split("': '") for a in line]
            for each in line:
                if len(each[0]) != 0:
                    key_ = each[0]
                    val_ = each[1]
                    y[2 + pos_[key_]] = val_

            tmpy.append(y)
        Y.append(HPsorted(tmpy))
    return X, Y


def Datashow(x,y,save = False):
    # 用来打印图片和标记，一次打印一张，x是脊椎图片本身，y是对应的标记
    img = color.grey2rgb(x)
    for i in y:
        cv2.drawMarker(img,(i[0],i[1]),(0,255,0),markerType = 3,markerSize = 8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        slab = i[2]
        cv2.putText(img,slab,(i[0]+len(slab),i[1]),font,0.5,(0,255,0),1)
    cv2.imshow('', img)
    cv2.waitKey(0)
    return 0


def getFrame(Ytag, bias=0):
    tmp = np.array([int(i[1]) for i in Ytag])
    aver_y = np.mean(tmp[:-2] - tmp[1:-1])
    len_x = 2*aver_y
    lab = []
    for obj in Ytag:
        lab.append([obj[0] - len_x/2 - bias, obj[1] - aver_y/2 - bias, obj[0] + len_x/2 + bias, obj[1] + aver_y/2 + bias, obj[2], obj[3]])
    return lab
