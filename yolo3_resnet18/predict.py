'''
predict.py有几个注意点
1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
3、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
4、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
在原图上利用矩阵的方式进行截取。
5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
'''
import sys
sys.path.append('/Users/wang/PycharmProjects/yolo3-pytorch-master/Res')
from PIL import Image
import numpy as np
from yolo import YOLO
#训练YOLO的
import Res.train
#train是训练椎间盘的，训练后模型记为zhuijianpan_resnet
import Res.zhuiti_train
#train是训练椎体的，训练后模型记为zhuiti_resnet
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image, pridict_rec_list = yolo.detect_image(image)
        r_image = np.array(r_image)
        print("脊椎定位成功！")
        cut_imagelist = []
        label_list = []
        for i in range(len(pridict_rec_list)):
            ymin = pridict_rec_list[i][0]
            ymax = pridict_rec_list[i][1]
            xmin = pridict_rec_list[i][2]
            xmax = pridict_rec_list[i][3]
            cut_imagelist.append(r_image[ymin:ymax,xmin:xmax])
        for i in range(len(pridict_rec_list)):
            if i in [0, 2, 4, 6, 8, 10]:
                cut_imagelist.append(Res.zhuijianpan_resnet(cut_imagelist[i]))
            if i in [1, 3, 5, 7, 9]:
                cut_imagelist.append(Res.zhuiti_resnet(cut_imagelist[i]))

        #定位图像和疾病分类可视化
        plt.figure()
        for i in range(11):
            plt.subplot(11,1,i+1)
            plt.imshow(cut_imagelist[i])
            plt.title(str(cut_imagelist[i]))
            plt.show()
        print("脊椎疾病分类成功！")


