# 这个代码用来预测老师所给数据之外的数据

from skimage import io
import Load
from PIL import Image, ImageDraw, ImageFont
from yolo import YOLO
import time
from Res import resnet
import numpy as np
import xml.etree.ElementTree as ET
from skimage import io,color
import os
from torchvision import transforms
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def predict_disease(x, gt):
    imgname = x.split('/')[-1].split('.jpg')[0]
    xml_file = open((r'img/Annotations' + '/' + imgname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('\t<folder>' + 'VOC' + '</folder>\n')
    xml_file.write('\t<filename>' + imgname + '.jpg' + '</filename>\n')

    xml_file.write('\t<source>\n')
    xml_file.write('\t\t<database>' + 'The VOC Database' + '</database>\n')
    xml_file.write('\t\t<annotation>' + 'PASCAL VOC' + '</annotation>\n')
    xml_file.write('\t\t<image>flickr</image>\n')
    xml_file.write('\t\t<flickrid>325991873</flickrid>\n')
    xml_file.write('\t</source>\n')

    xml_file.write('\t<owner>\n')
    xml_file.write('\t\t<flickrid>archin</flickrid>\n')
    xml_file.write('\t\t<name>?</name>\n')
    xml_file.write('\t</owner>\n')

    img = io.imread(x)
    width = img.shape[1]
    height = img.shape[0]
    xml_file.write('\t<size>\n')
    xml_file.write('\t\t<width>' + str(width) + '</width>\n')
    xml_file.write('\t\t<height>' + str(height) + '</height>\n')
    xml_file.write('\t\t<depth>3</depth>\n')
    xml_file.write('\t</size>\n')
    xml_file.write('\t<segmented>0</segmented>\n')

    for img_each_label in gt:
        spt = img_each_label
        objnms = spt[4]
        xml_file.write('\t<object>\n')
        xml_file.write('\t\t<name>' + objnms + '</name>\n')
        xml_file.write('\t\t<pose>Unspecified</pose>\n')
        xml_file.write('\t\t<truncated>0</truncated>\n')
        xml_file.write('\t\t<difficult>0</difficult>\n')
        xml_file.write('\t\t<bndbox>\n')
        xml_file.write('\t\t\t<ymin>' + str(spt[0]) + '</ymin>\n')
        xml_file.write('\t\t\t<ymax>' + str(spt[1]) + '</ymax>\n')
        xml_file.write('\t\t\t<xmin>' + str(spt[2]) + '</xmin>\n')
        xml_file.write('\t\t\t<xmax>' + str(spt[3]) + '</xmax>\n')
        xml_file.write('\t\t</bndbox>\n')
        xml_file.write('\t</object>\n')
    xml_file.write('</annotation>')


def getMSE(picname, pre_rec):
        in_file = open('./img/AnnotationsLabel/%s.xml' % (picname), encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        labmap = {}
        MSEsum = 0
        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            cls = cls.split(',')[0]
            if cls not in classes or int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            xmin = int(float(xmlbox.find('xmin').text))
            ymin = int(float(xmlbox.find('ymin').text))
            xmax = int(float(xmlbox.find('xmax').text))
            ymax = int(float(xmlbox.find('ymax').text))
            center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            labmap.update({cls: center})
        for each in pre_rec:
            precenter = [(each[2] + each[3]) / 2, (each[0] + each[1]) / 2]
            a = np.array([labmap[each[4]] - np.array(precenter)])
            b = np.transpose(a)
            MSEsum += np.matmul(a, b) / 2
        return MSEsum / len(pre_rec)


path = './img/extra'
x, ytag = Load.DataLoad(path)

if len(ytag) != 0:
    for itm in range(len(x)):
        imgname = x[itm].split('/')[-1].split('.jpg')[0]
        xml_file = open((r'img/AnnotationsLabel' + '/' + imgname + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>' + 'VOC' + '</folder>\n')
        xml_file.write('\t<filename>' + imgname + '.jpg' + '</filename>\n')

        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>' + 'The VOC Database' + '</database>\n')
        xml_file.write('\t\t<annotation>' + 'PASCAL VOC' + '</annotation>\n')
        xml_file.write('\t\t<image>flickr</image>\n')
        xml_file.write('\t\t<flickrid>325991873</flickrid>\n')
        xml_file.write('\t</source>\n')

        xml_file.write('\t<owner>\n')
        xml_file.write('\t\t<flickrid>archin</flickrid>\n')
        xml_file.write('\t\t<name>?</name>\n')
        xml_file.write('\t</owner>\n')

        img = io.imread(x[itm])
        width = img.shape[1]
        height = img.shape[0]
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(width) + '</width>\n')
        xml_file.write('\t\t<height>' + str(height) + '</height>\n')
        xml_file.write('\t\t<depth>3</depth>\n')
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>0</segmented>\n')

        gt = Load.getFrame(ytag[itm], bias=3)
        for img_each_label in gt:
            spt = img_each_label
            objnms = spt[5].split(',')
            for objname in objnms:
                xml_file.write('\t<object>\n')
                xml_file.write('\t\t<name>' + spt[4] + ',' + objname + '</name>\n')
                xml_file.write('\t\t<pose>Unspecified</pose>\n')
                xml_file.write('\t\t<truncated>0</truncated>\n')
                xml_file.write('\t\t<difficult>0</difficult>\n')
                xml_file.write('\t\t<bndbox>\n')
                xml_file.write('\t\t\t<xmin>' + str(int(spt[0])) + '</xmin>\n')
                xml_file.write('\t\t\t<ymin>' + str(int(spt[1])) + '</ymin>\n')
                xml_file.write('\t\t\t<xmax>' + str(int(spt[2])) + '</xmax>\n')
                xml_file.write('\t\t\t<ymax>' + str(int(spt[3])) + '</ymax>\n')
                xml_file.write('\t\t</bndbox>\n')
                xml_file.write('\t</object>\n')
        xml_file.write('</annotation>')
        xml_file.close()

    savepath = './img/pred/'
    classes = ['L1', 'L2', 'L3', 'L4', 'L5', 'T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']
    classes2 = ('v1', 'v2', 'v3', 'v4', 'v5')
    test_imgs = io.ImageCollection('./img/extra/*.jpg').files
    data_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # 处理为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 彩色图三通道和方差的归一化,归一化到[-1,1]
        ])
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=10)

    yolo = YOLO()
    count = 0
    preVerti = torch.load('Res/net_050_zhuiti.pth', map_location='cpu')  # 预测锥体的权值
    preDisc = torch.load('Res/net_050.pth', map_location='cpu')  # 预测椎间盘的权值

    predictZHUITI = resnet.ResNet18(num_classes=2)
    predictZHUITI.load_state_dict(preVerti)
    predictZHUIJIANPAN = resnet.ResNet18(num_classes=5)
    predictZHUIJIANPAN.load_state_dict(preDisc)

    all_mse = 0
    all_accu = 0

    aver_time = []
    for img in test_imgs:
        img = img.replace('\\', '/')
        count += 1
        start_time = time.time()
        picname = img.split('/')[-1].split('.')[0]
        image = Image.open(img)
        tmp_real_lab = {}

        r_image, pridict_rec_list = yolo.detect_image(image)  # 预测

        MSE = getMSE(picname, pridict_rec_list)  # 定位误差

        accuracy = 0  # 分类精度

        predict_disease(img, pridict_rec_list)  # 保存预测结果

        ##############
        in_file = open('img/Annotations/%s.xml' % (picname), encoding='utf-8')
        reallab = open('img/AnnotationsLabel/%s.xml' % (picname), encoding='utf-8')
        tree = ET.parse(reallab)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls = cls.split(',')
            tmp_real_lab.update({cls[0]: cls[1]})
            tmp_real_lab[cls[0]]
        reallab.close()

        tree = ET.parse(in_file)
        root = tree.getroot()
        I = io.imread(img)
        I = color.grey2rgb(I)
        for obj in root.iter('object'):
            difficult = 0
            cls = obj.find('name').text.split(',')[0]
            if len(cls) > 3:  # 椎间盘
                xmlbox = obj.find('bndbox')
                xmin = int(float(xmlbox.find('xmin').text))
                ymin = int(float(xmlbox.find('ymin').text))
                xmax = int(float(xmlbox.find('xmax').text))
                ymax = int(float(xmlbox.find('ymax').text))
                filepath = './img/' + picname + '/zhuijianpan/'
                input_ = data_transform(Image.fromarray(I[ymin:ymax, xmin:xmax]))
                input_ = torch.unsqueeze(input_, dim=0)
                disc = torch.squeeze(predictZHUIJIANPAN(input_), dim=0).detach().numpy()  # 你们的预测 椎间盘 的函数,disc是记录疾病名称
                index = np.argmax(disc)
                disc = classes2[index]
                if disc == tmp_real_lab[cls]:
                    accuracy += 1
                obj.find('name').text = cls + ',' + disc
                tree.write('img/Annotations/%s.xml' % (picname))
            else:
                xmlbox = obj.find('bndbox')
                xmin = int(float(xmlbox.find('xmin').text))
                ymin = int(float(xmlbox.find('ymin').text))
                xmax = int(float(xmlbox.find('xmax').text))
                ymax = int(float(xmlbox.find('ymax').text))
                filepath = './img/' + picname + '/zhuiti/'
                input_ = data_transform(Image.fromarray(I[ymin:ymax, xmin:xmax]))
                input_ = torch.unsqueeze(input_, dim=0)
                vertebra = torch.squeeze(predictZHUITI(input_), dim=0).detach().numpy()  # 你们的预测 锥体 的函数,vertebra是记录疾病名称
                index = np.argmax(vertebra)
                vertebra = classes2[index]
                if vertebra == tmp_real_lab[cls]:
                    accuracy += 1
                obj.find('name').text = cls + ',' + vertebra
                tree.write('img/Annotations/%s.xml' % (picname))

        in_file.close()
        accuracy = accuracy * 100 / len(pridict_rec_list)
        all_mse += MSE
        all_accu += accuracy
        print('检测：' + picname + '，MSE = ', MSE, '，诊断准确性Accuracy = %.2f' % (accuracy), "%")

        end_time = time.time()
        aver_time.extend([end_time - start_time])
        ##############
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        with open('img/Annotations/%s.xml' % (picname), encoding='utf-8') as in_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            image = Image.fromarray(I)
            for obj in root.iter('object'):
                cls = obj.find('name').text
                xmlbox = obj.find('bndbox')
                xmin = int(float(xmlbox.find('xmin').text))
                ymin = int(float(xmlbox.find('ymin').text))
                xmax = int(float(xmlbox.find('xmax').text))
                ymax = int(float(xmlbox.find('ymax').text))
                draw = ImageDraw.Draw(image)
                draw.rectangle([xmin, ymin, xmax, ymax], width=1)
                draw.text((xmax, ymin), cls, fill=(0, 255, 0), font=font)
            image.save(savepath + picname + '.png', 'PNG')

    print("共检测图片", count, '张，平均每张耗时:%.4f' % (sum(aver_time) / count), 's,', '测试集总定位均方误差：', all_mse / count,
          '测试集总诊断精度：%.2f' % (all_accu / count), '%')

else:
    print("No pics in /img/extra!!")





