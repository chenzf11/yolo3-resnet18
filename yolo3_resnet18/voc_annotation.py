import xml.etree.ElementTree as ET
from os import getcwd

sets=['train', 'val', 'test']
#-----------------------------------------------------#
#   这里设定的classes顺序要和model_data里的txt一样
#-----------------------------------------------------#
classes = ['L1', 'L2', 'L3', 'L4', 'L5', 'T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

def convert_annotation(image_id, list_file):
    in_file = open('VOCdevkit/VOC/Annotations/%s.xml'%(image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        cls = cls.split(',')[0]
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    image_ids = open('VOCdevkit/VOC/ImageSets/Main/%s.txt'%(image_set), encoding='utf-8').read().strip().split()
    list_file = open('_%s.txt'%(image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC/JPEGImages/%s.jpg' % (wd, image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
