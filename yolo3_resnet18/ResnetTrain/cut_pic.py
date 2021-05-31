from skimage import io
import Load
import cv2
import pic_break

path = 'C:/Users/czf/Desktop/python/yolo3-pytorch-master/yolo3-pytorch-master/脊柱疾病智能诊断/train/data'
x, ytag = Load.DataLoad(path)

for itm in range(len(x)):

    imgname = x[itm].split('\\')[-1].split('.jpg')[0]
    img = io.imread(x[itm])
    gt = Load.getFrame(ytag[itm], bias=3)
    i = 0
    for img_each_label in gt:
        spt = img_each_label
        objnms = spt[5].split(',')
        for objname in objnms:
            xmin = int(spt[0])
            xmax = int(spt[2])
            ymin = int(spt[1])
            ymax = int(spt[3])
            cut_img = img[ymin:ymax,xmin:xmax]
            if i in [0,2,4,6,8,10]:
                filepath = 'cut_picture/zhuiti/train/'
                cv2.imwrite(filepath+'%4d%4d_%s.jpg' %(itm,i,objname),cut_img)
            if i in [1,3,5,7,9]:
                filepath = 'cut_picture/zhuiti/train/'
                cv2.imwrite(filepath + '%4d%4d_%s.jpg' % (itm, i, objname), cut_img)
        i = i+1
        print(i)


print('Cut Picture Finished!')

#将图片划分为训练集和测试集
fileDir = "C:/Users/czf/Desktop/机器人/ResNet-18-cifar10/cut_picture/zhuiti/train/"
tarDir = "C:/Users/czf/Desktop/机器人/ResNet-18-cifar10/cut_picture/zhuiti/test/"
pic_break.moveFile(fileDir,tarDir)