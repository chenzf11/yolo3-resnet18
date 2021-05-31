##数据处理过程中，需要制作训练集和测试集。
import os, random, shutil

#fileDir = "./source/"  # 源图片文件夹路径
#tarDir = './result/'  # 移动到新的文件夹路径
def moveFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    print("Picture Splict Finished!")
    return

#"C:/Users/czf/Desktop/机器人/ResNet-18-cifar10/cut_picture/zhuijianpan/train"
















