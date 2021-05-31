import argparse
from resnet import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import warnings


import resnet
warnings.filterwarnings("ignore")

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 135   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 4      #批处理尺寸(batch_size)
LR = 0.1        #学习率



#数据处理
#数据处理函数，compose是联合的意思
# torchvision输出的是PILImage，值的范围是[0, 1]，我们将其转化为tensor数据，并归一化为[-1, 1]。
data_transform = transforms.Compose(
     [transforms.ToTensor(),#处理为Tensor
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#彩色图三通道和方差的归一化,归一化到[-1,1]
     ])

#定义数据类
class my_dataset(Dataset):
    def __init__(self, store_path, split, name, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.name = name
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        #glob.glob跟Windows里面的文件搜索差不多
        #获取指定目录下的所有图片
        for file in glob.glob(self.store_path + '/' + split + '/*.jpg'):
            cur_path = file.replace('\\', '/')
            cur_label = cur_path.split('_')[-1].split('.jpg')[0]

            if  cur_label=='v1' or cur_label=='v2':
                self.img_list.append(cur_path)
                self.label_list.append(self.name[cur_label])

    def __getitem__(self, item):
    #在自定义数据集是必须使用Image.open(x).convert('RGB')，否则打开的彩色图像为RGBA四通道
        img = Image.open(self.img_list[item]).convert('RGB')
        img = img.resize((32,32), Image.ANTIALIAS)
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label

    def __len__(self):
        return len(self.img_list)
# 使用自定义类
store_path = 'C:/Users/czf/Desktop/机器人/ResNet-18-cifar10/cut_picture/zhuiti/'#存储路径
split = 'train'
name = {'v1':0, 'v2':1}
#split是train，生成训练数据集
train_dataset = my_dataset(store_path, split, name, data_transform)
#dataloader是一个数据处理迭代器，输入一个dataset，它会基于某种抽样方式选出一个batch-size大小的内容，shuffle=false是顺序打乱的意思
# 将训练集划分成很多份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)#子进程=0，否则会报错
#生成测试训练集
split2 = 'test'
test_dataset = my_dataset(store_path, split2, name, data_transform)
testloader = DataLoader(test_dataset,batch_size=4, shuffle=False, num_workers=0)


# Cifar-10的标签
classes = ('v1', 'v2')

# 模型定义-ResNet
zhuiti_resnet = ResNet18(num_classes=2).to(device)


# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
if __name__ == "__main__":
    best_acc = 60  #2 初始化best test accuracy  85
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                zhuiti_resnet.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = zhuiti_resnet(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        zhuiti_resnet.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = zhuiti_resnet(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(zhuiti_resnet.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
