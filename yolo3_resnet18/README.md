## README



#### 运行条件

执行下列操作前，

请一定要下载权值文件(Epoch16-Total_Loss32.4839-Val_Loss33.5745.pth)，将其放在logs目录下

​					(net_050_zhuiti.pth)，(net_050.pth)，将其放在Res目录下

权值文件可通过百度网盘下载

链接地址：[https://pan.baidu.com/s/1j0787chX6TKkQJ9xeUHCEg](https://pan.baidu.com/s/1j0787chX6TKkQJ9xeUHCEg)

提取码：1e40



如果想要更简便地运行yolo3-resnet18网络，可以下载 ‘替换.zip’，压缩包中包括yolo3-resnet18网络所需所有的文件夹，解压后将其中所有文件夹全部复制到项目下即可

链接地址：[https://pan.baidu.com/s/1qD1n-5RawTnHE82JrphD9Q](https://pan.baidu.com/s/1qD1n-5RawTnHE82JrphD9Q)

提取码：kv6j



若有需要，可通过百度网盘下载脊柱疾病智能诊断图像集

链接地址：[https://pan.baidu.com/s/1q3NiGIw_NHGmO0qYsq-rfw](https://pan.baidu.com/s/1q3NiGIw_NHGmO0qYsq-rfw)

提取码：1lv2



#### Yolo的训练

​	1.作业所提供的文件均已经准备好，如果训练数据没有新增，请直接运行train.py

​	2.如果需要新加训练数据，请将图片(studyX.jpg)与标签(studyX.txt)一起放在'脊柱疾病智能诊	断/train/data'下，注意名字需要一致

​	3.严格按照下列顺序，运行代码：

​		getxml.py -> VOCdevkit/VOC/voc2yolo3.py -> voc_annotations.py -> train.py

​	4.其他：	

​		训练过程中，代码的损失下降情况详见logs目录下的文件夹，此处已经提供两个文件夹， 里面分别是两次	训练的损失下降情况

​		训练前需要一个起始权值文件，下载后请放在logs目录下

​		训练过程中，每经过一个个epoch，会保存一次权值，保存地址在logs目录，每个pth文件的文件名会显示对应的loss



#### Resnet的训练

1、运行pic_break.py

2、运行Load.py

3、运行cut_pic.py，将原图切割成11张小图分成椎间盘和椎体两部分，每部分又按照
	   pic_break随机分为train和test，修改6、25、28、37、38、42、43行路径

4、运行resnet.py，构建网络

5、运行train.py，训练椎间盘的resnet模型，修改73行椎间盘分割图片存储路径

6、运行zhuiti_train.py，训练椎体的resnet模型，修改72行椎体分割图片存储路径



查看Resnet的实际表现效果，请跟随“网络测试”部分教程。测试正确率是打印在命令行窗口的Accuracy。



#### 网络测试

如果想查看我们的网络对 测试集 的效果，请直接运行  test_trainset.py

如果想查看我们的网络对 训练集 的效果，请直接运行  test_test.py

如果想查看我们的网络对 新数据 的效果，请将新的数据(如newPicX.jpg, newPicX.txt,名字必须相同，且标签的格式必须与作业所给数据的格式相同)一起放到 img/extra 目录下，然后直接运行 test_final.py

最终的损失、耗时、准确度会在命令行打印，可视化结果请查看 img/pred 目录



#### 其他

如果想要替换resnet的权值文件，需要将新权值文件放置在Res文件夹下。

特别注意：更改Resnet的权值文件后，需要在test_trainset/test_test/test_final中做相应修改：test_trainset 第113行修改为新的预测锥体的权值， 第114行改为新的预测椎间盘的权值，其余两个文件类似，test_test在110与111行，test_train在113与114行

#### 改进方向
1、Unet网络中的注意力模型结构和参数的修改和优化。在进行总结和问题分析时，小组觉得问题可能出在我们将Unet的输出进行了3次MaxPooling后进行4次卷积，得到注意力模型输出这个过程中。在对注意力模型学习研究后，对注意力模型的结构和参数进行修改和优化预计可以提升定位效果。

#### 参考资料
1、U-Net: Convolutional Networks for Biomedical Image Segmentation. Olaf Ronneberger, Philipp Fischer, and Thomas Brox
2、Unet神经网络为什么会在医学图像分割表现好？ - 知乎 (zhihu.com)
3、https://github.com/bubbliiiing/yolo3-pytorch
