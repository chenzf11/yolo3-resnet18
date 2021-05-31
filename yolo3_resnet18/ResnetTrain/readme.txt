想要查看网络效果，只需要跟随README.md运行test函数即可，该文件夹中是Resnet训练代码
！！！！注意Resnet训练部分上一个独立的项目！！！！

Resnet的训练：请严格按照下列顺序进行
	1、运行pic_break.py
	2、运行Load.py
	3、运行cut_pic.py，将原图切割成11张小图分成椎间盘和椎体两部分，每部分又按照
	   pic_break随机分为train和test，修改6、25、28、37、38、42、43行路径
	4、运行resnet，构建网络
	5、运行train.py，训练椎间盘的resnet模型，修改73行椎间盘分割图片存储路径
	6、运行zhuiti_train.py，训练椎体的resnet模型，修改72行椎体分割图片存储路径

查看Resnet的实际表现效果，请跟随“网络测试”部分教程。测试正确率是打印在命令行窗口的Accuracy。
