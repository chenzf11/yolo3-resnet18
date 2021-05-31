import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.yolo3 import YoloBody
from nets.yolo_training import YOLOLoss, LossHistory, weights_init
from utils.dataloader import YoloDataset, yolo_dataset_collate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = net(images)
            losses      = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs     = net(images_val)
                losses      = []
                num_pos_all = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for i in range(3):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    loss_history.append_loss(total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' %(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    Cuda = False
    normalize = False
    input_shape = (256, 256)
    num_classes = 11
    anchors_path = 'model_data/yolo_anchors.txt'  # 先验框路径
    anchors = get_anchors(anchors_path)
    model = YoloBody(anchors, num_classes)  # 实例化
    weights_init(model)

    model_path      = "logs/Epoch2-Total_Loss322.5670-Val_Loss127.5157.pth"  # 导入初始权值，目的是更快的训练
    print('Loading weights into state dict...')

    device          = torch.device('cpu')  # if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), Cuda, normalize)
    loss_history = LossHistory("logs/")

    annotation_path = '_train.txt'  # 加载数据
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-3
        Batch_size      = 10
        Init_Epoch      = 0
        Freeze_Epoch    = 35
        
        optimizer       = optim.Adam(net.parameters(),lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()
            
    if True:
        lr              = 1e-4
        Batch_size      = 10
        Freeze_Epoch    = 30
        Unfreeze_Epoch  = 30

        optimizer       = optim.Adam(net.parameters(),lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
        
        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_size      = num_train//Batch_size
        epoch_size_val  = num_val//Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
