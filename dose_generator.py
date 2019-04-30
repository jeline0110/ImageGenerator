# coding=utf-8
# author: LianJie
# created: 2019.3.4
# version: 2
# description: Train a single network to generate ctdose image.
# environment: pytorch:1.0 python:3.5 cuda 9.0 cudnn 7.0
# 改变数据读取方式，分批放到内存上

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data 
import numpy as np
import os
import pdb
import cv2
import time
#from visualize import make_dot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument("--logs_dir", type=str, default="./training1/", help="path to logs directory")
parser.add_argument("--mode", type=str, default="train", help="mode:train or test")
parser.add_argument("--fintune", type=bool, default=False, help="If we train from the old model?")
parser.add_argument("--ckpt_path", type=str, default="./training/model-20.pth", help="path to model ckpt.")
parser.add_argument("--ct_dir_train", type=str, default="/home/scf/gan/ctdose_img/train/ct512/", help="path to CT for training")
parser.add_argument("--cts1_dir_train", type=str, default="/home/scf/gan/ctdose_img/train/structCT512_8bit_outfield/", help="path to CTS1 for training")
parser.add_argument("--cts2_dir_train", type=str, default="/home/scf/gan/ctdose_img/train/structCT_disONLY512/", help="path to CTS2 for training")
parser.add_argument("--dose_dir_train", type=str, default="/home/scf/gan/ctdose_img/train/dose512/", help="path to DOSE for training")
parser.add_argument("--ct_dir_test", type=str, default="/home/scf/gan/ctdose_img/val/ct512/", help="path to CT for test")
parser.add_argument("--cts1_dir_test", type=str, default="/home/scf/gan/ctdose_img/val/structCT512_8bit_outfield/", help="path to CTS1 for test")
parser.add_argument("--cts2_dir_test", type=str, default="/home/scf/gan/ctdose_img/val/structCT_disONLY512/", help="path to CTS2 for test")
parser.add_argument("--test_out_dir", type=str, default="/home/scf/gan/ctdose_img/val/test_16bit/", help="path to generator out in the test phase")
parser.add_argument("--val_out_dir", type=str, default="./val1/", help="path to generator out in the training process.")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for Adam Optimizer")
parser.add_argument("--sample_num", type=int, default=6950, help="How many samples to be used?")
FLAGS =  parser.parse_args()

MAX_EPOCH = 20 

#数据类，读取训练数据
class My_Data:
    def __init__(self):
        #建议输入图片以512的尺寸送入模型，效果较好
        self.ct_dir = FLAGS.ct_dir_train
        self.cts1_dir = FLAGS.cts1_dir_train
        self.cts2_dir = FLAGS.cts2_dir_train
        self.dose_dir =  FLAGS.dose_dir_train
        self.sample_num = FLAGS.sample_num
        self.ct_list, self.cts1_list, self.cts2_list, self.dose_list, = self.data_processor()
        self.idx = 0
        self.read_num = 1000 # 每次放到内存中的图片数量 若还是报内存错误 减小此值
        if not os.path.exists(FLAGS.logs_dir): os.makedirs(FLAGS.logs_dir)
    
    #返回训练数据路径
    def data_processor(self):
        ct_list = []
        cts1_list = []
        cts2_list = []
        dose_list = []
        img_paths = np.sort(os.listdir(self.ct_dir))
        for img in img_paths[:self.sample_num]:
            img_ct_path = os.path.join(self.ct_dir, img)
            img_cts1_path = os.path.join(self.cts1_dir, img)
            img_cts2_path = os.path.join(self.cts2_dir, img)
            img_dose_path = os.path.join(self.dose_dir, img)
            ct_list.append(img_ct_path)
            cts1_list.append(img_cts1_path)
            cts2_list.append(img_cts2_path)
            dose_list.append(img_dose_path)

        return ct_list, cts1_list, cts2_list, dose_list
    
    #加载数据
    def load_data(self, ct_path, cts1_path, cts2_path, dose_path):
        ct_img = cv2.imread(ct_path,2)  # 原深度 单通道 cv2以bgr顺序读取图片 0-4095
        cts1_img = cv2.imread(cts1_path,0) # 8位深度 单通道 0-255
        cts2_img = cv2.imread(cts2_path,0) # 0-255
        dose_img = cv2.imread(dose_path,2) # 0- 8000
        
        ctf_img = []
        ct_img = np.array(ct_img, dtype=np.float32)
        ct_img = (ct_img/2047.5)-1
        ctf_img.append(ct_img)

        cts1_img = np.array(cts1_img, dtype=np.float32)
        cts1_img = (cts1_img/127.5)-1
        ctf_img.append(cts1_img)

        cts2_img = np.array(cts2_img, dtype=np.float32)
        cts2_img = (cts2_img/127.5)-1 
        ctf_img.append(cts2_img)
        ctf_img = np.array(ctf_img).astype(np.float32)

        dose_img = np.array(dose_img, dtype=np.float32)
        dose_img = (dose_img/4000.0)-1
        dose_img = dose_img[...,np.newaxis]

        return ctf_img, dose_img

    #以tensor形式读取数据
    def data_to_tensor(self):
        images_ctf = []
        images_dose = []
        
        for i in range(self.read_num):
            if self.idx == self.sample_num:
                self.idx = 0
            ct_path = self.ct_list[self.idx]
            cts1_path = self.cts1_list[self.idx]
            cts2_path = self.cts2_list[self.idx]
            dose_path = self.dose_list[self.idx]
            img_ctf, img_dose = self.load_data(ct_path,cts1_path,cts2_path,dose_path)
            img_dose = img_dose.transpose(2,0,1) # cv2:h,w,c -> pytorch:c,h,w

            images_ctf.append(img_ctf)
            images_dose.append(img_dose)
            if self.idx%1000 == 0: print('Data is loading!')
            self.idx += 1

        images_ctf = np.array(images_ctf).astype(np.float32)
        images_dose = np.array(images_dose).astype(np.float32)

        images_ctf = torch.from_numpy(images_ctf) #转为tensor
        images_dose = torch.from_numpy(images_dose)
        print('Data loading completed!')

        return images_ctf, images_dose

class Bottleneck(nn.Module): # 继承nn.Moudle中的模块
    def __init__(self, ch1, ch2, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(ch1, ch2, kernel_size=1, bias=False) # resnet 中省略了偏置项
        self.bn1 = nn.BatchNorm2d(ch2)
        self.conv2 = nn.Conv2d(ch2, ch2, kernel_size=3, 
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.conv3 = nn.Conv2d(ch2, ch2 * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch2 * 4)
        self.relu = nn.ReLU(inplace=True) # inplace=True 对输出结果不产生影响,设为True可节省内存
        self.downsample = downsample
 
    def forward(self, x): # 一般不需显式调用forward函数,直接调用类名即可
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: # 一个是改变通道数，另一作用是降低维度 在每个block的第一次迭代使用
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
      
class Decode(nn.Module):
    def __init__(self, ch1, ch2, dropout=0.0):
        super(Decode, self).__init__()
        layers = [  nn.ConvTranspose2d(ch1, ch2, 
                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.BatchNorm2d(ch2),
                    nn.ReLU(inplace=True),

                    nn.ConvTranspose2d(ch2, ch2, 
                        kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(ch2),
                    nn.ReLU(inplace=True) ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.decode = nn.Sequential(*layers)

    def forward(self, x, skip_layer):
        out = self.decode(x)
        out = torch.cat((out, skip_layer), 1)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) # 是否添加待定 
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.Resblock(64,64,3,1)
        self.layer2 = self.Resblock(256,128,4,2)
        self.layer3 = self.Resblock(512,256,6,2)
        self.layer4 = self.Resblock(1024,512,3,2)
        self.decode1 = Decode(2048,1024,dropout=0.5) #因为cat的关系 输入通道*2
        self.decode2 = Decode(2048,512,dropout=0.5)
        self.decode3 = Decode(1024,256)
        self.decode4 = Decode(512,64)
        self.decode5 = nn.ConvTranspose2d(128, 1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d): 
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0) 

    def Resblock(self, ch1, ch2, block_num, stride=1):
        downsample = None
        if stride == 2 or ch1 == 64: # 第一层Resblock的Bottleneck不需要降采样，但需要改变通道数
            downsample =  nn.Sequential(
                nn.Conv2d(ch1, ch2 * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch2 * 4),)
        
        layers = []
        layers.append(Bottleneck(ch1,ch2,stride,downsample))
        for i in range(1, block_num):
            layers.append(Bottleneck(ch2*4,ch2))

        return nn.Sequential(*layers) # *参数 转为元组

    def forward(self, x):
        # x -> channel=1, size=224
        ly1 = self.conv1(x)
        ly1 = self.bn1(ly1)
        ly1 = self.relu(ly1)    #1->64 224->112

        ly2 = self.maxpool(ly1) #64->64 112->56
        ly2 = self.layer1(ly2)  #64->256 56->56
                                #final:64->256 112->56

        ly3 = self.layer2(ly2) #256->512 56->28
        ly4 = self.layer3(ly3) #512->1024 28->14
        ly5 = self.layer4(ly4) #1024->2048 14->7

        ly6 = self.decode1(ly5,ly4) #2048->1024+1024 7->14 #因为cat的原因，通道变为双倍
        ly7 = self.decode2(ly6,ly3) #2048->512+512 14->28
        ly8 = self.decode3(ly7,ly2) #1024->256+256 28->56
        ly9 = self.decode4(ly8,ly1) #512->64+64 56->112
        ly10 = self.decode5(ly9) #128->64 112->224
        out = self.tanh(ly10)

        return out

#评估类，用于可视化，检测的模型生成图片效果
class Val:
    def __init__(self):
        if not os.path.exists(FLAGS.val_out_dir): os.makedirs(FLAGS.val_out_dir)

    def load_data(self,img_path1,img_path2,img_path3):
        img = []

        img1 = cv2.imread(img_path1,2)
        img1 = np.array(img1, dtype=np.float32)
        img1 = (img1/2047.5)-1
        img.append(img1)

        img2 = cv2.imread(img_path2,0)
        img2 = np.array(img2, dtype=np.float32)
        img2 = (img2/127.5)-1
        img.append(img2)

        img3 = cv2.imread(img_path3,0)
        img3 = np.array(img3, dtype=np.float32)
        img3 = (img3/127.5)-1
        img.append(img3)
        img = np.array(img).astype(np.float32)

        img = img[np.newaxis,...]
        img = torch.from_numpy(img) # 转为tensor

        return img

    def save_data(self, img, name, mode='val'):
        img = img.detach().cpu()
        img = img.numpy().squeeze(0).transpose((1,2,0))
        #按照标签数据的形式保存
        img = (img+1)*4000.0
        img = np.array(img, dtype=np.uint16)
        #为方便显示，对生成的图片做一个转换，转为单通道8位图像
        #img = (img+1)*127.5
        #img = np.array(img, dtype=np.uint8)
        if mode == 'val':
            cv2.imwrite(FLAGS.val_out_dir + '%s.png'%name, img)
        elif mode == 'test':
            cv2.imwrite(FLAGS.test_out_dir + '{}.png'.format(name), img)

def train():
    val = Val()
    my_data = My_Data()
    
    generator = Generator()
    loss_func = torch.nn.L1Loss()
    generator = generator.cuda()
    loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(generator.parameters(), lr=FLAGS.learning_rate, betas=(0.5, 0.999))
    
    start_epoch = 0
    #选择是否进行fintune
    if FLAGS.fintune:
        generator.load_state_dict(torch.load(FLAGS.ckpt_path))
        print("Model restored...")
        start_epoch = int(FLAGS.ckpt_path.split('-')[1].split('.')[0])
        print("Continue train model at %d epoch" %(start_epoch))

    for epoch in range(start_epoch,MAX_EPOCH):
        count = int(FLAGS.sample_num/my_data.read_num) + 1
        for i in range(count): # 分批次读取数据
            train_ct, train_dose = my_data.data_to_tensor() # 数据转为tensor类型
            torch_dataset = Data.TensorDataset(train_ct, train_dose) # 将数据转为torch可以识别的数据类型
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=FLAGS.batch_size, shuffle=True) # 将数据放入loader中
    
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                gen_output = generator(batch_x)
                gen_loss = loss_func(gen_output,batch_y)*512*512
                optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                optimizer.step()
            
                if step % 10 == 0:
                    print("Epoch:%d/%d, Number:%d/%d, Batch:%d/%d, gen_loss:%g" % 
                        ((epoch+1),MAX_EPOCH,(i+1),count,(step+1),len(loader),gen_loss))
        
            #输出中间结果，查看模型效果
            ctf_img = val.load_data('ct.png','cts1.png','cts2.png')
            ctf_img = ctf_img.cuda()
            gen_img = generator(ctf_img)
            img_name = str(epoch+1)+'-'+str(step)
            val.save_data(gen_img, img_name, mode='val')
            
        print("Model saved")
        torch.save(generator.state_dict(), FLAGS.logs_dir+"model-%d.pth"%(epoch+1)) #只保存模型参数

#测试函数
def test():
    if not os.path.exists(FLAGS.test_out_dir): os.makedirs(FLAGS.test_out_dir)
    val = Val()
    generator = Generator()
    generator = generator.cuda()
    generator.load_state_dict(torch.load(FLAGS.ckpt_path))
    print("Model restored...")
    
    #测试一组图片
    i = 0
    for img in np.sort(os.listdir(FLAGS.ct_dir_test)):
        img_path1 = os.path.join(FLAGS.ct_dir_test, img)
        img_path2 = os.path.join(FLAGS.cts1_dir_test, img)
        img_path3 = os.path.join(FLAGS.cts2_dir_test, img)
        img_name = os.path.basename(img_path1).split('.png')[0]
        ctf_img = val.load_data(img_path1,img_path2,img_path3)
        ctf_img = ctf_img.cuda()
        gen_img = generator(ctf_img)
        val.save_data(gen_img, img_name, mode='test')
        i += 1
        print('the %d ct image has been transformed!'%i)

if __name__ == "__main__":
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()