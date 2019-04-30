# coding=utf-8
# author: LianJie
# created: 2018.12.19
# version: final(7)
# description: Train a GAN network(pix2pix) to transfer image(MRT-->CT)
# refer_paper: Image-to-Image Translation with Conditional Adversarial Networks
# environment: tensorflow:1.8 python:3.5 

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import os
from random import shuffle
import pickle as cPickle
import pdb
import collections
from PIL import Image
import cv2
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "./gpu_test/", "path to logs directory")
tf.flags.DEFINE_string("mode", "train", "mode:train or test")
tf.flags.DEFINE_bool("fintune", "False", "If we train from the old model?")
tf.flags.DEFINE_string("ckpt_path", "./training/model.ckpt-39000", "path to model ckpt.")
tf.flags.DEFINE_string("mri_dir_train", "/home/scf/gan/img/train/MRI/", "path to MRI images for training")
tf.flags.DEFINE_string("ct_dir_train", "/home/scf/gan/img/train/CT/", "path to CT images for training")
tf.flags.DEFINE_string("mri_dir_test", "/home/scf/gan/img/test/MRI/", "path to MRI images for test")
tf.flags.DEFINE_string("ct_dir_test", "/home/scf/gan/img/test/ct_label/", "path to CT images for test")
tf.flags.DEFINE_string("test_out_dir", "./test/", "path to generator out in the test phase")
tf.flags.DEFINE_string("val_out_dir", "./val/", "path to generator out in the training process.")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("sample_num", "1364", "How many samples to be used?")
tf.flags.DEFINE_bool("debug", "False", "debug Flag.")
tf.flags.DEFINE_string("debug_dir", "./debug/", "path to save debug image.")

Model = collections.namedtuple("Model", "dis_loss, gen_loss_GAN, gen_loss_L1, output, train_op")

#最大训练步数，L1 loss 不再下降，且有上升趋势即可停止训练
MAX_ITERATION = 5000

#数据打包
def mypickle(filename, data):
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

#读取数据 
def myunpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'rb')
    dict = cPickle.load(fo)    
    fo.close()

    return dict

#数据类，读取训练数据，以batch形式送入模型训练
class Data:
    def __init__(self):
        #建议输入图片以512的尺寸送入模型，效果较好
        #self.img_size = 256
        self.mri_dir = FLAGS.mri_dir_train
        self.ct_dir =  FLAGS.ct_dir_train
        self.batch_size = FLAGS.batch_size
        self.sample_num = FLAGS.sample_num
        self.mri_list, self.ct_list = self.data_processor()
        self.idx = 0
        self.data_num = len(self.mri_list)
        self.rnd_list = np.arange(self.data_num) 
        shuffle(self.rnd_list)
        if not os.path.exists(FLAGS.logs_dir): os.makedirs(FLAGS.logs_dir)
        if not os.path.exists(FLAGS.debug_dir): os.makedirs(FLAGS.debug_dir)
    
    #以batch形式读取数据
    def next_batch(self):
        images_mri = []
        images_ct = []

        for i in range(self.batch_size):
            if self.idx == self.data_num:
                self.idx = 0          
                shuffle(self.rnd_list)
            cur_idx = self.rnd_list[self.idx]
            mri_path = self.mri_list[cur_idx]
            ct_path = self.ct_list[cur_idx]
            img_mri,img_ct = self.load_data(mri_path,ct_path)
            images_mri.append(img_mri)
            images_ct.append(img_ct)

            self.idx += 1

        images_mri = np.array(images_mri).astype(np.float32)
        images_ct = np.array(images_ct).astype(np.float32)

        return images_mri, images_ct
    
    #保存训练数据的路径，方便数据读取
    def data_processor(self):
        data_dic = './mri_ct_train_dic_%d' % self.sample_num
        if not os.path.exists(data_dic):
            mri_list = []
            ct_list = []
            img_paths = np.sort(os.listdir(self.mri_dir))
            for img in img_paths[:self.sample_num]:
                img_mri_path = os.path.join(self.mri_dir, img)
                img_ct_path = os.path.join(self.ct_dir, img)
                mri_list.append(img_mri_path)
                ct_list.append(img_ct_path)
            dic = {'mri_list':mri_list,'ct_list':ct_list}
            mypickle(data_dic, dic)
        else:
            dic = myunpickle(data_dic)
            mri_list = dic['mri_list']
            ct_list = dic['ct_list']

        return mri_list, ct_list
    
    #加载数据
    def load_data(self, mri_path, ct_path):
        #mri_img = Image.open(mri_path)
        #mri_img = mri_img.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        #ct_img = Image.open(ct_path)
        #ct_img = ct_img.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        
        mri_img = cv2.imread(mri_path,0)  # 8位深度 单通道
        #mri_img = cv2.resize(mri_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        ct_img = cv2.imread(ct_path,2) # 原深度 单通道 
        #ct_img = cv2.resize(ct_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        #debug 模式，方便训练数据可视化，可选择关闭
        if FLAGS.debug:
            mri_name = os.path.dirname(mri_path).split('/')[-1]
            ct_name = os.path.dirname(ct_path).split('/')[-1]
            i = len(os.listdir(FLAGS.debug_dir)) + 1
            #mri_img.save('./debug/%s_%d.png'%(mri_name,i))
            #ct_img.save('./debug/%s_%d.png'%(ct_name,i))
            cv2.imwrite('./debug/%s_%d.png'%(mri_name,i), mri_img)
            cv2.imwrite('./debug/%s_%d.png'%(ct_name,i), ct_img)
            #生成10张图片后暂停
            if i >= 10: 
                FLAGS.debug = False
                pdb.set_trace()
        
        #mri 数据便签范围0-255 8位单通道
        mri_img = np.array(mri_img, dtype=np.float32)
        #[0-255] -> [-1,1]
        mri_img = (mri_img/127.5)-1
        mri_img = mri_img[...,np.newaxis]
        
        #ct 数据便签范围0-4095 16位单通道
        ct_img = np.array(ct_img, dtype=np.float32)
        #[0-4095] -> [-1,1]
        ct_img = (ct_img/2047.5)-1
        ct_img = ct_img[...,np.newaxis]

        return mri_img, ct_img

#网络类，用以构建网络
class Layer:
    def __init__(self):
        self.gen_input = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="generator_input")
        self.target = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="target_image")
        self.is_training = tf.placeholder(tf.bool, shape=[], name='batch_norm_flag')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_ratio')
    
    #当batch_size为1时，为instance nrom
    def batch_norm(self,input):
        return tf.layers.batch_normalization(input, axis=3, epsilon=1e-5, momentum=0.1, training=self.is_training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    
    def lrelu(self,x,a):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
    
    #生成器卷积层
    def gen_conv(self,input,channel,relu=True,norm=True,scope=''):
        with slim.arg_scope([slim.conv2d],
                            kernel_size = [4,4],
                            stride = 2,
                            padding = 'SAME',
                            activation_fn = None,
                            weights_initializer = tf.random_normal_initializer(0, 0.02),
                            ):
            #正常顺序 conv2d->batch_norm->relu 可结合generator函数进行相应修改
            if relu:
                conv = self.lrelu(input, 0.2)
                conv = slim.conv2d(conv, channel, scope=scope)
            elif not relu: 
                conv = slim.conv2d(input, channel, scope=scope)
            if norm:
                conv = self.batch_norm(conv)

            return conv
    
    #生成器反卷积层
    def gen_deconv(self,input,channel,relu=True,norm=True,scope=''):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size = [4,4],
                            stride = 2,
                            padding = 'SAME',
                            activation_fn = None,
                            weights_initializer = tf.random_normal_initializer(0, 0.02),
                            ):
            if relu:
                deconv = tf.nn.relu(input)
                deconv = slim.conv2d_transpose(deconv, channel, scope=scope)
            elif not relu:
                deconv = slim.conv2d_transpose(input, channel, scope=scope)
            if norm:
                deconv = self.batch_norm(deconv)

            return deconv
    
    #判别器卷积层
    def dis_conv(self,input,channel,stride,relu=True,norm=True,scope=''):
        with slim.arg_scope([slim.conv2d],
                            kernel_size = [4,4],
                            padding = 'VALID',
                            activation_fn = None,
                            weights_initializer = tf.random_normal_initializer(0, 0.02),
                            ):
            conv = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            conv = slim.conv2d(conv, channel, stride=stride, scope=scope)
            if norm:
                conv = self.batch_norm(conv)
            if relu:
                conv = self.lrelu(conv,0.2)

            return conv
    
    #生成器 给定训练数据分辨率为512 建议就以512输入模型，生成效果较256要好很多
    def generator(self,gen_input):
        #256x256(分辨率变化注释可忽略，此为针对256的输入)
        conv0 = self.gen_conv(gen_input,64,relu=False,norm=False,scope='conv0')
        #128x128
        conv1 = self.gen_conv(conv0,128,scope='conv1')
        #64x64
        conv2 = self.gen_conv(conv1,256,scope='conv2')
        #32x32
        conv3 = self.gen_conv(conv2,512,scope='conv3')
        #16x16
        conv4 = self.gen_conv(conv3,512,scope='conv4')
        #8x8
        conv5 = self.gen_conv(conv4,512,scope='conv5')
        #4x4
        conv6 = self.gen_conv(conv5,512,scope='conv6')
        #2x2
        conv7 = self.gen_conv(conv6,512,scope='conv7')
        #1x1
         
        #1x1
        deconv8 = self.gen_deconv(conv7,512,scope='deconv8')
        deconv8 = slim.dropout(deconv8, self.keep_prob, scope='dropout8')
        #2x2
        deconv9 = tf.concat([deconv8, conv6],axis=3)
        deconv9 = self.gen_deconv(deconv9,512,scope='deconv9')
        deconv9 = slim.dropout(deconv9, self.keep_prob, scope='dropout9')
        #4x4
        deconv10 = tf.concat([deconv9, conv5],axis=3)
        deconv10 = self.gen_deconv(deconv10,512,scope='deconv10')
        deconv10 = slim.dropout(deconv10, self.keep_prob, scope='dropout10') 
        #8x8
        deconv11 = tf.concat([deconv10, conv4],axis=3)
        deconv11 = self.gen_deconv(deconv11,512,scope='deconv11')
        #16x16
        deconv12 = tf.concat([deconv11, conv3],axis=3)
        deconv12 = self.gen_deconv(deconv12,256,scope='deconv12')
        #32x32
        deconv13 = tf.concat([deconv12, conv2],axis=3)
        deconv13 = self.gen_deconv(deconv13,128,scope='deconv13')
        #64x64
        deconv14 = tf.concat([deconv13, conv1],axis=3)
        deconv14 = self.gen_deconv(deconv14,64,scope='deconv14')
        #128x128
        deconv15 = tf.concat([deconv14, conv0],axis=3)
        deconv15 = self.gen_deconv(deconv15,1,norm=False,scope='deconv15')
        gen_output = tf.tanh(deconv15)
        #256x256

        return gen_output
    
    #判别器
    def discriminator(self,dis_input):
        #256x256(分辨率变化注释可忽略，此为针对256的输入)
        conv16 = self.dis_conv(dis_input,64,2,norm=False,scope='conv16')
        #128x128
        conv17 = self.dis_conv(conv16,128,2,scope='conv17')
        #64x64
        conv18 = self.dis_conv(conv17,256,2,scope='conv18')
        #32x32
        conv19 = self.dis_conv(conv18,512,1,scope='conv19')
        #31x31
        conv20 = self.dis_conv(conv19,1,1,relu=False,norm=False,scope='conv20')
        dis_output = tf.sigmoid(conv20)
        #30x30

        return dis_output

    #构建pix2pix模型
    def pix2pix_model(self):
        with tf.variable_scope("generator"):
           gen_output = self.generator(self.gen_input)
        
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                dis_input1 = tf.concat([self.gen_input, self.target], axis=3)
                predict_real = self.discriminator(dis_input1) 

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True): 
                dis_input2 = tf.concat([self.gen_input, gen_output], axis=3)
                predict_fake = self.discriminator(dis_input2)
        
        #判别器loss
        # predict_real => 1
        # predict_fake => 0
        dis_loss = tf.reduce_mean(-(tf.log(predict_real + 1e-12) + tf.log(1 - predict_fake + 1e-12)))
        
        #生成器loss
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + 1e-12))  
        gen_loss_L1 = tf.reduce_mean(tf.abs(self.target - gen_output))
        gen_loss = gen_loss_GAN * 1.0 + gen_loss_L1 * 100.0 # gen_loss_L1 降为0.005左右即可停止训练
        
        #判别器训练参数
        dis_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        dis_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5).minimize(dis_loss,var_list=dis_vars)
        
        #生成器训练参数
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        #训练生成器时，判别器参数不变 执行完dis_op 后执行gen_op，迭代训练
        with tf.control_dependencies([dis_op]):  
            gen_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5).minimize(gen_loss,var_list=gen_vars)
        
        '''
          在采用随机梯度下降算法训练神经网络时，使用 tf.train.ExponentialMovingAverage 滑动平均操作的意义在于
        提高模型在测试数据上的健壮性（robustness）。tensorflow 下的 tf.train.ExponentialMovingAverage 需要
        提供一个衰减率（decay）。该衰减率用于控制模型更新的速度。
        '''
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([dis_loss, gen_loss_GAN, gen_loss_L1])

        return Model(
            dis_loss=ema.average(dis_loss),             
            gen_loss_GAN=ema.average(gen_loss_GAN),          
            gen_loss_L1=ema.average(gen_loss_L1),           
            output=gen_output,                        
            train_op=tf.group(update_losses,gen_op))

#评估类，用于可视化，检测的模型生成图片效果
class Val:
    def __init__(self):
        if not os.path.exists(FLAGS.test_out_dir): os.makedirs(FLAGS.test_out_dir)
        if not os.path.exists(FLAGS.val_out_dir): os.makedirs(FLAGS.val_out_dir)

    def load_data(self, img_path):
        #img = Image.open(img_path)
        #img = img.resize((256, 256), Image.ANTIALIAS)
        img = cv2.imread(img_path,0)
        img = np.array(img, dtype=np.float32)
        img = (img/127.5)-1
        img = img[np.newaxis,...,np.newaxis]

        return img

    def save_data(self, img, name, mode='val'):
        img = np.squeeze(img)
        #按照标签数据的形式保存
        #img = (img+1)*2047.5
        #img = np.array(img, dtype=np.uint16)
        #为方便显示，对生成的图片做一个转换，转为单通道8位图像
        img = (img+1)*127.5
        img = np.array(img, dtype=np.uint8)
        #img = Image.fromarray(img)
        #img = img.resize((512, 512), Image.ANTIALIAS)
        if mode == 'val':
            #img.save(FLAGS.val_out_dir + '%d.png'%name)
            cv2.imwrite(FLAGS.val_out_dir + '%d.png'%name, img)
        elif mode == 'test':
            #img.save(FLAGS.test_out_dir + '{}.png'.format(name))
            cv2.imwrite(FLAGS.test_out_dir + '{}.png'.format(name), img)

#训练函数
def train():
    data = Data()
    val = Val()
    layer = Layer()
    model = layer.pix2pix_model()

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=3)
    sess.run(tf.global_variables_initializer())
    start_step = 0
    #选择是否进行fintune
    if FLAGS.fintune:
        saver.restore(sess, FLAGS.ckpt_path)
        print("Model restored...")
        start_step = int(FLAGS.ckpt_path.split('-')[1]) + 1
        print("Continue train model at %d step" %(start_step-1))
    
    s_time = time.time()
    for itr in range(start_step, MAX_ITERATION+1):
        #读取训练数据
        mri_images, ct_images = data.next_batch()
        #is_training 需要设为True 实际测试时，keep_prob设为1.0 0.5 影响不大
        feed_dict = {layer.gen_input:mri_images, layer.target:ct_images,
                layer.keep_prob:0.5, layer.is_training:True}

        sess.run(model.train_op, feed_dict=feed_dict)
        dis_loss, gen_loss_GAN, gen_loss_L1 = sess.run([model.dis_loss, 
            model.gen_loss_GAN, model.gen_loss_L1], feed_dict=feed_dict)

        if itr % 10 == 0:             
            print("Step: %d, dis_loss:%g, gen_loss_GAN: %g, gen_loss_L1: %g" 
                % (itr, dis_loss, gen_loss_GAN, gen_loss_L1))
        
        #保存模型，输出中间结果，查看模型效果
        if itr % 1000 == 0:
            mri_img = val.load_data('mri.png')
            feed_dict = {layer.gen_input:mri_img, layer.keep_prob:0.5, layer.is_training:True} # is_training 需要设为True
            gen_img = sess.run(model.output, feed_dict=feed_dict)
            #val.save_data(gen_img, itr, mode='val')

            print("Model saved")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
        
        #停止训练，保存模型
        if itr == MAX_ITERATION:
            e_time = time.time()
            f_time = e_time - s_time
            print("Model training finished")
            print(f_time)
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            break

#测试函数
def test():
    val = Val()
    layer = Layer()
    model = layer.pix2pix_model()

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
     
    #加载模型
    saver.restore(sess, FLAGS.ckpt_path)
    print("Model restored...")
    
    #测试一组图片
    i = 0
    for img in np.sort(os.listdir(FLAGS.mri_dir_test)):
        img_path = os.path.join(FLAGS.mri_dir_test, img)
        img_name = os.path.basename(img_path).split('.png')[0]
        mri_img = val.load_data(img_path)
        feed_dict = {layer.gen_input:mri_img, layer.keep_prob:0.5, layer.is_training:True}
        gen_img = sess.run(model.output, feed_dict=feed_dict)
        val.save_data(gen_img, img_name, mode='test')
        i += 1
        print('the %d mri image has been transformed!'%i)
     
    '''
    #测试单张图片
    mri_img = val.load_data('test1.png')
    feed_dict = {layer.gen_input:mri_img, layer.keep_prob:0.5, layer.is_training:True}
    gen_img = sess.run(model.output, feed_dict=feed_dict)
    val.save_data(gen_img, 'result', mode='test')
    '''

# 将结果写入HTML网页 在终端运行firefox index.html 即可查看结果 
def append_index():
    files_list = []
    for file in  np.sort(os.listdir(FLAGS.mri_dir_test)):
        file_name = os.path.basename(file)
        file_list = {'name': file_name}
        input_path = os.path.join(FLAGS.mri_dir_test,file_name)
        output_path = os.path.join(FLAGS.test_out_dir,file_name)
        target_path = os.path.join(FLAGS.ct_dir_test,file_name)
        file_list['input'] = input_path
        file_list['output'] = output_path
        file_list['target'] = target_path
        files_list.append(file_list)
        #pdb.set_trace()

    index_path = './index_new.html'
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
 
    for file_list in files_list:
        index.write("<tr>")
        index.write("<td>%s</td>" % file_list["name"])
 
        for kind in ["input", "output", "target"]:
            index.write("<td><img src='%s' width=256px height=256px></td>" % file_list[kind])
 
        index.write("</tr>")

if __name__ == "__main__":
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()
        #append_index()