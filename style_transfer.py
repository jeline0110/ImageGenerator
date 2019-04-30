# coding=utf-8
# author: LianJie
# created: 2018.12.27
# version: 2
# description: Train a style transfer network to transfer image(MRT-->CT)

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import os
from random import shuffle
import pickle as cPickle
import pdb
from PIL import Image
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "./training_v2/", "path to logs directory")
tf.flags.DEFINE_string("mode", "train", "mode:train or test")
tf.flags.DEFINE_bool("fintune", "False", "If we train from the old model?")
tf.flags.DEFINE_string("ckpt_path", "/home/scf/gan/style_transfer/training/model.ckpt-10500", "path to model ckpt.")
tf.flags.DEFINE_string("vgg_model_path", "vgg_16.ckpt", "path to vgg16 model ckpt.")
tf.flags.DEFINE_string("content_train_dir", "style_transfer_gray/content/", "path to MRI images for training")
tf.flags.DEFINE_string("style_train_dir", "style_transfer_gray/style/", "path to CT images for training")
tf.flags.DEFINE_string("data_dic_name", "style_transfer_gray", "name of data dic")
tf.flags.DEFINE_string("mri_test_dir", "style_transfer/content/", "path to MRI images for test")
tf.flags.DEFINE_string("test_out_dir", "./test/", "path to generator out in the test phase")
tf.flags.DEFINE_string("val_out_dir", "./val_v2/", "path to generator out in the training process.")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("sample_num", "1", "How many samples to be used?")
tf.flags.DEFINE_bool("debug", "True", "debug Flag.")
tf.flags.DEFINE_string("debug_dir", "./debug/", "path to save debug image.")

MAX_ITERATION = 20000

def mypickle(filename, data):
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def myunpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'rb')
    dict = cPickle.load(fo)    
    fo.close()

    return dict

class Data:
    def __init__(self):
        self.img_size = 256 
        self.mri_dir = FLAGS.content_train_dir
        self.ct_dir =  FLAGS.style_train_dir
        self.batch_size = FLAGS.batch_size
        self.sample_num = FLAGS.sample_num
        self.mri_list, self.ct_list = self.data_processor()
        self.idx = 0
        self.data_num = len(self.mri_list)
        self.rnd_list = np.arange(self.data_num) 
        shuffle(self.rnd_list)
        if not os.path.exists(FLAGS.logs_dir): os.makedirs(FLAGS.logs_dir)
        if not os.path.exists(FLAGS.debug_dir): os.makedirs(FLAGS.debug_dir)

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
            img_mri = self.load_data(mri_path)
            img_ct = self.load_data(ct_path)
            images_mri.append(img_mri)
            images_ct.append(img_ct)

            self.idx += 1

        images_mri = np.array(images_mri).astype(np.float32)
        images_ct = np.array(images_ct).astype(np.float32)

        return images_mri, images_ct
        
    def data_processor(self):
        data_dic = FLAGS.data_dic_name + '_dic_%d' % self.sample_num
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
    
    def load_data(self, img_path):
        img = cv2.imread(img_path) # cv2 读取直接读取三通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        if FLAGS.debug:
            dir_name = os.path.dirname(img_path).split('/')[-1]
            i = len(os.listdir(FLAGS.debug_dir)) + 1
            cv2.imwrite('./debug/%s_%d.png'%(dir_name,i), img)
            if i == 12: # 生成10张图片后暂停
                pdb.set_trace()
                FLAGS.debug = False

        img = np.array(img, dtype=np.float32)
        img = img - 127.5

        return img

class Loss:
    def gram(self, layer):
        shape = tf.shape(layer)
        num_images = shape[0]
        width = shape[1]
        height = shape[2]
        num_filters = shape[3]
        filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
        grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

        return grams

    def style_loss(self, f1, f2, f3, f4): # 输入顺序：生成器输出，内容，风格
        gen_f, _, style_f = tf.split(f1, 3, 0)
        size = tf.size(gen_f)
        style_loss = tf.nn.l2_loss(self.gram(gen_f) - self.gram(style_f))*2 / tf.to_float(size)

        gen_f, _, style_f = tf.split(f2, 3, 0)
        size = tf.size(gen_f)
        style_loss += tf.nn.l2_loss(self.gram(gen_f) - self.gram(style_f)) * 2 / tf.to_float(size)

        gen_f, _, style_f = tf.split(f3, 3, 0)
        size = tf.size(gen_f)
        style_loss += tf.nn.l2_loss(self.gram(gen_f) - self.gram(style_f)) * 2 / tf.to_float(size)

        gen_f, _, style_f = tf.split(f4, 3, 0)
        size = tf.size(gen_f)
        style_loss += tf.nn.l2_loss(self.gram(gen_f) - self.gram(style_f)) * 2 / tf.to_float(size)

        return style_loss

    def content_loss(self, f3):
        gen_f, content_f, _ = tf.split(f3, 3, 0)
        size = tf.size(gen_f)
        content_loss = tf.nn.l2_loss(gen_f - content_f) *2 / tf.to_float(size)

        return content_loss

class Layer:
    def instance_norm(self, x):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    def relu(self, x):

        return tf.nn.relu(x)
    
    def img_scale(self, x, scale):
        weight = x.get_shape()[1].value # 返回元组 不需要sess tf.shape 需要sess 返回tensor
        height = x.get_shape()[2].value

        try:
            out = tf.image.resize_nearest_neighbor(x, size=(weight*scale, height*scale))
        except:
            out = tf.image.resize_images(x, size=[weight*scale, height*scale])

        return out

    def res_moudle(self, x, channel, name):
        with tf.variable_scope(name_or_scope=name):
            res1 = slim.conv2d(x, channel, [3,3], stride=1, scope='conv1')
            res1 = self.relu(res1)
            res2 = slim.conv2d(res1, channel, [3,3], stride=1, scope='conv2')
            res2 = self.relu(res2)

            return x+res2

    def generator(self, gen_input, name):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=None):
            with tf.variable_scope(name_or_scope=name) as gen_vs:
                gen_input = tf.pad(gen_input, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT') #排除边缘效应
                net = slim.conv2d(gen_input, 32, [9,9], stride=1, scope='conv1')
                net1 = self.relu(self.instance_norm(net))

                net = slim.conv2d(net1, 64, [3,3], stride=2, scope='conv2')
                net2 = self.instance_norm(net)

                net = slim.conv2d(net2, 128, [3,3], stride=2, scope='conv3')
                net3 = self.instance_norm(net)

                net4 = self.res_moudle(net3, 128, name='residual1')
                net5 = self.res_moudle(net4, 128, name='residual2')
                net6 = self.res_moudle(net5, 128, name='residual3')
                net7 = self.res_moudle(net6, 128, name='residual4')

                net = self.img_scale(net7,2)
                net = slim.conv2d(net, 64, [3,3], stride=1, scope='conv4')
                net8 = self.relu(self.instance_norm(net))

                net = self.img_scale(net8,2)
                net = slim.conv2d(net, 32, [3,3], stride=1, scope='conv5')
                net9 = self.relu(self.instance_norm(net))

                net = slim.conv2d(net9,3, [9,9], stride=1, scope='conv6')
                net10 = tf.nn.tanh(self.instance_norm(net))
                
                gen_vars = tf.contrib.framework.get_variables(gen_vs)
                net10 = net10*127.5

                height = net10.get_shape()[1].value
                width = net10.get_shape()[2].value

                gen_output = tf.image.crop_to_bounding_box(net10, 10, 10, height-20, width-20)

                return gen_output, gen_vars

    def my_vgg16(self, inputs):
        with tf.variable_scope('vgg_16'):
             net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
             f1 = net

             net = slim.max_pool2d(net, [2, 2], scope='pool1')
             net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
             f2 = net

             net = slim.max_pool2d(net, [2, 2], scope='pool2')
             net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
             f3 = net

             net = slim.max_pool2d(net, [2, 2], scope='pool3')
             net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
             f4 = net
             
             exclude = ['vgg_16/conv5','vgg_16/fc6','vgg_16/pool5','vgg_16/pool4',
                     'vgg_16/fc7','vgg_16/global_pool','vgg_16/fc8/squeezed','vgg_16/fc8']
             
             return f1,f2,f3,f4,exclude

class Val:
    def __init__(self):
        self.training_size = 256
        self.original_size = 512
        if not os.path.exists(FLAGS.test_out_dir): os.makedirs(FLAGS.test_out_dir)
        if not os.path.exists(FLAGS.val_out_dir): os.makedirs(FLAGS.val_out_dir)

    def load_data(self, img_path):
        img = cv2.imread(img_path) # cv2 读取直接读取三通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.training_size, self.training_size))
        img = np.array(img, dtype=np.float32)

        img = img - 127.5
        img = img[np.newaxis,...]

        return img

    def save_data(self, img, name, mode='val'):
        img = np.squeeze(img) + 127.5
        img = cv2.resize(img, (self.original_size, self.original_size))
        if mode == 'val':
        	cv2.imwrite(FLAGS.val_out_dir + '%d.png'%name, img)
        elif mode == 'test':
        	cv2.imwrite(FLAGS.test_out_dir + '{}.png'.format(name), img)

class Run():
    def __init__(self):
        self.sess = tf.Session()
        print("Setting up Saver...")
        self.style_weight = 100.0
        self.content_image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="generator_input")
        self.style_image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="style_image")
        self.loss = Loss()
        self.data = Data()
        self.layer = Layer()
        self.val = Val()

    def build_model(self):
        self.gen_output, gen_vars = self.layer.generator(self.content_image,name='transform')
 
        inputs = tf.concat([self.gen_output,self.content_image,self.style_image], axis=0) # 输入顺序：生成器输出，内容，风格
        f1,f2,f3,f4,exclude = self.layer.my_vgg16(inputs)
        
        self.style_loss = self.loss.style_loss(f1,f2,f3,f4)
        self.content_loss = self.loss.content_loss(f3)
        total_loss =  self.content_loss + self.style_weight*self.style_loss

        vgg_vars = slim.get_variables_to_restore(include=['vgg_16'],exclude=exclude)
        init_vgg_vars = slim.assign_from_checkpoint_fn(FLAGS.vgg_model_path, vgg_vars)
        init_vgg_vars(self.sess)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(total_loss,var_list=gen_vars) #只训练generator
        
        all_vars = tf.global_variables()
        init_vars =  [var for var in all_vars if 'vgg_16' not in var.name]
        self.sess.run(tf.variables_initializer(var_list=init_vars))
        self.saver = tf.train.Saver(max_to_keep=1, var_list=gen_vars)

    def train(self):
        start_step = 0
        if FLAGS.fintune:
            self.saver.restore(self.sess, FLAGS.ckpt_path)
            print("Model restored...")
            start_step = int(FLAGS.ckpt_path.split('-')[1]) + 1
            print("Continue train model at %d step" %(start_step-1))
        
        stop_num = 0
        for itr in range(start_step, MAX_ITERATION+1):
            mri_images, ct_images = self.data.next_batch() 
            feed_dict = {self.content_image:mri_images, self.style_image:ct_images}

            self.sess.run(self.train_op, feed_dict=feed_dict)
            c_loss, s_loss = self.sess.run([self.content_loss, self.style_loss], 
                feed_dict=feed_dict)

            if itr % 1 == 0:              
                print("Step: %d, content_loss:%g, style_loss: %g" % (itr, c_loss, s_loss*100))
            '''
            if s_loss <= 0.006:
                stop_num += 1
            '''
            if itr % 100 == 0:
                mri_img = self.val.load_data('mri.png')
                feed_dict = {self.content_image:mri_img}
                gen_img = self.sess.run(self.gen_output, feed_dict=feed_dict)
                self.val.save_data(gen_img, itr, mode='val')

                print("Model saved")
                self.saver.save(self.sess, FLAGS.logs_dir + "model.ckpt", itr)
               
            if stop_num >= 30 or itr==MAX_ITERATION: 
                print("Model training finished")
                self.saver.save(self.sess, FLAGS.logs_dir + "model.ckpt", itr)
                break

    def test(self):
        gen_output, _ = self.layer.generator(self.content_image,name='transform')
        gen_vars = slim.get_variables_to_restore(include=['transform'])
        init_gen_vars = slim.assign_from_checkpoint_fn(FLAGS.ckpt_path, gen_vars)
        init_gen_vars(self.sess)

        i = 0
        for img in np.sort(os.listdir(FLAGS.mri_test_dir)):
            img_path = os.path.join(FLAGS.mri_test_dir, img)
            img_name = os.path.basename(img_path).split('.png')[0]
            mri_img = self.val.load_data(img_path)
            feed_dict = {self.content_image:mri_img}
            gen_img = self.sess.run(gen_output, feed_dict=feed_dict)
            self.val.save_data(gen_img, img_name, mode='test')
            i += 1
            print('the %d mri image has been transformed!'%i)

if __name__ == "__main__":
    run = Run()
    if FLAGS.mode == 'train':
        run.build_model()
        run.train()
    elif FLAGS.mode == 'test':
        run.test()