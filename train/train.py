#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.layers.core import Reshape, Masking, Lambda, Permute
from tensorflow.python.keras.layers.recurrent import GRU, LSTM
from tensorflow.python.keras.layers.wrappers import Bidirectional, TimeDistributed
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
import densenet


img_h = 32
img_w = 280
batch_size = 128
maxlabellength = 20

def get_session(gpu_fraction=1.0):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)

    # set ocrs generator

    import random    
    input_height = 32
    input_width = 280
    bg_color = (255, 255, 255)
    fg_color = (0, 0, 0)
    font_ch = ImageFont.truetype('../fonts/simsun.ttf', input_height, 0)
    char_list = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
    char_list = [ch.strip('\n') for ch in char_list]

    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            if random.randint(0, 10) < 8:
                img1 = Image.open(os.path.join(image_path, j)).convert('L')
                img = np.array(img1, 'f') / 255.0 - 0.5

                x[i] = np.expand_dims(img, axis=2)
                # print('imag:shape', img.shape)
                str1 = image_label[j]
                label_length[i] = len(str1)

                if(len(str1) <= 0):
                    print("len < 0", j)
                input_length[i] = imagesize[1] // 8
                labels[i, :len(str1)] = [int(k) - 1 for k in str1]
            else:
                img = Image.new("RGB", (input_width, input_height), bg_color)
                height_offset_range = input_height // 8
                width_offset_range = input_height // 4
                char = ""
                for _ in range(random.randint(13, 16)):
                    char = char + str(random.randint(0, 9))

                if random.randint(0, 8) > 6:
                    char = str(char[:(len(char) - 2)]) + 'X' + str(char[(len(char) - 2):])

                offset_x = random.randint(-width_offset_range, width_offset_range)
                offset_y = random.randint(-height_offset_range, height_offset_range)
                ImageDraw.Draw(img).text((offset_x, offset_y), char, fg_color, font=font_ch)

                def rotate_image(img, rotate_range=3):
                    img = img.convert('RGBA')
                    ratate = random.randint(-rotate_range, rotate_range)
                    rot = img.rotate(ratate)
                    bg_ = Image.new('RGBA', rot.size, (255,) * 4)
                    # bg_ = Image.new("RGBA", img.size, bg_color)
                    img = Image.composite(rot, bg_, rot)
                    img = img.convert("RGB")
                    return img
                img = rotate_image(img).convert('L')

                img = np.array(img, 'f') / 255.0 - 0.5
                x[i] = np.expand_dims(img, axis=2)

                label_length[i] = len(char)
                input_length[i] = imagesize[1] // 8
                labels[i, :len(char)] = [char_list.index(k) - 1 for k in char]

        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    char_set = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    nclass = len(char_set)

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = './models/pretrain_model/keras.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen('data_train.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen('data_test.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4**epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
    	steps_per_epoch = 3607567 // batch_size,
    	epochs = 10,
    	initial_epoch = 0,
    	validation_data = test_loader,
    	validation_steps = 36440 // batch_size,
    	callbacks = [checkpoint, earlystop, changelr, tensorboard])

