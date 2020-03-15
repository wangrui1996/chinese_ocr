#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from scipy import ndimage
from PIL import Image, ImageFont, ImageDraw

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.layers.core import Reshape, Masking, Lambda, Permute
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
from src.models import densenet
from src.image_generator import generate_ocr
#import densenet
import cairo
import numpy
alphabet = u'0123456789 '

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

np.random.seed(55)

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck
# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
    bg_color = (255, 255, 255)
    fg_color = (0, 0, 0)
    font_size = random.randint(h // 2, h)
    font_ch = ImageFont.truetype('simsun.ttf', font_size, 0)

    img = Image.new("RGB", (w, h), bg_color)
    max_shift_x = w - len(text)*font_size//2
    max_shift_y = h - font_size
    top_left_x = random.randint(0, max_shift_x)
    ImageDraw.Draw(img).text((random.randint(0, max_shift_x), random.randint(0, max_shift_y)), text, fg_color, font=font_ch)
    a = numpy.array(img)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)
    import cv2

    return a

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret
from tensorflow.python import keras
import random

# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h, downsample_factor,
                 absolute_max_string_len=8):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(alphabet) + 1

    def get_random_number(self, str_len=10):
        select_range = ['0', '1', '2', '3', '4', '5', '6','7', '8', '9', ' ', "||", '|']
        number = ""
        for i in range(str_len):
            number += random.choice(select_range)

        return number
    def filter_number(self, number):
        re = ""
        for i in number:
            if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']:
                re += i
        if len(re) == 0:
            re = " "
        return re
    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        X_data = np.ones([size, self.img_h, self.img_w, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                X_data[i, :, :,0] = self.paint_func('')[0, :, :]
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 4
                label_length[i] = 1
                source_str.append('')
            else:
                X_test = self.get_random_number(random.choice(range(1, self.absolute_max_string_len)))
                X_data[i, :, :, 0] = (
                        self.paint_func(X_test)[0, :, :])
                label = self.filter_number(X_test)
                assert len(label)!=0
                labels[i, 0:len(label)] = text_to_labels(label)
                input_length[i] = self.img_w // self.downsample_factor - 4
                label_length[i] = len(label)
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.minibatch_size, train=True)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.minibatch_size, train=False)
            yield ret

    def on_train_begin(self, logs={}):
        self.paint_func = lambda text: paint_text(
            text, self.img_w, self.img_h,
            rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 3 <= epoch < 6:
            self.paint_func = lambda text: paint_text(
                text, self.img_w, self.img_h,
                rotate=False, ud=True, multi_fonts=False)
        elif 6 <= epoch < 9:
            self.paint_func = lambda text: paint_text(
                text, self.img_w, self.img_h,
                rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(
                text, self.img_w, self.img_h,
                rotate=True, ud=True, multi_fonts=True)

class VizCallback(keras.callbacks.Callback):

    def __init__(self, base_model, model_save_path):
        self.base_model = base_model
        self.output_dir = model_save_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(
            os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        out_path = os.path.join(self.output_dir,"ocrs.tflite")
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_model)



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:-2, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    # clipnorm seems to speeds up convergence
    #sgd = SGD(learning_rate=0.02,
    #          decay=1e-6,
    #          momentum=0.9,
    #          nesterov=True)
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model




if __name__ == '__main__':
    img_w = 256
    img_h = 32
    minibatch_size = 32
    model_save_path = "./save_models"
    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=8)

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, img_gen.get_output_size())

    modelPath = './models/pretrain_model/keras.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    viz_cb = VizCallback(basemodel, model_save_path=model_save_path)
    print('-----------Start training-----------')
    model.fit_generator(
        generator=img_gen.next_train(),
        steps_per_epoch=5,
        epochs=1000,
        validation_data=img_gen.next_val(),
        validation_steps=2,
        callbacks=[viz_cb, img_gen])
    #model.fit_generator(img_gen.next_train(),
   # 	steps_per_epoch = 10000,
    #	epochs = 100,
    #	initial_epoch = 0,
    #	validation_data = test_loader,
    	#validation_steps = 36440 // batch_size,
    #	callbacks = [checkpoint, earlystop, changelr, tensorboard])

