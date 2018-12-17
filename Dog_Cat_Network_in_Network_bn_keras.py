#coding:gbk
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
import data_input
import cv2
import os
from sklearn.model_selection import train_test_split
import time
from keras import backend as K

"""
     ʹ��nin����ѵ��è�����ݼ�
     ���Ԥ��ͼƬ�����
"""

# �Ľ���������batch normalization  ����ݶ�����
# ��conv �� activation ֮�����һ��bn��

start = time.clock()



batch_size    = 128
epochs        = 200
iterations    = 391
num_classes   = 2
dropout       = 0.5
weight_decay  = 0.0001
log_filepath  = './nin_bn'  # ��־·��
train_dir = './train/'  # ���ݼ�·�����Լ�����https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data��


"""
    ����GPU��Դʹ����
    ��allow_growth option����һ��ʼ����������GPU������
    Ȼ�������������ӣ����ڲ����ͷ��ڴ棬���Իᵼ����Ƭ
"""
if('tensorflow' == K.backend()):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    # for i in range(3):
    #     x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
    #     x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - np.mean(x_train[:,:,:,i])) / np.std(x_train[:,:,:,i])
        x_test[:,:,:,i] = (x_test[:,:,:,i] - np.mean(x_test[:,:,:,i]))/ np.std(x_test[:,:,:,i])
    return x_train, x_test

def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:    
        return 0.002
    return 0.0004

def build_model():
  model = Sequential()

  model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal", input_shape=x_train.shape[1:]))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
  
  model.add(Dropout(dropout))
  
  model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
  
  model.add(Dropout(dropout))
  
  model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(2, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(GlobalAveragePooling2D())
  model.add(Activation('softmax'))

  sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  return model

if __name__ == '__main__':

    # load data

    (x_train, y_train) =data_input.get_files(train_dir)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    #(x_test, y_test) = (x_train, y_train) # �������ݼ��õ���ѵ�����ݼ�
    print("y1",y_train.shape, y_train[0])
    """
    keras.utils.to_categorical(y_train, num_classes):one-hot����
    (3, 1)
 [[1]
 [0]
 [1]]
(3, 1, 3)
 [[[0. 1. 0.]]

 [[1. 0. 0.]]

 [[0. 1. 0.]]]
    """
    y_train = keras.utils.to_categorical(y_train, num_classes)
    print("y2",y_train.shape, y_train[0])
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train, x_test = color_preprocessing(x_train, x_test)

    # build network
    model = build_model()
    print(model.summary())

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)  # ��̬����ѧϰ��
    cbks = [change_lr,tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')

    """ͼ�����㣺ƽ�ơ���ת..."""
    datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
    datagen.fit(x_train)

    # start training
    y_train.shape = (len(y_train),2)
    y_test.shape = (len(y_test), 2)
    print("shape:",y_test.shape)
    """
    1��fit_generator ��fit��������ʡ�ڴ�
    2��flow():���ɾ��������������׼�����batch����,����һ������ѭ���в��ϵķ���batch����
    """
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),steps_per_epoch=iterations,epochs=epochs,callbacks=cbks,validation_data=(x_test, y_test))
    model.save('nin.h5')  # ����ģ��
    end = time.clock()
    print("ѵ��ʱ�䣺",end-start)

    # ��ͼ����з���
    print("-----------------��ʼԤ���Լ���ͼ��-------------------")
    my_test = []
    for file in os.listdir("./cats"):
        im = cv2.imread("./cats/"+file)
        im = cv2.resize(im,(32,32))
        my_test.append(im)
    for file in os.listdir("./dogs"):
        im = cv2.imread("./dogs/"+file)
        im = cv2.resize(im,(32,32))
        my_test.append(im)

    t = np.array(my_test)
    t = t.astype('float32')

    t[:, :, :, 0] = (t[:, :, :, 0] - np.mean(t[:, :, :, 0])) / np.std(t[:, :, :, 0])
    t[:, :, :, 1] = (t[:, :, :, 1] - np.mean(t[:, :, :, 1])) / np.std(t[:, :, :, 1])
    t[:, :, :, 2] = (t[:, :, :, 2] - np.mean(t[:, :, :, 2])) / np.std(t[:, :, :, 2])

    """
        ����ʹ�ñ���ѵ���õ�ģ��ֱ�ӽ���Ԥ��
    """
    from keras.models import load_model
    my_model = load_model('nin.h5')  # ����ģ��
    preds = my_model.predict(t)
    print('Predicted:', preds)

