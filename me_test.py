#CASME-II 0.86275  0.88163-04281643
###########################################________TEST
#SMIC_E_Train_VAL DATA
import sys
sys.path.append("/content/drive/My Drive/drive1/")
#!chdir('/content/drive/"My Drive"/drive1')

import itertools
import json
from keras.models import model_from_json
import os
import cv2
import tensorflow as tf
import numpy as np
#print(np.__path__)
import imageio
import keras
import keras.utils
#from medels import *
#from keras_lookahead import Lookahead
#from keras_radam import RAdam
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop, rmsprop, adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras import backend as K
import matplotlib.pyplot as plt

import keras.backend as K
from keras.callbacks import LearningRateScheduler
 

from sklearn.utils import class_weight

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #jinyong gpu
import sys

#from SMIC.models import c3d, densenet_3d, inception_3d, c3d_new, inception_3d_new, densenet_3d_new, resnet_3d_new, /
    #drn_new, densenet_3d_c2t20, resnet_3d_c2t20, densenet_3d_LEARn_all_t20
#from SMIC.models.c3d_new import lrcn, lstm, mlp, conv_3d
#from models import se_3densenet_34
#from models import DenseNet3D
from my_models import se_densenet_3d
##K.set_image_dim_ordering('th') #gai th tf
#K.image_dim_ordering()=='tf'
#K.image_data_format() == '0'

image_rows, image_columns,image_depth = 64, 64, 20  #64, 64, 96
#K.image_data_format() == 'channels_first'

#K.image_dim_ordering()=='tf'

training_list = []
#angrypath = '../../workspace/micro-expression/data/angry/'
#happypath = '../../workspace/micro-expression/data/happy/'
#disgustpath = '../../workspace/micro-expression/data/disgust/'

#angrypath = 'D:/LEAR_all_video/casme_sq_TIM20_pic/anger/'
#happypath = 'D:/LEAR_all_video/casme_sq_TIM20_pic/happy/'
#disgustpath='D:/LEAR_all_video/casme_sq_TIM20_pic/disgust/'


negativepath = "/content/drive/My Drive/SMIC_e/SMIC_e/negative/"
positivepath = "/content/drive/My Drive/SMIC_e/SMIC_e/positive/"
surprisepath = "/content/drive/My Drive/SMIC_e/SMIC_e/surprise/"

#otherspath = "D:/LEAR_all_video/casme2_TIM_20_4cls_pic/others/"

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/421_SMIC_E_TRAIN_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/421_SMIC_E_TRAIN_labels_64_3cls.npy')#数据有问题

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/CASME_II_E_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/CASME_II_E_labels_64_3cls.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/drive2/numpy_training_datasets/SMIC_train_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/drive2/numpy_training_datasets/SMIC_train_labels_64_3cls.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/microexpstcnn_images_64_3cls_cas(me)2.npy')
#traininglabels =np.load('/content/drive/My Drive/microexpstcnn_labels_64_3cls_cas(me)2.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/casme2_e_train_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/casme2_e_train_labels_64_3cls.npy')


# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/EVM20_numpy_training_datasets/424_CASME2_E_EVM20_TV_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/EVM20_numpy_training_datasets/424_CASME2_E_EVM20_TV_labels_64_3cls.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/EVM20_numpy_training_datasets/424_CASME_II_E_EVM20_TV_train_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/EVM20_numpy_training_datasets/424_CASME_II_E_EVM20_TV_train_labels_64_3cls.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/CASME_II_E_train_val_test_data/CASME_II_E_train_val_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/CASME_II_E_train_val_test_data/CASME_II_E_train_val_labels_64_3cls.npy')


# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/EVM20_numpy_training_datasets/424_SMIC_E_EVM20_TV_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/EVM20_numpy_training_datasets/424_SMIC_E_EVM20_TV_labels_64_3cls.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/SMIC_E_TV_numpy/425_SMIC_E_TV_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/SMIC_E_TV_numpy/425_SMIC_E_TV_labels_64_3cls.npy')

# Load training images and labels that are stored in numpy array
#training_set = np.load('/content/drive/My Drive/SMIC_E_TV_numpy/425_CASME2_E_TV_train_images_64_3cls.npy')
#traininglabels =np.load('/content/drive/My Drive/SMIC_E_TV_numpy/425_CASME2_E_TV_train_labels_64_3cls.npy')
#/content/drive/My Drive/421_SMIC_E_TRAIN_images_64_3cls.npy

#print(training_set.shape)




"""
# MicroExpSTCNN Model
model = Sequential()        #(3, 3, 15)
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))

#model.add(Convolution3D(32, (3, 3, 3), activation='relu'))
#model.add(MaxPooling3D(pool_size=(3, 3, 1)))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
"""

num_classes=3
#model = c3d.c3d_model(num_classes, input_shape=(1, image_rows, image_columns, image_depth))
#model = c3d_new.c3d(c3d)
###model = se_3densenet_34.createModel(num_classes, input_shape=(1, image_rows, image_columns, image_depth))
#model = resnet_3d_c2t20.resnet_3d(num_classes, input_shape=(1, image_rows, image_columns, image_depth))
#model = drn_new.c3d_model()
#model = c3d_new.conv_3d(conv_3d)
#model = DenseNet3D.DenseNet3D_FCN(input_shape=(1, 64, 64, 20),nb_dense_block=4, growth_rate=16,/
 #                      nb_layers_per_block=3, upsampling_type='upsampling', classes=3, activation='softmax')

model = se_densenet_3d.SEDenseNet(input_shape=(image_rows, image_columns,image_depth,1),
                        depth=7,
                        nb_dense_block=3,
                        growth_rate=12,
                        nb_filter=-1,
                        nb_layers_per_block=-1,
                        bottleneck=True,
                        reduction=0.0,
                        dropout_rate=0.1,
                        weight_decay=1e-2,
                        subsample_initial_block=True,
                        maxpool_initial_block=True,
                        include_top=True,
                        weights=None,
                        input_tensor=None,
                        classes=num_classes,
                        activation='softmax')
#test:maxpool-true

# 2e-2  1e-3

#cas(me)2:weight_decay=1e-2 lr=0.0005

#model.summary()
#model = densenet_3d_LEARn_all_t20.densenet_3d(num_classes,input_shape=(1, image_rows, image_columns, image_depth))
#lr=0.0002
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 15 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)
reduce_r = ReduceLROnPlateau(monitor='val_loss',verbose=1,factor=0.8,patience=3, mode='auto',min_lr=0)
#ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
optimizer = 'adam' #Lookahead(RAdam(lr))#RAdam adam
#model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
model.compile(optimizer=adam(lr=0.001),loss = 'categorical_crossentropy', metrics = ['acc'])
#keras.backend.get_session().run(tf.global_variables_initializer()) #gai


model.summary()




# Load pre-trained weights


model.load_weights('D:/lh/2021/smic_single/hdf5/-89-0.939024_SMIC_333.hdf5')
#/content/drive/My Drive/my_hdf5/-90-0.975610_SMIC_E_improvement.hdf5
#model.load_weights('./checkpoints/weights.88-0.893939.hdf5')
###加载训练模型

class Metrics(tf.contrib.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return

filepath="D:/lh/2021/hdf5/-{epoch:02d}-{val_acc:.6f}_SMIC_E_improvement.hdf5"


# Spliting the dataset into training and validation sets
#train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, 
#                                    test_size=0.111, random_state=4,stratify=traininglabels)

# Save validation set in a numpy array

#np.save('/content/drive/My Drive/drive1/numpy_validation_datasets/SMIC_E_val_images_64_3cls.npy', validation_images)
#np.save('/content/drive/My Drive/drive1/numpy_validation_datasets/SMIC_E_val_labels_64_3cls.npy', validation_labels)


# Load validation set from numpy array

#validation_images = np.load('/content/drive/My Drive/TEST_numpy_datasets/SMIC_E_TEST_train_images_64_3cls.npy')
#validation_labels = np.load('/content/drive/My Drive/TEST_numpy_datasets/SMIC_E_TEST_train_labels_64_3cls.npy')

#validation_images = np.load('/content/drive/My Drive/TEST_numpy_datasets/424_CASME_II_E_TEST_images_64_3cls.npy')
#validation_labels = np.load('/content/drive/My Drive/TEST_numpy_datasets/424_CASME_II_E_TEST_labels_64_3cls.npy')

#validation_images = np.load('/content/drive/My Drive/TEST_numpy_datasets/CASME2_E_TEST_images_64_3cls.npy')
#validation_labels = np.load('/content/drive/My Drive/TEST_numpy_datasets/CASME2_E_TEST_labels_64_3cls.npy')

#validation_images = np.load('/content/drive/My Drive/EVM20_numpy_TEST_datasets/424_SMIC_E_EVM20_TEST_train_images_64_3cls.npy')
#validation_labels = np.load('/content/drive/My Drive/EVM20_numpy_TEST_datasets/424_SMIC_E_EVM20_TEST_train_labels_64_3cls.npy')

#validation_images = np.load('/content/drive/My Drive/EVM20_numpy_TEST_datasets/424_CASME2_E_EVM20_TEST_train_images_64_3cls.npy')
#validation_labels = np.load('/content/drive/My Drive/EVM20_numpy_TEST_datasets/424_CASME2_E_EVM20_TEST_train_labels_64_3cls.npy')

#validation_images = np.load('/content/drive/My Drive/EVM20_numpy_TEST_datasets/428_CASME_II_E_EVM20_TEST_train_images_64_3cls.npy')
#validation_labels = np.load('/content/drive/My Drive/EVM20_numpy_TEST_datasets/428_CASME_II_E_EVM20_TEST_train_labels_64_3cls.npy')

validation_images = np.load('D:/lh/2021/smic_single/numpy/smic_test_images.npy')
validation_labels = np.load('D:/lh/2021/smic_single/numpy/smic_test_labels.npy')

print(validation_images.shape)
print(validation_labels.shape)
validation_images =np.transpose((validation_images),(0,2,3,4,1))


if not os.path.exists('D:/lh/2021/hdf5/'):
    os.makedirs('D:/lh/2021/hdf5/')
ck_callback = keras.callbacks.ModelCheckpoint('D:/lh/2021/hdf5/weights.{epoch:02d}-{val_f1:.6f}.hdf5',
                                                 monitor='val_f1',
                                                 mode='max', verbose=2,
                                                 save_best_only=True,
                                                 save_weights_only=True)
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs')

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [Metrics(valid_data=(validation_images, validation_labels)),checkpoint,reduce_r,reduce_lr,ck_callback,tb_callback]

#callbacks_list = [Metrics(valid_data=(validation_images, validation_labels)),checkpoint,ck_callback,tb_callback]
#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(np.ravel(train_labels,order='C')),
#                                                  np.ravel(train_labels,order='C'))
#print('class_weights',class_weights)#class_weights [ 0.75  1.5 ]
#class_weight_dict = dict(enumerate(class_weight))

"""
n_samples=164
n_clsses=3
clsses0=70
clsses1=51
clsses2=43





print("clsses0_weight=",n_samples/(n_clsses*clsses0))
print("clsses1_weight=",n_samples/(n_clsses*clsses1))
print("clsses2_weight=",n_samples/(n_clsses*clsses2))

n1_samples=341
n1_clsses=3
cl1sses0=102
cl1sses1=88
cl1sses2=151

print("clsses0_weight=",n1_samples/(n1_clsses*cl1sses0))
print("clsses1_weight=",n1_samples/(n1_clsses*cl1sses1))
print("clsses2_weight=",n1_samples/(n1_clsses*cl1sses2))


clsses0_weight= 0.780952380952381
clsses1_weight= 1.0718954248366013
clsses2_weight= 1.2713178294573644

cl1sses0_weight= 1.1143790849673203
cl1sses1_weight= 1.2916666666666667
cl1sses2_weight= 0.7527593818984547
"""

# Training the model
#hist = model.fit(train_images, train_labels, 
#                 validation_data = (validation_images, validation_labels),
#                 callbacks=callbacks_list, 
#                 batch_size = 32, 
#                 epochs = 100, 
#                 shuffle=True, 
#                 class_weight=None)#{0:0.5782312925170068,1:18.214285714285715,2:1.1305418719211822,3:0.36796536796536794,4:1.3492063492063493,5:5.204081632653061,6:1.457142857142857})#{0:1.114379,1:1.29166,2:0.75275})
#{0:0.780952380952381,1:1.0718954248366013,2:1.2713178294573644})
                 
#CAS(ME)2:{0:1.114379,1:1.29166,2:0.75275})
#SMIC:{0:0.780952380952381,1:1.0718954248366013,2:1.2713178294573644})
#CASME-II:{0:0.5782312925170068,1:18.214285714285715,2:1.1305418719211822,3:0.36796536796536794,4:1.3492063492063493,5:5.204081632653061,6:1.457142857142857}
"""
result_dir='/content/drive/My Drive/drive2/result_SMIC_e/'
#now = datetime.datetime.now()
#now = time.strftime("%Y-%m-%d %H:%M:%S")
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
plt.savefig(os.path.join(result_dir,'0maxp_accuracy_incep_3cls_0.0002_03232124.png'))
plt.close()

plt.plot(hist.history['loss'], marker='.')
plt.plot(hist.history['val_loss'], marker='.')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig(os.path.join(result_dir, '0maxp_loss_incep_3cls_0.0002_03232124.png'))
plt.close()


# Finding Confusion Matrix using pretrained weights

predictions = model.predict(validation_images)
predictions_labels = np.argmax(predictions, axis=1)
validation_labels = np.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)

"""

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    #Input
    #- cm : 计算出的混淆矩阵的值
    #- classes : 混淆矩阵中每一行每一列对应的列
    #- normalize : True:显示百分比, False:显示个数
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

pred_y = model.predict(validation_images)
pred_label = np.argmax(pred_y, axis=1)
true_label = np.argmax(validation_labels, axis=1)

confusion_mat = confusion_matrix(true_label, pred_label)
exp=['negative','positive','surprise']
#exp=['Disgust','Fear','Happiness','Others','Repression','Sadness','Surprise']
#exp=['Anger','Disgust','Happy']
#exp=['Negative','Positive','Surprise']
plot_confusion_matrix(confusion_mat, classes = exp,normalize=False)#True False
print(confusion_mat)
plt.savefig('D:/lh/2021/smic_single/plt/smic333.png', format='png')
plt.show()
###输出测试结果