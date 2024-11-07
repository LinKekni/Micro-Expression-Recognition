#SMIC_E_Train_VAL DATA
import sys
#sys.path.append("/content/drive/My Drive/drive1/")
#!chdir('/content/drive/"My Drive"/drive1')
import warnings
warnings.filterwarnings("ignore")
import itertools
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
##调用模型
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


#disgustpath = "D:/LEAR_all_video/casme2_TIM_20_6cls_pic/disgust/"
#happinesspath = "D:/LEAR_all_video/casme2_TIM_20_6cls_pic/happiness/"
#repressionpath = "D:/LEAR_all_video/casme2_TIM_20_6cls_pic/repression/"
#sadnesspath = "D:/LEAR_all_video/casme2_TIM_20_6cls_pic/sadness/"
#surprisepath = "D:/LEAR_all_video/casme2_TIM_20_6cls_pic/surprise/"
#otherspath = "D:/LEAR_all_video/casme2_TIM_20_6cls_pic/others/"
## dis hap rep sad sur others

directorylisting = os.listdir(negativepath)
#print(directorylisting)
for video in directorylisting:
    frames = []
    videopath = negativepath + video
    #loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framelisting = os.listdir(videopath)
    framerange = [x + 0 for x in range(20)]#72 96
    #print(framerange)
    for frame in framerange:
        imagepath = videopath + "/" + framelisting[frame]
        image = cv2.imread(imagepath)
        imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    frames = np.asarray(frames)
    videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)
    print('negative=', training_list)

directorylisting = os.listdir(positivepath)
for video in directorylisting:
    frames = []
    videopath = positivepath + video
    #loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framelisting=os.listdir(videopath)
    framerange = [x + 0 for x in range(20)]#72 96
    for frame in framerange:
            imagepath = videopath + "/" + framelisting[frame]
            image = cv2.imread(imagepath)
            imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
            grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
            frames.append(grayimage)
    frames = np.asarray(frames)
    videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)
    print('positive=',training_list)

directorylisting = os.listdir(surprisepath)
for video in directorylisting:
        frames = []
        videopath = surprisepath + video
        #loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
        framelisting=os.listdir(videopath)
        framerange = [x + 0 for x in range(20)]#72 96
        for frame in framerange:
                imagepath = videopath + "/" + framelisting[frame]
                image = cv2.imread(imagepath)
                imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
        frames = np.asarray(frames)
        videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)
        print('surprise=',training_list)

#print('training_list',training_list)
training_list = np.asarray(training_list)
trainingsamples = len(training_list)
print('trainingsamples',trainingsamples)

traininglabels = np.zeros((trainingsamples, ), dtype = int)


### traininglabels[0:76] = 0
### traininglabels[76:170] = 1
### traininglabels[170:206] = 2

#traininglabels[0:102] = 0
#traininglabels[102:253] = 1
#traininglabels[253:341] = 2


#neg pos sur others
#traininglabels[0:449] = 0
#traininglabels[449:641] = 1
#traininglabels[641:686] = 2
#traininglabels[156:255] = 3

#traininglabels[0:699] = 0 #gai
#traininglabels[699:1239] = 1 #gai
#traininglabels[1239:1669] = 2 #gai

traininglabels[0:699] = 0 #gai
traininglabels[699:1209] = 1 #gai
traininglabels[1209:1639] = 2 #gai
700
510
430
# dis hap rep sad sur others
#traininglabels[0:63] = 0
#traininglabels[63:95] = 1
#traininglabels[95:122] = 2
#traininglabels[122:129] = 3
#traininglabels[129:154] = 4
#traininglabels[154:253] = 5


traininglabels = np_utils.to_categorical(traininglabels, 3)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = np.zeros((trainingsamples,1,image_rows, image_columns,image_depth ))
#print(training_set.shape())

for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h,:,:,:]

training_set = training_set.astype('float32')
training_set -= np.mean(training_set)
training_set /= np.max(training_set)

#print(trainingframes.shape()) 
#print(training_set.shape())
# Save training images and labels in a numpy array
np.save('/content/drive/My Drive/drive1/numpy_training_datasets/SMIC_E_images_64_3cls.npy', training_set)
np.save('/content/drive/My Drive/drive1/numpy_training_datasets/SMIC_E_labels_64_3cls.npy', traininglabels)




####以上是处理数据集部分、分为训练集和验证集



# Load training images and labels that are stored in numpy array
training_set = np.load('D:/lh/2021/smic_single/numpy/smic_train_images.npy')
traininglabels =np.load('D:/lh/2021/smic_single/numpy/smic_train_labels.npy')

#/content/drive/My Drive/421_SMIC_E_TRAIN_images_64_3cls.npy

#print(training_set.shape)
training_set =np.transpose((training_set),(0,2,3,4,1))




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
###模型加载



def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)
reduce_r = ReduceLROnPlateau(monitor='val_loss',verbose=1,factor=0.8,patience=3, mode='auto',min_lr=0)
#ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
optimizer = 'adam' #Lookahead(RAdam(lr))#RAdam adam
#model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
model.compile(optimizer=adam(lr=0.01),loss = 'categorical_crossentropy', metrics = ['acc'])
#keras.backend.get_session().run(tf.global_variables_initializer()) #gai

###定义训练参数
model.summary()




# Load pre-trained weights
"""
model.load_weights('weights_microexpstcnn/weights-improvement-53-0.88.hdf5')
"""
#model.load_weights('./checkpoints/weights.88-0.893939.hdf5')


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

filepath="D:/lh/2021/smic_single/hdf5/-{epoch:02d}-{val_acc:.6f}_SMIC_333.hdf5"
###训练结束、输出训练hdf5权重

# Spliting the dataset into training and validation sets
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, 
                                    test_size=0.111, random_state=4,stratify=traininglabels)
###训练集与验证集划分

# Save validation set in a numpy array

np.save('D:/lh/2021/smic_single/numpy/smic_val_images.npy', validation_images)
np.save('D:/lh/2021/smic_single/numpy/smic_val_labels.npy', validation_labels)
###保存划分后的验证集

# Load validation set from numpy array

validation_images = np.load('D:/lh/2021/smic_single/numpy/smic_val_images.npy')
validation_labels = np.load('D:/lh/2021/smic_single/numpy/smic_val_labels.npy')

###加载验证集

if not os.path.exists('D:/lh/2021/smic_single/hdf5'):
    os.makedirs('D:/lh/2021/smic_single/hdf5')
ck_callback = keras.callbacks.ModelCheckpoint('D:/lh/2021/smic_single/hdf5/weights.{epoch:02d}-{val_f1:.6f}.hdf5',
                                                 monitor='val_f1',
                                                 mode='max', verbose=2,
                                                 save_best_only=True,
                                                 save_weights_only=True)
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs')

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [Metrics(valid_data=(validation_images, validation_labels)),checkpoint,reduce_r,reduce_lr,ck_callback,tb_callback]

#callbacks_list = [Metrics(valid_data=(validation_images, validation_labels)),checkpoint,ck_callback,tb_callback]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(np.ravel(train_labels,order='C')),
                                                  np.ravel(train_labels,order='C'))
print('class_weights',class_weights)#class_weights [ 0.75  1.5 ]
#class_weight_dict = dict(enumerate(class_weight))

###验证参数定义

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
hist = model.fit(train_images, train_labels, 
                 validation_data = (validation_images, validation_labels),
                 callbacks=callbacks_list, 
                 batch_size = 32, 
                 epochs = 100, 
                 shuffle=True, 
                 class_weight=None)
#{0:0.780952380952381,1:1.0718954248366013,2:1.2713178294573644})
                 
#CAS(ME)2:{0:1.114379,1:1.29166,2:0.75275})
#SMIC:{0:0.780952380952381,1:1.0718954248366013,2:1.2713178294573644})
#CASME-II:{0:0.5782312925170068,1:18.214285714285715,2:1.1305418719211822,3:0.36796536796536794,4:1.3492063492063493,5:5.204081632653061,6:1.457142857142857}

result_dir='D:/lh/2021/smic_single/plt/'
#now = datetime.datetime.now()
#now = time.strftime("%Y-%m-%d %H:%M:%S")
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
plt.savefig(os.path.join(result_dir,'smic_333.png'))
plt.close()

plt.plot(hist.history['loss'], marker='.')
plt.plot(hist.history['val_loss'], marker='.')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig(os.path.join(result_dir, 'SMIC333.png'))
plt.close()


# Finding Confusion Matrix using pretrained weights

predictions = model.predict(validation_images)
predictions_labels = np.argmax(predictions, axis=1)
validation_labels = np.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)
###得到训练过程准确率、loss曲线、混淆矩阵的plt结果

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
exp=['anger','happy','disgust']
plot_confusion_matrix(confusion_mat, classes = exp,normalize=True)#True
print(confusion_mat)
plt.savefig('./CAS(ME)2_confusion_matrix3.png', format='png')
plt.show()
"""