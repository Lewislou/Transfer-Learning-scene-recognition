#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import shutil # 复制

# 随机数
np.random.seed(2500)

#root_train = 'H:\\MATLAB\\data\\train_split' #  切分后的训练集路径
#root_val = 'H:\\MATLAB\\data\\val_split' # 切分后的验证集路径
#root_train = 'H:\\MATLAB\\training\\training' #  切分后的训练集路径
root_train = '/home/wl4u19/data/train_split' #
root_val = '/home/wl4u19/data/val_split' # 切分后的验证集路径
root_total = 'C:\\Users\\wl4u19\\Wei Lou\\aug_data' # 原来训练集图像存放的地方

# scene的种类，15种
SceneNames = ['bedroom','Coast','Forest','Highway','industrial','Insidecity','kitchen','livingroom','Mountain','Office','OpenCountry','store','Street','Suburb','TallBuilding']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8


# In[2]:


'''
import cv2
image_size = 256
for scene in SceneNames:
    total_images = os.listdir(os.path.join(root_total, scene))
    for img in total_images:
        print(os.path.join(root_total,scene,img))       
        pic = cv2.imread(os.path.join(root_total,scene,img))
        print(pic.shape)
        pic = cv2.resize(pic,(image_size,image_size))
        #pic = cv2.cvtColor(cv2.resize(pic,(image_size,image_size)),cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(root_total,scene,img),pic)
    '''


# In[3]:

# In[3]:


###Inception V3 Training
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
import os
from keras.layers import Flatten, Dense, AveragePooling2D,GlobalAveragePooling2D,Dropout
from keras.models import Model,Sequential
from keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from keras.models import load_model
print('Loading Inception Weights ...')
learning_rate = 0.000001  #学习率
img_width = 256 #图片宽度 299 -> 200
img_height = 256  #图片高度
nbr_train_samples = 13200 #训练集数量
nbr_validation_samples = 3300 # 验证集数量
batch_size = 128
nbr_epochs = 20
train_data_dir = root_train # 训练集路径
val_data_dir = root_val # 验证集路径

SceneNames = ['bedroom','Coast','Forest','Highway','industrial','Insidecity','kitchen','livingroom','Mountain','Office','OpenCountry','store','Street','Suburb','TallBuilding']
print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, 
                                weights='imagenet',
                                input_tensor=None, 
                                input_shape=(256, 256, 3))

#Compile each model
print('Adding Average Pooling Layer and Softmax Output Layer to InceptionResNetV2 ...')
output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((6, 6), strides=(6, 6), name='avg_pool')(output) 
output = Flatten(name='flatten')(output) 
output = Dense(15, activation='softmax', name='predictions')(output)
InceptionV3_model = Model(InceptionV3_notop.input, output) 
optimizer = Adam(lr = learning_rate, decay = 0.00001)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
v3_model_file = "V3.hdf5" 
best_model = ModelCheckpoint(v3_model_file, monitor='val_loss', verbose = 1, save_best_only = True)



train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,)
val_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir, #文件路径来自于子文件夹，flow表示接受数据和标签为参数
        target_size = (img_width, img_height), #图像将被resize成该尺寸
        batch_size = batch_size, #
        shuffle = True, #是否随机打乱数据
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = SceneNames, #子文件夹的列表
        class_mode = 'categorical')#该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签


validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = SceneNames,
        class_mode = 'categorical')
#InceptionV3_model = load_model('V3.hdf5')
history =InceptionV3_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])

'''
from keras.models import load_model
batch_size = 16
nbr_epochs = 30
learning_rate = 0.00001  #学习率
VGG16_model = load_model('VGG16.hdf5') 
history2 = VGG16_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size))
VGG16_model.save('VGG16.hdf5')
'''


# In[7]:


import matplotlib.pyplot as plt
print(history.history.keys())
plt.switch_backend('agg')    #服务器上面保存图片  需要设置这个
#acc
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc1.jpg')
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss1.jpg')


# In[ ]:


###Inception V2 Training
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
import os
from keras.layers import Flatten, Dense, AveragePooling2D,GlobalAveragePooling2D,Dropout
from keras.models import Model,Sequential
from keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from keras.models import load_model
print('Loading Inception Weights ...')
learning_rate = 0.00001  #学习率
img_width = 256 #图片宽度 299 -> 200
img_height = 256  #图片高度
nbr_train_samples = 13200 #训练集数量
nbr_validation_samples = 3300 # 验证集数量
batch_size = 128
nbr_epochs = 20
train_data_dir = root_train # 训练集路径
val_data_dir = root_val # 验证集路径

SceneNames = ['bedroom','Coast','Forest','Highway','industrial','Insidecity','kitchen','livingroom','Mountain','Office','OpenCountry','store','Street','Suburb','TallBuilding']
print('Loading InceptionResNetV2 Weights ...')
InceptionResNetV2_notop = InceptionResNetV2(include_top=False, 
                                weights='imagenet',
                                input_tensor=None, 
                                input_shape=(256, 256, 3))

#Compile each model
print('Adding Average Pooling Layer and Softmax Output Layer to InceptionResNetV2 ...')
output = InceptionResNetV2_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((6, 6), strides=(6, 6), name='avg_pool')(output) 
output = Flatten(name='flatten')(output) 
output = Dense(15, activation='softmax', name='predictions')(output)
InceptionResNetV2_model = Model(InceptionResNetV2_notop.input, output) 
optimizer = Adam(lr = learning_rate, decay = 0.000001)
InceptionResNetV2_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
v2_model_file = "V2.hdf5" 
best_model = ModelCheckpoint(v2_model_file, monitor='val_loss', verbose = 1, save_best_only = True)



train_datagen = ImageDataGenerator(
    featurewise_center=True,
     featurewise_std_normalization=True,
    shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)
val_datagen = ImageDataGenerator( 
    featurewise_center=True,
     featurewise_std_normalization=True)
train_generator = train_datagen.flow_from_directory(
        train_data_dir, #文件路径来自于子文件夹，flow表示接受数据和标签为参数
        target_size = (img_width, img_height), #图像将被resize成该尺寸
        batch_size = batch_size, #
        shuffle = True, #是否随机打乱数据
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = SceneNames, #子文件夹的列表
        class_mode = 'categorical')#该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签


validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = SceneNames,
        class_mode = 'categorical')
#InceptionResNetV2_model = load_model('V2.hdf5')
history2 = InceptionResNetV2_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])

'''
from keras.models import load_model
batch_size = 16
nbr_epochs = 30
learning_rate = 0.00001  #学习率
VGG16_model = load_model('VGG16.hdf5') 
history2 = VGG16_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size))
VGG16_model.save('VGG16.hdf5')
'''


# In[ ]:


print(history2.history.keys())
plt.switch_backend('agg')    #服务器上面保存图片  需要设置这个
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc2.jpg')
#loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss2.jpg')


# In[ ]:


###ResNet50 Training
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
import os
from keras.layers import Flatten, Dense, AveragePooling2D,GlobalAveragePooling2D,Dropout
from keras.models import Model,Sequential
from keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from math import ceil

print('Loading Inception Weights ...')
learning_rate = 0.00001  #学习率
img_width = 256 #图片宽度 299 -> 200
img_height = 256  #图片高度
nbr_train_samples = 13200 #训练集数量
nbr_validation_samples = 3300 # 验证集数量
batch_size = 128
nbr_epochs = 20
train_data_dir = root_train # 训练集路径
val_data_dir = root_val # 验证集路径

SceneNames = ['bedroom','Coast','Forest','Highway','industrial','Insidecity','kitchen','livingroom','Mountain','Office','OpenCountry','store','Street','Suburb','TallBuilding']
print('Loading ResNet50 Weights ...')
ResNet50_notop = ResNet50(include_top=False, 
                                weights='imagenet',
                                input_tensor=None, 
                                input_shape=(256, 256, 3))

#Compile each model
print('Adding Average Pooling Layer and Softmax Output Layer to InceptionResNetV2 ...')
output = ResNet50_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((6, 6), strides=(6, 6), name='avg_pool')(output) 
output = Flatten(name='flatten')(output) 
output = Dense(15, activation='softmax', name='predictions')(output)
ResNet50_model = Model(ResNet50_notop.input, output) 
optimizer = Adam(lr = learning_rate, decay = 0.000001)
ResNet50_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
ResNet50_model_file = "Res.hdf5" 
best_model = ModelCheckpoint(ResNet50_model_file, monitor='val_loss', verbose = 1, save_best_only = True)



train_datagen = ImageDataGenerator(
    featurewise_center=True,
     featurewise_std_normalization=True,
    shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)
val_datagen = ImageDataGenerator( 
    featurewise_center=True,
     featurewise_std_normalization=True)
train_generator = train_datagen.flow_from_directory(
        train_data_dir, #文件路径来自于子文件夹，flow表示接受数据和标签为参数
        target_size = (img_width, img_height), #图像将被resize成该尺寸
        batch_size = batch_size, #
        shuffle = True, #是否随机打乱数据
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = SceneNames, #子文件夹的列表
        class_mode = 'categorical')#该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签


validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = SceneNames,
        class_mode = 'categorical')
history3 = ResNet50_model.fit_generator( 
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])

'''
from keras.models import load_model
batch_size = 16
nbr_epochs = 30
learning_rate = 0.00001  #学习率
VGG16_model = load_model('VGG16.hdf5') 
history2 = VGG16_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size))
VGG16_model.save('VGG16.hdf5')
'''


# In[ ]:


print(history3.history.keys())
plt.switch_backend('agg')    #服务器上面保存图片  需要设置这个
plt.plot(history3.history['acc'])
plt.plot(history3.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc3.jpg')
#loss
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss3.jpg')


# In[ ]:


###MobileNetV2 Training
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
import os
from keras.layers import Flatten, Dense, AveragePooling2D,GlobalAveragePooling2D,Dropout
from keras.models import Model,Sequential
from keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from keras.models import load_model
print('Loading MobileNetV2 Weights ...')
learning_rate = 0.00001  #学习率
img_width = 256 #图片宽度 299 -> 200
img_height = 256  #图片高度
nbr_train_samples = 13200 #训练集数量
nbr_validation_samples = 3300 # 验证集数量
batch_size = 128
nbr_epochs = 20
train_data_dir = root_train # 训练集路径
val_data_dir = root_val # 验证集路径

SceneNames = ['bedroom','Coast','Forest','Highway','industrial','Insidecity','kitchen','livingroom','Mountain','Office','OpenCountry','store','Street','Suburb','TallBuilding']
print('Loading MobileNetV2 Weights ...')
MobileNetV2_notop = MobileNetV2(include_top=False, 
                                weights='imagenet',
                                input_tensor=None, 
                                input_shape=(256, 256, 3))

#Compile each model
print('Adding Average Pooling Layer and Softmax Output Layer to MobileNetV2 ...')
output = MobileNetV2_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((6, 6), strides=(6, 6), name='avg_pool')(output) 
output = Flatten(name='flatten')(output) 
output = Dense(15, activation='softmax', name='predictions')(output)
MobileNetV2_model = Model(MobileNetV2_notop.input, output) 
optimizer = Adam(lr = learning_rate, decay = 0.000001)
MobileNetV2_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
MobileNetV2_model_file = "Mob.hdf5" 
best_model = ModelCheckpoint(MobileNetV2_model_file, monitor='val_loss', verbose = 1, save_best_only = True)



train_datagen = ImageDataGenerator(
    featurewise_center=True,
     featurewise_std_normalization=True,
    shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)
val_datagen = ImageDataGenerator( 
    featurewise_center=True,
     featurewise_std_normalization=True)
train_generator = train_datagen.flow_from_directory(
        train_data_dir, #文件路径来自于子文件夹，flow表示接受数据和标签为参数
        target_size = (img_width, img_height), #图像将被resize成该尺寸
        batch_size = batch_size, #
        shuffle = True, #是否随机打乱数据
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = SceneNames, #子文件夹的列表
        class_mode = 'categorical')#该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签


validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = SceneNames,
        class_mode = 'categorical')
#MobileNetV2_model= load_model('Mob.hdf5')
history4 = MobileNetV2_model.fit_generator( 
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])

'''
from keras.models import load_model
batch_size = 16
nbr_epochs = 30
learning_rate = 0.00001  #学习率
VGG16_model = load_model('VGG16.hdf5') 
history2 = VGG16_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size))
VGG16_model.save('VGG16.hdf5')
'''


# In[ ]:


print(history4.history.keys())
plt.switch_backend('agg')    #服务器上面保存图片  需要设置这个
plt.plot(history4.history['acc'])
plt.plot(history4.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc4.jpg')
#loss
plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss4.jpg')


# In[ ]:


'''
nbr_epochs = 30 


history1 = InceptionV3_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])
history2 = VGG16_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])
history3 = InceptionResNetV2_model.fit_generator(
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])
history4 = ResNet50_model.fit_generator( 
        train_generator,
        #samples_per_epoch = nbr_train_samples,
        steps_per_epoch = ceil(nbr_train_samples / batch_size),
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = ceil(nbr_validation_samples / batch_size),
        callbacks = [best_model])
'''




# In[ ]:




