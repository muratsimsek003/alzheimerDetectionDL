import os
for dirname, _, filenames in os.walk('datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import InputLayer, BatchNormalization,Activation, MaxPool2D
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.xception import Xception

import tqdm
import keras
import glob
import cv2
import warnings
warnings.filterwarnings("ignore")

import scipy
physical_devices = tensorflow.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
seed = 1842
tensorflow.random.set_seed(seed)
np.random.seed(seed)

batch_size=60
img_size=(224,224)
image_generator = ImageDataGenerator(rescale=1/255., validation_split=0) #shear_range =.25, zoom_range =.2, horizontal_flip = True, rotation_range=20)
train_data = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory='dataset/train',
                                                 shuffle=True,
                                                 target_size=img_size,
                                                 subset="training",
                                                 class_mode='categorical')

image_generator = ImageDataGenerator(rescale=1/255,validation_split=0.2)
validation_data= image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory='dataset/test',
                                                 shuffle=True,
                                                 target_size=img_size,
                                                 class_mode='categorical')


submission = image_generator.flow_from_directory(
                                                 directory='dataset/test',
                                                 shuffle=False,
                                                subset="validation",
                                                 target_size=img_size,
                                                 class_mode=None)


fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize=(20,20))

for i in range(5):
    rand1 = np.random.randint(len(train_data))
    rand2 = np.random.randint(60)
    ax[i].imshow(train_data[rand1][0][rand2])
    ax[i].axis('off')
    a = train_data[rand1][1][rand2]
    if a[0] == 1:
        ax[i].set_title('Mild Dementia')
    elif a[1] == 1:
        ax[i].set_title('Moderate Dementia')
    elif a[2] == 1:
        ax[i].set_title('Non Demetia')
    elif a[3] == 1:
        ax[i].set_title('Very Mild Dementia')

plt.show()

batch_size=60
epoch=35
callback= keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=8,
                                            restore_best_weights=True)

#---------------------------------------------
#####VGG16########
#---------------------------------------------
vgg16 = VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

for layer in vgg16.layers:
    layer.trainable = False


x = Flatten()(vgg16.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelvgg16 = Model(inputs=vgg16.input, outputs=out)

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

#compiling
modelvgg16.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

#Summary
modelvgg16.summary()

hist_vgg16=modelvgg16.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

loss_vgg16,accuracy_vgg16 = modelvgg16.evaluate(validation_data)



vgg19 = VGG19(input_shape=(224, 224, 3), weights="imagenet", include_top=False)


for layer in vgg19.layers:
    layer.trainable = False


x = Flatten()(vgg19.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelvgg19 = Model(inputs=vgg19.input, outputs=out)

# compiling
modelvgg19.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


modelvgg19.summary()

hist_vgg19=modelvgg19.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

#---------------------------------------------
#####VGG19########
#---------------------------------------------
vgg19 = VGG19(input_shape=(224, 224, 3), weights="imagenet", include_top=False)


for layer in vgg19.layers:
    layer.trainable = False


x = Flatten()(vgg19.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelvgg19 = Model(inputs=vgg19.input, outputs=out)

# compiling
modelvgg19.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


modelvgg19.summary()

hist_vgg19=modelvgg19.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

loss_vgg19,accuracy_vgg19= modelvgg19.evaluate(validation_data)


###########RESNET50###############
rn50 = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in rn50.layers:
    layer.trainable = False
x = Flatten()(rn50.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelrn50= Model(inputs=rn50.input, outputs=out)

#compiling
modelrn50.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
#Summary
modelrn50.summary()


hist_rn50=modelrn50.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

loss_rn50,accuracy_rn50= modelrn50.evaluate(validation_data)


#########RESNET101^##################

rn101 = ResNet101(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in rn101.layers:
    layer.trainable = False
x = Flatten()(rn101.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelrn101= Model(inputs=rn101.input, outputs=out)

# Compiling
modelrn101.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

#Summary
modelrn101.summary()

hist_rn101=modelrn101.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

loss_rn101,accuracy_rn101= modelrn101.evaluate(validation_data)


#####MOBİLENET#########

mobilenet = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in mobilenet.layers:
    layer.trainable = False
x = Flatten()(mobilenet.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelmnet= Model(inputs=mobilenet.input, outputs=out)

#Compiling
modelmnet.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
#Summary
modelmnet.summary()


hist_mnet=modelmnet.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

loss_mnet,accuracy_mnet= modelmnet.evaluate(validation_data)

mobilenetv2= MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in mobilenetv2.layers:
    layer.trainable = False
x = Flatten()(mobilenetv2.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modelmnetv2= Model(inputs=mobilenetv2.input, outputs=out)

#Compiling
modelmnetv2.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

#Summary
modelmnetv2.summary()

hist_mnetv2=modelmnetv2.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)

loss_mnetv2,accuracy_mnetv2= modelmnetv2.evaluate(validation_data)



###########DENSENET169###############################
#####################################################



dnet169 = DenseNet169(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in dnet169.layers:
    layer.trainable = False
x = Flatten()(dnet169.output)
x = Dense(128, activation='relu')(x)
out = Dense(4, activation='softmax')(x)

modeldnet169= Model(inputs=dnet169.input, outputs=out)

# Compiling
modeldnet169.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Summary
modeldnet169.summary()

hist_dnet169=modeldnet169.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=callback)


loss_dnet169,accuracy_dnet169= modeldnet169.evaluate(validation_data)