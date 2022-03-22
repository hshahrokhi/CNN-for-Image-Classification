# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:51:30 2022

@author: Hamed Shahrokhi
"""
#%%Description
#    • This code uses Minimalist Histopathology Image Analysis Dataset (MHIST) as described in this link here https://bmirds.github.io/MHIST/. If you use this dataset, make sure to cite their paper at https://link.springer.com/chapter/10.1007/978-3-030-77211-6_2
#    • This code utilizes sequential models from Keras API of tensorflow library
#    • This model utilizes modelcheckpoint callbacks to store the best model through iterations
#    • After training, the best model is loaded and evaluated against test data
#    • Through trials, the structure of the model as well as then hyperparameters are optimized to achive higher performace (accuracy in this case)
#    • If you are to use this code, make sure that you updated the properties.yml file and all the directorites within the code to reflect the correct address to image files.
#      Just search for "change directory" in the code and you will know where you need to update your directory address



#%% importing packages

import pandas as pd
import os
import yaml
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

#%% Changing current directory

os.chdir(r'C:\Users\HMD\Desktop\Classification\data\data processing') #change directory: change it to where you placed this code 

#%%Constants
batchSize=32
epoch=100
learningRate=1e-3
validationSplit=0.1 # percentage of training data that will be used for validation


#%% image and excel description directories
propertyFileAddress= os.path.join(os.getcwd(),"properties.yml")

if not os.path.isfile(propertyFileAddress):
    print("properties file does not exist! add the properties.yml to the directory")
    os._exit(1)
else:
    print("properties file found")
    with open(propertyFileAddress) as f:
        properties=yaml.safe_load(f)
        imageFilesDirectory= properties['Directories']['rawImageDirectory']
        annotationFileDirectory=properties['Directories']['annotationsCSV']

if os.path.exists(os.path.join(imageFilesDirectory,"train")):
    shutil.rmtree(os.path.join(imageFilesDirectory,"train"))
    
os.mkdir(os.path.join(imageFilesDirectory,"train"))
os.mkdir(os.path.join(imageFilesDirectory,"train","SSA"))
os.mkdir(os.path.join(imageFilesDirectory,"train","HP"))
trainDirectory=os.path.join(imageFilesDirectory,"train")
trainSSADirectory=os.path.join(imageFilesDirectory,"train","SSA")
trainHPDirectory=os.path.join(imageFilesDirectory,"train","HP")

if os.path.exists(os.path.join(imageFilesDirectory,"test")):
    shutil.rmtree(os.path.join(imageFilesDirectory,"test"))
os.mkdir(os.path.join(imageFilesDirectory,"test"))
os.mkdir(os.path.join(imageFilesDirectory,"test","SSA"))
os.mkdir(os.path.join(imageFilesDirectory,"test","HP"))
testDirectory=os.path.join(imageFilesDirectory,"test")
testSSADirectory=os.path.join(imageFilesDirectory,"test","SSA")
testHPDirectory=os.path.join(imageFilesDirectory,"test","HP")

if os.path.exists(os.path.join(imageFilesDirectory,"validation")):
    shutil.rmtree(os.path.join(imageFilesDirectory,"validation"))
os.mkdir(os.path.join(imageFilesDirectory,"validation"))
os.mkdir(os.path.join(imageFilesDirectory,"validation","SSA"))
os.mkdir(os.path.join(imageFilesDirectory,"validation","HP"))
validationDirectory=os.path.join(imageFilesDirectory,"validation")
validationSSADirectory=os.path.join(imageFilesDirectory,"validation","SSA")
validationHPDirectory=os.path.join(imageFilesDirectory,"validation","HP")

print("directories are successfully created")
#%% Moving Image Files to Proper Directories
annotations=pd.read_csv(os.path.join(annotationFileDirectory,"annotations.csv"))

trainSampleSize=0
testSampleSize=0
rollingCounterForValidation=0
for i in range(0,len(annotations)):
    if annotations['Partition'][i]=='train':
        trainSampleSize+=1
        rollingCounterForValidation+=1
        if rollingCounterForValidation==10:
            
            if annotations['Majority Vote Label'][i]=='SSA':
                shutil.copyfile(os.path.join(imageFilesDirectory,annotations['Image Name'][i]),os.path.join(validationSSADirectory,annotations['Image Name'][i]))
            elif annotations['Majority Vote Label'][i]=='HP':
                shutil.copyfile(os.path.join(imageFilesDirectory,annotations['Image Name'][i]),os.path.join(validationHPDirectory,annotations['Image Name'][i]))
            rollingCounterForValidation=0
        else:
            if annotations['Majority Vote Label'][i]=='SSA':
                shutil.copyfile(os.path.join(imageFilesDirectory,annotations['Image Name'][i]),os.path.join(trainSSADirectory,annotations['Image Name'][i]))
            elif annotations['Majority Vote Label'][i]=='HP':
                shutil.copyfile(os.path.join(imageFilesDirectory,annotations['Image Name'][i]),os.path.join(trainHPDirectory,annotations['Image Name'][i]))
    elif annotations['Partition'][i]=='test':
        testSampleSize+=1
        if annotations['Majority Vote Label'][i]=='SSA':
            shutil.copyfile(os.path.join(imageFilesDirectory,annotations['Image Name'][i]),os.path.join(testSSADirectory,annotations['Image Name'][i]))
        elif annotations['Majority Vote Label'][i]=='HP':
            shutil.copyfile(os.path.join(imageFilesDirectory,annotations['Image Name'][i]),os.path.join(testHPDirectory,annotations['Image Name'][i]))
            
            

print("image files are successfully copied to proper folders")

#%% Creating data feeder for training, validation and testing

#loading a sample image to get its size to use for target size
img = mpimg.imread(os.path.join(imageFilesDirectory,annotations['Image Name'][int(np.random.rand()*len(annotations))]))
# imgplot = plt.imshow(img)
width,height, channel=img.shape
# print("random sample image with the size of {}, {}, {}".format(str(width),str(heigh),str(channel)))
target_size=(width,height)


#creating datagenerators
datagen = ImageDataGenerator(rotation_range=90,zoom_range=0.2,horizontal_flip=True, vertical_flip=True)

train_it = datagen.flow_from_directory(trainDirectory, class_mode='categorical', batch_size=batchSize,target_size=target_size,color_mode='rgb',shuffle=True)
print(train_it.class_indices)

test_it = datagen.flow_from_directory(testDirectory, class_mode='categorical', batch_size=batchSize,target_size=target_size,color_mode='rgb',shuffle=True)
print(test_it.class_indices)

valid_it = datagen.flow_from_directory(validationDirectory, class_mode='categorical', batch_size=batchSize,target_size=target_size,color_mode='rgb',shuffle=True)
print(test_it.class_indices)

#%% Model Building


# defining a customized callback (ended up not using this)
class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.epoch_accuracy = {} # loss at given epoch
        self.epoch_loss = {} # accuracy at given epoch

    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch. 
        return

    def on_epoch_end(self, epoch, logs={}):
        # things done on end of the epoch
        self.epoch_accuracy[epoch] = logs.get("acc")
        self.epoch_loss[epoch] = logs.get("loss")
        self.model.save_weights("name-of-model-%d.h5" %epoch)
        print("epoch============================================================================================",epoch)
checkpointt = CustomModelCheckPoint()


#creating a model checkpoint for saving the best model
model_checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    'HHCNNModel.epoch{epoch:02d}-loss{val_accuracy:.2f}', monitor='val_accuracy', verbose=1, save_best_only=True)


#structure of the CNN
def diseaseDetectionModel():
    model = tf.keras.Sequential([
            
            tf.keras.layers.ZeroPadding2D(padding=(3,3), 
                         input_shape=(width, height, channel), data_format="channels_last"),
            tf.keras.layers.Conv2D(16, (3, 3), name = 'conv0'),
            tf.keras.layers.BatchNormalization(axis = 3, name = 'bn0'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2), name='max_pool0'),
            tf.keras.layers.Conv2D(32, (5,5), name = 'conv1'),
            tf.keras.layers.BatchNormalization(axis = 3, name = 'bn1'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2), name='max_pool1'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax', name='fc') 

        ])  
    return model

#instantiating the model
diseaseDetectionModelInstance = diseaseDetectionModel()

# getting the summary of the model (number of parameters, output of each latyer, etc.)
print(diseaseDetectionModelInstance.summary())


#setting up the optimizer and compiling the model
optimizer=Adam(learning_rate=learningRate)

diseaseDetectionModelInstance.compile(optimizer=optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

#priting the outputshape of each layer of the model
for layer in diseaseDetectionModelInstance.layers:
    print(layer.name,":",layer.output_shape)

#training the model
history=diseaseDetectionModelInstance.fit(train_it, validation_data=valid_it, epochs=epoch, callbacks=[tf.keras.callbacks.History(),model_checkpoint_callback]) #, steps_per_epoch=50,validation_steps=32 ,callbacks=[tf.keras.callbacks.History()]

#saving trained model's weights
diseaseDetectionModelInstance.save_weights("weights")

#%% Visualizing training performance


print(history.history.keys())
print(diseaseDetectionModelInstance.metrics_names)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%Loading the best model

NEWMODEL=tf.keras.models.load_model(r'C:\Users\HMD\Desktop\Classification\data\data processing\HHCNNModel.epoch51-loss0.77') #change directory: change it to reflect the address for the best saved model

#evaluting the model on test data
xxx=NEWMODEL.evaluate_generator(test_it, 10)

#testing the output of the model for a few known images from test directorty
img = mpimg.imread(r'C:\Users\HMD\Desktop\Classification\data\MHIST\images\MHIST_dcc.png') #change directory: change it to the address of an arbiterary known image from test directory
imgg=img.reshape((-1,224,224,3))
zzz= NEWMODEL.predict_generator(imgg)

#%% Prediction for all the HP or SSA in training

fieless=os.listdir(r'C:\Users\HMD\Desktop\Classification\data\MHIST\images\train\SSA') #change directory: change it to a directory that you want to evalue its images; an example could be SSA images in test directory
resultsSSA=[]
for f in fieless:
    imgAddress=os.path.join(r'C:\Users\HMD\Desktop\Classification\data\MHIST\images\train\SSA',f)#change directory: change it to a directory that you want to evalue its images; an example could be SSA images in test directory
    img = mpimg.imread(imgAddress)
    imgg=img.reshape((-1,224,224,3))
    zzz= NEWMODEL.predict_generator(imgg)
    resultsSSA.append(zzz)
    
    
fieless=os.listdir(r'C:\Users\HMD\Desktop\Classification\data\MHIST\images\train\HP') #change directory: change it to a directory that you want to evalue its images; an example could be SSA images in test directory
resultsHP=[]
for f in fieless:
    imgAddress=os.path.join(r'C:\Users\HMD\Desktop\Classification\data\MHIST\images\train\HP',f)#change directory: change it to a directory that you want to evalue its images; an example could be SSA images in test directory
    img = mpimg.imread(imgAddress)
    imgg=img.reshape((-1,224,224,3))
    zzz= NEWMODEL.predict_generator(imgg)
    resultsHP.append(zzz)













       



