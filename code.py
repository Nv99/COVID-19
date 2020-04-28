"""
here we are going to write a machine learning code which will help us to classify the differece in between the people having corona 
virus or not
The dataset for this is downloded from kaggle which includes the different X-Ray images of people showing whether they are suffering from
pnemonia(corona) or are normal(i.e., did not have any issues)

"""
"""
To create this model we will use different libraries of python such as using pandas library for reading Comma Seperated Values file
(csv file) to using computer vision library such as cv2 for reading and resizeing the image etc

"""

import pandas as pd                   #data processing for CSV file, it makes importing and analyzing the data much easier
import numpy as np                    #linear alzebra


#the thing inside the brackets is the path where I have saved my csv file 
#the command pd.read_csv will help us to read the contents of the csv data

corona = pd.read_csv('C:/Users/DELL/Downloads/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

corona.head()                        #this command will give us the overview of the csv files contents as it returns top 5(by default) 
                                     # rows of a data frame or series

corona.head()                        #this function is used to get a concise summary of the dataframe
                                     #in short in order to get the quick overview of the dataset we use .info command

IMG_SIZE = 50

#here we have different columns 
#column label contains category of normal and pnemonia therefore we have to divide them accourdingly
#if we succesfully divided the normal and pnemonia category then it will become easy for us to further divide it into train and test datasets

normal_people = corona.loc[corona["Label"]=="Normal"]  #.loc command access a group of rows and columns
normal_people.shape                                    # this will give us the shape of the newly created datasets

pnemonia_people = corona.loc[corona["Label"]=="Pnemonia"] 
pnemonia_people.shape

#Now we will further divide the normal_people dataset into train and test so that it will become easy for us to further train and test the 
#specified images, the same will be done for the pnemonia_people

normal_train = normal_people.loc[normal_people["Dataset_type"]=="TRAIN"]
normal_train.shape

normal_test = normal_people.loc[normal_people["Dataset_type"]=="TEST"]
normal_test.shape

pnemonia_train = pnemonia_people.loc[pnemonia_people["Dataset_type"]=="TRAIN"]
pnemonia_train.shape

pnemonia_test = pnemonia_people.loc[pnomonia_people["Dataset_type"]=="TEST"]
pnemonia_test.shape

#Now we will import the computer vision library and read and resize the images then we will save our result in the list which we will make later
#The path to the images will be created with the help of the csv file first we will copy and paste the path of our images here and will further
#add the string of the image names,as the images(dataset) is labeled with respect to the csv files

train_X = []
train_Y = []

import cv2

for index in range(0,1342):
  path_normal = 'C:/Users/DELL/Downloads/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+str(normal_train.iloc[index]['X_ray_image_name'])
  new_array_normal = cv2.resize(cv2.imread(path_normal,1),(IMG_SIZE,IMG_SIZE))
  test_X.append(new_array_normal)
  test_Y.append(0.0)                             #this is the label which we have provided

for index in range(0,1500):
  path_pnemonia = 'C:/Users/DELL/Downloads/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+str(pnemonia_train.iloc[index]['X_ray_image_name'])
  new_array_pnemonia = cv2.resize(cv2.imread(path_pnemonia,1), (IMG_SIZE,IMG_SIZE))
  train_X.append(new_array_pnemonia)
  train_Y.append(1.0)
  
 
test_X = []
test_Y = []
for index in range(0,234):
    path_normal = 'C:/Users/DELL/Downloads/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+str(normal_test.iloc[index]['X_ray_image_name'])
    new_array_normal = cv2.resize(cv2.imread(path_normal,1),(IMG_SIZE,IMG_SIZE))
    test_X.append(new_array_normal)
    test_Y.append(0.0)
for index in range(0,390):
    path_pnemonia = 'C:/Users/DELL/Downloads/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+str(pnemonia_test.iloc[index]['X_ray_image_name'])
    new_array_pnemonia = cv2.resize(cv2.imread(path_pnemonia,1),(IMG_SIZE,IMG_SIZE))
    test_X.append(new_array_pnemonia)
    test_Y.append(1.0)
 
import numpy as np
train_X = np.array(train_X)              
train_Y = np.array(train_Y)

# importing the important libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

#Building the model
model = Sequential()

#3 convolutional layer

model.add(Conv2D(32, (3, 3), input_shape = train_X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2 hidden layers

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

#The output layer with 3 neurons

model.add(Dense(3))
model.add(Activation("softmax"))

#compiling the model with some basic parameters
#here we have used the adam optimizer

model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

#Training the model with 50 iterations(epochs) 
#validation_split corresponds to the percentage of images used for the validation phase compared to all the images

history = model.fit(train_X,train_Y, batch_size = 64, epochs = 50, validation_split = 0.1)

#.summary returns the summarized representationof the index

model.summary()

#Printing a graph showing the accuracy changes during the training phase

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')

#for testing the model we will make use of test_X and test_Y list which we have made earlier 

test_X = np.array(test_X)
test_Y = np.array(test_Y)
model.evaluate(test_X,test_Y)      #this command is used to evaluate the model

#this set of commands below is used to predict wheater our generalization is good or bad
#here we will pass the path of the test image which we have to predict accordingly

image = 'C:/Users/DELL/Downloads/covid-chest-xray/test/IM-0001-0001.jpeg'
new_array = cv2.resize(cv2.imread(image,1),(IMG_SIZE,IMG_SIZE))
new_array = new_array.reshape(-1,50,50,3)
prediction = model.predict([new_array])
prediction = list(prediction[0])
print(prediction)
print(prediction.index(max(prediction)))
