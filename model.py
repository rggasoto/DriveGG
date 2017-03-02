from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation,Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Sequential, Model
import cv2, numpy as np
from TrainingData import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
from math import pi

center = 0
steering = 1
angleTS = 0.25

def vgg(inputshape):
    #Creates tensor to receive input with desired shape
    input_layer = Input(shape=inputshape)
    #Creates VGG16 base model with weights from imagenet. 
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    #Extend base model to include two fully connected with dropout and a final 1 neuron layer for the steering.
    layer = base_model.output
    layer = Flatten()(layer) 
    layer = Dense(1024, activation='relu', name='fc')(layer)
    #layer = Dropout(0.5)(layer)
    #The idea for this two neurons layers is that it gives a linear combination on how much to steer left, and how much to steer right.
    layer = Dense(2, activation='linear', name='fc2')(layer)    
    #Final layer, this is the steering output.
    layer = Dense(1, activation='linear', name='predictions')(layer)

    model = Model(input=base_model.input, output=layer)
    return model

def get_nvidia_model(input_shape,lstm = False,lstm_count = 10):
    """Returns the Nvidia Model
    """
    # End to End Learning for Self-Driving Cars
    # Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner
    # Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller
    # Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba
    # https://arxiv.org/abs/1604.07316
    model = Sequential()
    #model.add(Input(shape=input_shape))    
    model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape = input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))    
    model.add(Flatten())
    if lstm:
        for i in range(lstm_count):
            model.add(LSTM(432,activation='sigmoid',dropout_W = 0.5,dropout_U = 0.5))
            model.add(Dropout(0.1))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    #model.compile(optimizer="adam", loss='mse')
    #model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size)
    #model.save('model.h5')
    return model

## Load data from CSV file
def loadcsv(name,discard_zero):
    csvfile = open(name, 'rt')
    #Load all lines from csv file and closes it
    lines = csvfile.readlines()    
    csvfile.close()
    X = []
    y = []
    X_out = []
    y_out = []
    for i in range(len(lines)):        
        row = lines[i].split(',')
        #print (row)
        cimg = row[center]       
        #limg = row[left]           
        #rimg = row[right]        
        # Input range: 0 - 254
	# Center 127
        steer = float(row[steering])
        #Discard any steering below 5 degrees
        if discard_zero and abs(steer) < 1e-2:
            continue
        X.append(cimg)        
        y.append(steer)
        #X.append(limg)
        # Augmentation to use left image - subtract an steering threshold  
        #y.append(steer - angleTS)
        #X.append(rimg)
        # Augmentation to use right image - sum an steering threshold
        #y.append(steer + angleTS)
    #Shuffle the data before splitting
    X_train, y_train = shuffle(X, y)
    #Return data with 20% split for training
    return train_test_split(X_train, y_train,test_size=0.20)


files = ['afternoon_clear.csv','afternoon_clear.csv','afternoon_clear.csv','afternoon_clear.csv','afternoon_clear.csv']
#Hyper parameters for each training
discard = [False,False,False,False,False]
batch_size = [64,64,64,64,64]
epochs = [1,1,1,1,1]
input_size=  (64, 64, 3);

model = get_nvidia_model(input_size);
model.compile('adam', 'mse')
#model.load_weights('model_new_4.h5')
filename = 'nvidia_{}' 
for i in range(len(files)):
    X_train, X_valid, y_train, y_valid = loadcsv(files[i],discard[i])
    history = model.fit_generator(getData(X_train,y_train,input_size,batch_size[i],True), nb_epoch=epochs[i],verbose=1,samples_per_epoch=len(X_train),validation_data = getData(X_valid,y_valid,input_size,batch_size[i],False),nb_val_samples = len(X_valid))
    
    with open(filename.format(i)+'.json', 'w') as f:
        f.write(model.to_json())
        model.save_weights(filename.format(i)+'.h5')

    
