import cv2, numpy as np
from augmentation import *
from random import random

def processImgTrain(img,angle):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #if random() > 0.8:
    #    img, angle = trans_image(img,angle,100)
    #if random() > 0.3:
    #    img = augment_brightness_camera_images(img)
    #if random() > 0.5:
    #    img = add_random_shadow(img)
    if random() > 0.5:
        img = cv2.flip(img,1)
        angle = -angle
    return processImg(img),angle

def processImg(img):
    #Crop top half
    h, w,l = img.shape
    img = img[int(h/2):,:,:]        
    yuv=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)  
   
    channels= cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))
    channels[0] = clahe.apply(channels[0])

    
   
    img=cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB)  
    
    mean = np.mean(img)    
    img = img - mean
    s = np.std(img)
    img = img/s
    img =  cv2.resize(img,(160,40), interpolation = cv2.INTER_AREA)
    #cv2.imshow('test',img)
    #cv2.waitKey(1)
    return img


def getData(data,angle,input_size,batch_size,train):
    '''    Generator for Batch loading the dataset. Receives the images path, their relative angle output, and how many images should be loaded each time
    '''
    #Creates index to be sampled based on size of the datased
    index = np.arange(len(data))
    while 1:
        #Randomly select images up to batch size
        #Allocates memory space for the images to be loaded
        batch_train = np.zeros([batch_size]+ list(input_size), dtype = np.float32)
        batch_angle = np.zeros((batch_size,), dtype = np.float32)   
        for i in range(batch_size):
            try:
                #select one random index at a time, without replacement
                random = int(np.random.choice(index,1,replace = False))
                index = np.delete(index,index==random)
            except :
                #If there are no indexes left, reset index array
                index = np.arange(len(data))    
                #cut batch size to selected items
                batch_train = batch_train[:i-1,:,:]
                batch_angle = batch_angle[:i-1]
                break                        
            if(train):
                batch_train[i],batch_angle[i]= processImgTrain(cv2.imread(data[random]), angle[random])
                #print(batch_angle[i])
            else:
                batch_train[i] = processImg(cv2.imread(data[random]))
                batch_angle[i] = angle[random]
        yield (batch_train, batch_angle)