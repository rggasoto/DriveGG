import argparse
import base64
import json

import numpy as np
import socket
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import tensorflow as tf
tf.python.control_flow_ops = tf
import _thread
import time
from math import pi
from TrainingData import processImg
char = None
try:
    from msvcrt import getch  # try to import Windows version
except ImportError:
    def getch():   # define non-Windows version
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
 
char = 0
# Fix error with Keras and TensorFlow

def keypress():
    global char
    while 1:
        char = getch()
        if char == b'\x03':
            break 
 

_thread.start_new_thread(keypress, ())


#sio = socketio.Server()
#app = Flask(__name__)
model = None
prev_image_array = None
kp = 64
desired_speed = 10.
steering_std = 0.5
#@sio.on('telemetry')
def telemetry(data):
    global desired_speed
    global steering_std
    global char
    ## The current steering angle of the car
    #steering_angle = data["steering_angle"]
    ## The current throttle of the car
    #throttle = data["throttle"]
    ## The current speed of the car
    data = data.split(b';')
    speed = float(data[0])
   
    e = desired_speed - speed
    # The current image from the center camera of the car
    imgString = data[1]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #Image comes as RGB, convert to BGR as is standard for openCV
    img = cv2.cvtColor(image_array,cv2.COLOR_RGBA2RGB )
    #r = image_array[:,:,0]
    #image_array[:,:,0] = image_array[:,:,2]
    #image_array[:,:,2] = r    
    #cv2.imshow('test',image_array)
    #cv2.waitKey(1)
    #h, w,l = img.shape
    #print(w,h,l)
   #img = cv2.resize(image_array,(int(w / 2),int(h / 2)), interpolation = cv2.INTER_AREA)
   # h, w,l = img.shape
    #img = img[int(h/2):,:,:]        
    img = processImg(img)
    cv2.imshow('test',img)
    cv2.waitKey(1)
    
    
    #std = np.std(newimg)
    #newimg = newimg/std
    transformed_image_array = img[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle=127
    steering_angle_orig = float(model.predict(transformed_image_array, batch_size=1))
    # This is for GTA inputs only
    steering_angle = steering_angle_orig/pi*180/35.0*127 + 127# /127.0*35.0)/180.0*Math.PI  
    #Proportional controller for throttle
    # (0 - 512) with 256 as 0
    throttle = 256+ kp*e
    
    #DEBUG: force 12.5 degrees steering angle
    if char == b'a':
        steering_angle = 0
    if char ==   b'd':
        steering_angle = 255
    #cut throttle for vehicle
    if char ==   b' ':
        throttle = 0
    #Increase / Decrease vehicle desired speed
    if char ==   b'w':
        desired_speed+=0.1
    if char ==   b's':
        desired_speed-=0.1
    print(char,steering_angle_orig, steering_angle,throttle,speed,desired_speed)
    #input()
    send_control(steering_angle, throttle)




def send_control(steering_angle, throttle):
    #print("sending",throttle,steering_angle)
    b = '{};{};{};<EOT>'.format(throttle,steering_angle,'0')
    #print(b)
    s_control.send(b.encode('utf8'))
    #sio.emit("steer", data={
    #'steering_angle': steering_angle.__str__(),
    #'throttle': throttle.__str__()
    #}, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
         #NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    img = cv2.imread('start.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB )
    #r = image_array[:,:,0]
    #image_array[:,:,0] = image_array[:,:,2]
    #image_array[:,:,2] = r    
    #cv2.imshow('test',image_array)
    #cv2.waitKey(1)
    #h, w,l = img.shape
    #print(w,h,l)
   #img = cv2.resize(image_array,(int(w / 2),int(h / 2)), interpolation = cv2.INTER_AREA)
   # h, w,l = img.shape
    #img = img[int(h/2):,:,:]        
    img = processImg(img)
    #cv2.imshow('test',img)
    #cv2.waitKey(1)
    
    
    #std = np.std(newimg)
    #newimg = newimg/std
    transformed_image_array = img[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    steering_angle = steering_angle/pi*180/35.0*127 + 127# /127.0*35.0)/180.0*Math.PI  
    #print(steering_angle)
    host = 'localhost' 
    control_port = 804;
    image_port = 809;
    s_control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_control.connect((host,control_port));
    send_control(128,255);
    data = b''
    while 1:
        print('connecting')
        s_image = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_image.bind((host,image_port));
        s_image.listen(1);
        conn, c_a = s_image.accept()
        print('connected, receiving data')
        while 1:
            try:
               
                data += conn.recv(4096)
                #print('data received')
                b = data.find(b';<EOM>')
                if b>0:
                    #print('full image received')
                    subdata = data[:b]
                    data = data[b+6:]
                    telemetry(subdata)
            except(a):
                print(a)
                break
    # wrap Flask application with engineio's middleware
    #app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    #eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
