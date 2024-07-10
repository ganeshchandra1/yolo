#code for image2vect and imageFinderCode is present here

import argparse
import sys
import os
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image    
import facenet 
import yolo
import math
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import glob
from PIL import Image,ImageColor


CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
  
def __init__(self, yolo_model = None, facenet_model = None):
        if yolo_model==None:
            self._yolo_model=yolo.load_model()
        else:
            self._yolo_model = yolo_model

        if facenet_model==None:
            self._facenet_model=facenet.load_model()
        else:
            self._facenet_model = facenet_model
# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def post_process(frame, outs, conf_threshold, nms_threshold,args):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        draw_predict(frame, confidences[i], left, top, right, bottom)
        crop_image(frame,left,top,right,bottom,args)
    return final_boxes

def crop_image(frame,left,top,right,bottom,args):
    im_array=[]
    facenet_model=facenet.load_model()
    cropped_image = frame[left:left+bottom, right:right+top]
    plt.imshow(cropped_image)
    im = Image.fromarray(frame)
    im = im.crop((left, top, right, bottom)).resize((160,160))
    im.save("cropped.jpeg")  
    im.show()
    im=np.array(list(im.getdata())).reshape((160,160,3))
    im_array.append(im)
    images_array = np.array(im_array, dtype="float32")
    images_array /=255.0
    images_array -=0.5 
    imageV = facenet_model.predict(images_array)
    imageVector = normalize(imageV, norm='l2')
    calculate_euclidean(imageVector)
    output_file = args.image[:-4].rsplit('/')[-1] + '_yolofacecrop.jpg'
    cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))


def calculate_euclidean(imageV):
    yolo_model=yolo.load_model()
    facenet_model=facenet.load_model()
    imgs_array=[]
    distances=[]
    image_list = []
    for filename in glob.glob('C:\\Users\\cgane\\vs\\yolofacenet\\dataset\\*.jpg'):
        im=Image.open(filename)
        image_list.append(im)
    image_list=image_list[:30]
    for img in image_list:
        image, out_scores, out_boxes, out_classes = yolo_model.detect_image(img)
        im_array=[]

        for out_box in out_boxes:
            top, left, bottom, right = out_box
            croppedImage = image.crop((left, top, right, bottom)).resize((160,160)) 
            croppedImage_array = np.array(list(croppedImage.getdata())).reshape((160,160,3))
            im_array.append(croppedImage_array)
            break
        im_array = np.array(im_array, dtype="float32")
        im_array /=255.0
        im_array -=0.5 

        #Feed cropped images into facenet
        imageVector = facenet_model.predict(im_array)
        #print(imageVector)
        

        #Normalize the vector
        imageVector = normalize(imageVector, norm='l2')
        distance = calculate_minimum_euclidean_distances(imageVector,imageV)
        distances.append(distance)
    dic=create_grp_ids(distances,image_list)
    display_similar(dic,1)
    
        
def calculate_minimum_euclidean_distances(img,imageV):
    distance = euclidean_distances(img.reshape((1,128)), imageV.reshape((1,128)))
    return distance
    #print("distance")
    #print(distance)
    
def create_grp_ids(distances,image_list):
    d={}
    for i in range(len(distances)):
        l=[]
        if i<30:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(1)
        if i>=30 and i<60:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(2) 
        if i>=60 and i<90:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(3)
        if i>=90 and i<120:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(4) 
        if i>=120 and i<150:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(5)
        if i>=150 and i<180:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(6) 
        if i>=180 and i<210:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(7)
        if i>=210 and i<240:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(8) 
        if i>=240 and i<270:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(9)
        if i>=270 and i<300:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(10) 
        if i>=300 and i<330:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(11)
        if i>=330 and i<360:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(12) 
        if i>=360 and i<390:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(13)
        if i>=390 and i<420:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(14) 
        if i>=420 and i<450:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(15)
        if i>=450 and i<480:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(16) 
        if i>=480 and i<510:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(17)
        if i>=510 and i<540:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(18) 
        if i>=540 and i<570:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(19)
        if i>=570 and i<600:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(20) 
        if i>=600 and i<630:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(21)
        if i>=630 and i<660:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(22) 
        if i>=660 and i<690:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(23)
        if i>=690 and i<720:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(24) 
        if i>=720 and i<750:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(25)
        if i>=750 and i<780:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(26) 
        if i>=780 and i<810:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(27)
        if i>=810 and i<840:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(28) 
        if i>=840 and i<870:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(29)
        if i>=870 and i<900:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(30) 
        if i>=900 and i<930:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(31)
        if i>=930 and i<960:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(32) 
        if i>=960 and i<990:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(33)
        if i>=990 and i<1020:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(34) 
        if i>=1020 and i<1050:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(35)
        if i>=1050 and i<1080:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(36) 
        if i>=1080 and i<1110:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(37)
        if i>=1110 and i<1140:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(38) 
        if i>=1140 and i<1170:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(39)
        if i>=1170 and i<1200:
            l.append(distances[i][0][0])
            l.append(image_list[i])
            l.append(40) 

        d[i]=l

    return d
def display_similar(d,id):
    lim=[]
    group = int(input(" Enter group:1-40 that picture belongs to "))
    tp=tn=fp=fn=0
    print(d)
    for key,value in d.items():
        if int(value[2])==group and value[0]<1.2:
            tp=tp+1
            lim.append(d[key][1])
        elif int(value[2]!=group) and value[0]<1.2:
            fp=fp+1
        elif int(value[2]==group) and not value[0]<1.2:
            fn=fn+1
        elif int(value[2]!=group) and not value[0]<1.2:
            tn=tn+1
    if tp+fp == 0:
        prec= 0
    else:
        prec= tp / (tp + fp)
    if tp+fn == 0:
        recall= 0  
    else:
        recall=tp / (tp + fn)
    print(tp,tn,fp,fn)
    print("Precision", prec)
    print("Recall" , recall)
    get_detected_images_as_one(lim)
            
def get_detected_images_as_one(lim, size_per_image = (160,160)):
        """
        Get a compliation of detected images as one
        """
        images=[]
        for i in lim:
            images.append(i.resize(size_per_image))

        widths = (math.ceil(math.sqrt(len(images))))*size_per_image[0]
        heights = (math.ceil(math.sqrt(len(images))))*size_per_image[1]
        new_im = Image.new('RGB', (widths, heights))

        x_offset = 0
        y_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,y_offset))

            x_offset = x_offset+size_per_image[0]
            if x_offset>=widths:
                x_offset=0
                y_offset=y_offset+size_per_image[1]
        new_im.save("your_file.jpeg")  
        new_im.show()
            
            
        
    
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom
