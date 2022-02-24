
# importing required libraries

import os
import sys
import time
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression
from gtts import gTTS
from tempfile import NamedTemporaryFile
from playsound import playsound
from operator import itemgetter
import pandas as pd 
from math import floor 
# Loading Pre-trained Models
def load_models():
    
    print("[INFO] loading EAST text detector...")
    east_model = cv2.dnn.readNet('./model/frozen_east_text_detection.pb')
    print("[INFO] loading CRNN text Recognition...")
    crnn_model = cv2.dnn.readNet("./model/crnn.onnx")
    print("[INFO] Models Loaded successfully")
    return east_model, crnn_model

def East_Image_Generator(img_path,width,height):
    """
    read and preprocessing image for east

    Args:
        img_path (str): [image path]
        width (float): [resized image width (should be multiple of 32) ]
        height (float): [resized image height (should be multiple of 32)"]

    Returns:
        [tuple]: [blob:preprocessed image for east'
                   h_ratio,
                   w_ratio]
    """
    img = cv2.imread(img_path)
    if img is None:
        print("[ERROR] Please inter a correct image path.")
        sys.exit()
        
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    h_ratio, w_ratio = h/height, w/width
    blob = cv2.dnn.blobFromImage(img,1,(width,height),(123.68,116.78,103.94),True,False)
    return img,blob,h_ratio,w_ratio

def East_Model(model,blob):
    """
    implementation for east model 

    Returns:
        [tuple]: [score:
                  geometry:]
    """
    model.setInput(blob)
    (geometry, score) = model.forward(model.getUnconnectedOutLayersNames())
    return geometry,score

def decode_predictions(score, geometry):
    rectangles = []
    confidence_scores = []
    for i in range(geometry.shape[2]):
        for j in range(geometry.shape[3]):
            if score[0][0][i][j] < 0.1 :
                continue
            bottom_x = int(j*4 + geometry[0][1][i][j])
            bottom_y = int(i*4 + geometry[0][2][i][j])
            
            top_x = int(j*4 - geometry[0][3][i][j])
            top_y = int(i*4 - geometry[0][0][i][j])
            rectangles.append((top_x,top_y,bottom_x,bottom_y))
            confidence_scores.append(float(score[0][0][i][j]))
    return rectangles,confidence_scores

def reshape_boxes(boxes,rW,rH):
    """   scale the bounding box coordinates based on the respective ratios
    """
    reshaped_boxes=[]

    for (startX, startY, endX, endY) in boxes:   
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        reshaped_boxes.append((startX, startY, endX, endY))
    
    return reshaped_boxes

def show_image_with_boxes(boxes,img):
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(img,(startX,startY),(endX,endY),(0,225,225),2)
    fig = plt.figure(figsize=(20,20))
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.savefig('./data/result1.jpg')
    

def most_likely(scores, char_set):
    text = ""
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        text += char_set[c]
    return text


def map_rule(text):
    char_list = []
    for i in range(len(text)):
        if i == 0:
            if text[i] != '-':
                char_list.append(text[i])
        else:
            if text[i] != '-' and (not (text[i] == text[i - 1])):
                char_list.append(text[i])
    return ''.join(char_list)


def best_path(scores, char_set):
    text = most_likely(scores, char_set)
    final_text = map_rule(text)
    return final_text


def CRNN_MODEL(model,img,boxes):
    alphabet_set = "0123456789abcdefghijklmnopqrstuvwxyz."
    blank = '-'
    char_set = blank + alphabet_set
    
    out_text = []
    
    img_copy = img.copy()
    img_preprocessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_,img_preprocessed = cv2.threshold(img_preprocessed,0.0,225.0,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img_preprocessed = cv2.medianBlur(img_preprocessed , 3)
    f = open('./data/out4.txt','w')

    for (startX, startY, endX, endY) in boxes:
        
        blob = cv2.dnn.blobFromImage(img_preprocessed[startY:endY, startX:endX], scalefactor=1/127.5, size=(100,32), mean=127.5)
        model.setInput(blob)
        scores = model.forward()
        out = best_path(scores, char_set)
        l=[out,startX, startY]
        f.write(" ".join(map(str,l)))
        f.write("\n")
        out_text.append(l)
        cv2.putText(img_copy, out, (startX, startY), 0, .7, (0, 0, 255), 2 )
    f.close()
    return out_text,img_copy

        
def speak(text, lang='en'):
    tempWavFile = NamedTemporaryFile(suffix='mp3')
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(tempWavFile) 
    tts.save("./out.mp3")
    #playsound(tempWavFile.name)
    tempWavFile.close()


def main():
    #construct parser to Detect and recognition text from the input image 
    parser = argparse.ArgumentParser(description="Detecting and recognition text from the image")

    parser.add_argument("-i", "--image", type=str, required=True,
                        help="path to the image")

    parser.add_argument("-w", "--width", type=int, default=736,
        help="resized image width (should be multiple of 32)")

    parser.add_argument("-H", "--height", type=int, default=416,
        help="resized image height (should be multiple of 32)")

    parser.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability required to inspect a region")

    args = vars(parser.parse_args())
    
    
    east_model, crnn_model = load_models()
    starttime = time.time()
    img,blob,rH,rW=East_Image_Generator(img_path=args['image'],width=args['width'],height=args['height'])
    geometry,score=East_Model(east_model,blob)
    rectangles,confidence_scores=decode_predictions(score, geometry)
    boxes = non_max_suppression(np.array(rectangles),probs=confidence_scores,overlapThresh=args['confidence'])
    reshaped_boxes=reshape_boxes(boxes,rW,rH)
    show_image_with_boxes(reshaped_boxes,img.copy())
    out_text,img_copy=CRNN_MODEL(crnn_model,img,reshaped_boxes)
    endtime = time.time()
    print(endtime-starttime)
    fig = plt.figure(figsize=(20,20))
    plt.imshow(img_copy)
    plt.axis('off')
    plt.savefig('./data/result2.jpg')  
    #L1=sorted(out_text, key=itemgetter(1),reverse=False)
    df= pd.DataFrame(out_text)
    df.iloc[:,1:]=df.iloc[:,1:]/100
    df.iloc[:,1]=df.iloc[:,1].apply(lambda x : floor(x))
    df.iloc[:,2]=df.iloc[:,2].apply(lambda x : floor(x))
    text = " ".join(list(df.sort_values([2,1])[0]))   
    print(text)   
    speak(text)
    #print(out_text)

if __name__ == '__main__':
    main()