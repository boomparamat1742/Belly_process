import torch
import cv2
import contrib
import numpy as np
import easygui
# from rembg import remove
# from PIL import Image
import random
import threading
import os
from importlib.resources import path
from turtle import width
from skimage import data, io
from rembg import remove
from PIL import Image
import time
import datetime
import json
import base64


import math


from screeninfo import get_monitors
width_screen = get_monitors()[0].width
height_screen = get_monitors()[0].height
# print(f"w = {width_screen}     h = {height_screen}")


def resizee(input):
    return cv2.resize(input, (720, 480), interpolation=cv2.INTER_AREA)


def saveImg(saveFrame, nameText):
    global name_count, CREATEFOLDERNAME, SAVEPATH
    final_path = os.path.join(SAVEPATH,     os.path.join(
        CREATEFOLDERNAME, f"{name_count}_{nameText}.jpg"))
    cv2.imwrite(final_path, saveFrame)
    # print(f"{name_count} : {final_path}")
    name_count += 1


def persen(frame):
    global name_count

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((6, 6), np.uint8)

    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([190, 255, 255])
    # mask = cv2.dilate(hsv_frame, low, high )
    mask = cv2.inRange(hsv_frame, low, high)
    saveImg(mask, "mask")
    
    result = cv2.bitwise_and(frame, frame, mask=mask, )
    img_erosion = cv2.erode(result, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion,kernel, iterations=1)
    opening = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN , kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE , kernel)
    saveImg(closing, "closing")

    
    
    # img = cv2.addWeighted(result,0.5,hsv_frame,0.5,10)
    # saveImg(img,"addwe")

    # cv2.imshow("result",result)
    # gray2=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    # result2 = cv2.bitwise_and(img, frame, mask=mask, )
    # opening = cv2.morphologyEx(result2, cv2.MORPH_OPEN , kernel)
    # dilation = cv2.dilate(result2,kernel,iterations= 1)
    # img_erosion = cv2.erode(result2, kernel, iterations=1)
    
    # gray_blur=cv2.GaussianBlur(gray2,(15,15),0)

    gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GrayMeat1", gray)
    meatPixel1 = cv2.countNonZero(gray)
    print("Total1", meatPixel1)

    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GrayMeat2", gray2)
    meatPixel2 = cv2.countNonZero(gray2)
    print("Total2", meatPixel2)
    
    img = cv2.addWeighted(closing, 0.3, frame, 0.5, 10)
    saveImg(img, "img")


    results = meatPixel1/meatPixel2*100
    print("results", results)

    persenMeat = results
    persenFat = 100 - results

    print("------------------------")
    print(f"meat = {round(persenMeat)}")
    print(f"fat = {round(persenFat)}")

    return {
        "meat": float("{:.2f}".format(persenMeat)),
        "fat": float("{:.2f}".format(persenFat)),
        "over":(img)

    }

    


def removeBG(cv2Image):
    global name_count

    img = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    imgOutput = remove(im_pil)

    opencvImage = cv2.cvtColor(np.array(imgOutput), cv2.COLOR_RGB2BGR)

    return opencvImage


def seg(imgOutput):
    global name_count

    imgOutput2 = imgOutput.copy()
    grayImage = cv2.cvtColor(imgOutput2, cv2.COLOR_BGR2GRAY)
    
    
    
    height, width = imgOutput.shape[:2]
    mask = np.zeros(imgOutput.shape[:2], np.uint8)
    backgroundmodel = np.zeros((1, 65), np.float64)
    #cv2.imshow("back",backgroundmodel)
    forgroundmodel = np.zeros((1, 65), np.float64)
    #cv2.imshow("forg",forgroundmodel)
    rect = (50,50,width-40,height-40)
    #rect = (47,79,2253,589)
    # rect(x1,y1,x2,y2)

    cv2.grabCut(imgOutput, mask, rect, backgroundmodel,
                forgroundmodel, 5, cv2.GC_INIT_WITH_RECT)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image_1 = imgOutput*mask[:, :, np.newaxis]
    image_2 = mask[:, :, np.newaxis]
    # cv2.imshow("test", image_1)

    background = imgOutput - image_1

    background[np.where((background > [255, 255, 255]).all(axis=2))] = [
        255, 255, 255]
    final = background = image_1
    saveImg(final, "final_bw")
    
    hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    lower_blue1 = np.array([90, 5, 40])
    upper_blue1 = np.array([130, 255, 255])
    mask_blue1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    blue1 = cv2.bitwise_and(imgOutput, imgOutput, mask=mask_blue1)


    lower_yellow1 = np.array([20, 40, 40])
    upper_yellow1 = np.array([40, 255, 255])
    mask_yellow1 = cv2.inRange(hsv,lower_yellow1, upper_yellow1)
    yellow1 = cv2.bitwise_and(imgOutput, imgOutput, mask=mask_yellow1)
	#cv2.imshow("yellow",results)

    lower_gray = np.array([120, 0.4, 130])
    upper_gray = np.array([135, 255, 255])
    mask_gray = cv2.inRange(hsv,lower_yellow1, upper_yellow1)


    mask2 = cv2.bitwise_or(mask_blue1,mask_yellow1)
    mask3 = cv2.bitwise_not(mask2,final)
    #cv2.imshow("mask",mask2)
    result = cv2.bitwise_and(final, final, mask=mask3)


    gray1 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(gray1, cv2.COLOR_RGB2BGR)

    (thresh, blackAndWhiteImage) = cv2.threshold(
        final, 20, 255, cv2.THRESH_BINARY)
    saveImg(blackAndWhiteImage, "mask_bw")

    return gray2


global isSave
isSave = True


def timer_save():
    global isSave
    isSave = True
    print(f"isSave = True")


# model = torch.hub.load('ultralytics/yolov5', 'custom', 'best_1.pt')
name_count = 1

SAVEPATH = os.path.join(os.getcwd(), "sm")


def one_imgage():
    global CREATEFOLDERNAME,SAVEPATH,img
    # CREATEFOLDERNAME = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pathh = easygui.fileopenbox()
    #pathh = cv2.resize(paths, (2123, 551), interpolation=cv2.INTER_AREA)

    CREATEFOLDERNAME = os.path.split(pathh)[1].split('.')[0]

    savepathfolder = os.path.join(SAVEPATH, CREATEFOLDERNAME)

    os.chdir(SAVEPATH)
    os.system(f"mkdir {CREATEFOLDERNAME}")
    time.sleep(1)

    imgInput = cv2.imread(pathh)
    saveImg(imgInput, "imgInput")
    
    out_frame = seg(removeBG(imgInput))
    obj = persen(out_frame)
    
    

    p_meat = obj['meat']
    p_fat = obj['fat']
    overs = obj['over']

    if p_meat >= 60:
        grade = 'A'
    elif p_meat >= 50:
        grade = 'B'
    elif p_meat >= 40:
        grade = 'C'
    else:
        grade = 'D'

   

    text_str = f"meat = {'{:.2f}'.format(p_meat)}%   fat = {'{:.2f}'.format(p_fat)}%  grade {grade}"
    out_frame = cv2.putText(overs, text_str, (400, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    saveImg(out_frame , "out_frame")
    
    # cv2.imshow("output", out_frame)
    
    
    values = (f'{p_meat} / {p_fat}')
            
    resizeds = cv2.resize(out_frame, (829,588),interpolation = cv2.INTER_AREA)
    retval, buffers = cv2.imencode('.jpg',resizeds)
    jpg_as_text = base64.b64encode(buffers)
           
    base64_strings = base64.b64encode(jpg_as_text).decode('utf-8')
    data_uri = 'data:image/jpg;base64,' + base64_strings
            
    dictionaryData = {
        "persenMeat":  p_meat,
        "persenFat": p_fat,
        "values": values,
        "grade": grade,
        "image": data_uri 
            
        }
            
    json_object = json.dumps(dictionaryData, indent=4)

    with open(os.path.join(savepathfolder, "_data.json"), "w") as outfile:
        outfile.write(json_object)
    
    out_frame = resizee(out_frame)

    

    print("----------------------------------------------------------------------------------------------------")
    cv2.waitKey(0)


def folder_image():
    global CREATEFOLDERNAME, name_count, SAVEPATH
    folder_path = easygui.diropenbox()
    
	
    for element in os.listdir(folder_path):

        use_path = os.path.join(folder_path, element)
        name_count = 1
        

        CREATEFOLDERNAME = element.split('.')[0]

        savepathfolder = os.path.join(SAVEPATH, CREATEFOLDERNAME)
        os.mkdir(savepathfolder)

        time.sleep(1)

        imgInput = cv2.imread(use_path)
        saveImg(imgInput, "imgInput")
        #imgInput = cv2.resize(imgInputs, (2123, 551), interpolation=cv2.INTER_AREA)

        out_frame = seg(imgInput )
        obj = persen(out_frame)

        p_meat = obj['meat']
        p_fat = obj['fat']
        overs = obj['over']

        if p_meat >= 60:
            grade = 'A'
        elif p_meat >= 50:
            grade = 'B'
        elif p_meat >= 40:
            grade = 'C'
        else:
            grade = 'D'

        dictionaryData = {
            "persenMeat": p_meat,
            "persenFat": p_fat,
            "grade": grade
        }

        json_object = json.dumps(dictionaryData, indent=4)

        with open(os.path.join(savepathfolder, "_data.json"), "w") as outfile:
            outfile.write(json_object)

        text_str = f"meat = {'{:.2f}'.format(p_meat)}%   fat = {'{:.2f}'.format(p_fat)}%  grade {grade}"
        out_frame = cv2.putText(overs, text_str, (400, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        saveImg(out_frame, "out_frame")
        out_frame = resizee(out_frame)
    
    values = (f'{p_meat} / {p_fat}')
            
    resizeds = cv2.resize(out_frame, (829,588),interpolation = cv2.INTER_AREA)
    retval, buffers = cv2.imencode('.jpg',resizeds)
    jpg_as_text = base64.b64encode(buffers)
           
    base64_strings = base64.b64encode(jpg_as_text).decode('utf-8')
    data_uri = 'data:image/jpg;base64,' + base64_strings
            
    dictionaryData = {
        "persenMeat":  p_meat,
        "persenFat": p_fat,
        "values": values,
        "grade": grade,
        "image": data_uri 
            
        }
            
    json_object = json.dumps(dictionaryData, indent=4)

    with open(os.path.join(savepathfolder, "_data.json"), "w") as outfile:
        outfile.write(json_object)

        

        print("----------------------------------------------------------------------------------------------------")


select = input("1 - file, 2 - folder : ")
if select == '1':
    one_imgage()
elif select == '2':
    folder_image()
