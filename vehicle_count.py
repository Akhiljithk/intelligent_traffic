# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *
import requests
import imutils
import warnings
import pyfirmata
import time

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('video.mp4')
input_size = 320

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

# image_file = 'test1.jpeg'
cam = cv2.VideoCapture(0)
result, image = cam.read()
cv2.imwrite("GeeksForGeeks.png", image)
image_file = 'GeeksForGeeks.png'

def from_static_image(image):
    img = cv2.imread(image)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)

    # Find the objects from the network output
    try:
        postProcess(outputs,img)
        isVechicle = True
    except:
        totalCount = 0
        isVechicle = False
    # count the frequency of detected classes
    if isVechicle:
        frequency = collections.Counter(detected_classNames)
        # print(frequency)
        # Draw counting texts in the frame
        cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    
        carCout = str(frequency['car']) 
        motorbikeCout = str(frequency['motorbike'])
        busCount = str(frequency['bus'])
        truckCount = str(frequency['truck'])
    
        totalCount = int(carCout)+int(motorbikeCout)+int(busCount)+int(truckCount)

    # cv2.imshow("image", img)

    # cv2.waitKey(0)

    # save the data to a csv file
    # with open("static-data.csv", 'a') as f1:
    #     cwriter = csv.writer(f1)
    #     cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
    # f1.close()

    return totalCount

def getImageFromCam(camUrl, cam1Url):
    headers = {
    'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
    }
    img_resp = requests.get(camUrl, headers=headers, verify=False)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    # cv2.imshow("Android_cam", img)
    # cv2.waitKey()
    cv2.imwrite(cam1Url, img)
    image_file = cam1Url
    totalCount = from_static_image(image_file) 
    return totalCount

if __name__ == '__main__':  

    comport='COM12'

    board=pyfirmata.Arduino(comport)

    g1=board.get_pin('d:2:o')
    r1=board.get_pin('d:3:o')
    y1=board.get_pin('d:4:o')

    g2=board.get_pin('d:5:o')
    r2=board.get_pin('d:6:o')
    y2=board.get_pin('d:7:o')

    g3=board.get_pin('d:8:o')
    r3=board.get_pin('d:9:o')
    y3=board.get_pin('d:10:o')

    g4=board.get_pin('d:11:o')
    r4=board.get_pin('d:12:o')
    y4=board.get_pin('d:13:o')

    cam1 = 'https://192.168.202.41:8080/shot.jpg'
    cam1Url = './cam1/shotOnCam1.png'

    # countFromCam1 = getImageFromCam(cam1, cam1Url)

    # print("countFromCam1 : ", countFromCam1)

    while True:
        countFromCam1 = getImageFromCam(cam1, cam1Url)
        print("detected vechicles ", countFromCam1)
        g1.write(1)
        r1.write(1)
        y1.write(1)
        
        g2.write(1)
        r2.write(1)
        y2.write(1)

        g3.write(1)
        r3.write(1)
        y3.write(1)

        g4.write(1)
        r4.write(1)
        y4.write(1)

        # if countFromCam1>0 :
        #     r.write(1)
        #     y.write(0)
        #     g.write(0)
        # else:
        #     r.write(0)
        #     y.write(0)
        #     g.write(1)

        # countFromCam1 = getImageFromCam(cam1, cam1Url)
        # print("detected vechicles ", countFromCam1)
        # led_1.write(1)
        # led_2.write(0)
        # led_3.write(0)
        # time.sleep(1)
        # countFromCam1 = getImageFromCam(cam1, cam1Url)
        # print("detected vechicles ", countFromCam1)
        # led_1.write(0)
        # led_2.write(1)
        # led_3.write(0)
        # time.sleep(1)
        # countFromCam1 = getImageFromCam(cam1, cam1Url)
        # print("detected vechicles ", countFromCam1)
        # led_1.write(0)
        # led_2.write(0)
        # led_3.write(1)
        # time.sleep(1)

        
    