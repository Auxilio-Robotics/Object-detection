from ultralytics import YOLO
import torch
import time
import numpy as np
import cv2

import rospy
from std_msgs.msg import String

def runModel(model, img):
    # Load a model
    
    # img = cv2.imread('/home/praveen/catkin_ws/src/yolo/simple_action_servers/bottles.jpeg')

    # img = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR) 
    results = model(img)
    boxes =  results[0].boxes
    box = boxes.xyxy
    box_cls =boxes.cls
    # for (b,cls) in zip(box,box_cls):
    #     img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
    #     img = cv2.putText(img, str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
    #                 1, (0,0,255), 1, cv2.LINE_AA)
    return box, box_cls



def yolo():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('yolo', anonymous=False)
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        runModel(model, img)
        # rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

    
if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    model.to('cuda:0')

    try:
        yolo()
    except rospy.ROSInterruptException:
        pass