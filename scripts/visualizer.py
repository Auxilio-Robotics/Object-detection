#!/usr/bin/env python

from ultralytics import YOLO
import torch
import time
import numpy as np
import cv2
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import String
from yolo.msg import Detections
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from rospy.numpy_msg import numpy_msg
import json



def callback(image,box_and_classes):
    rospy.loginfo(type(box_and_classes))
    try:
        msg = json.loads(box_and_classes.data)
        boxes = msg['boxes']
        box_classes = msg['box_classes']
        for (b,cls) in zip(boxes,box_classes):
            image = cv2.rectangle(image, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
            image = cv2.putText(image, str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255), 1, cv2.LINE_AA)

        pub.publish(cv_bridge.cv2_to_imgmsg(image))
    except Exception as e:
        print(e)
    # rospy.loginfo("Publishing")


def main():
    global pub, cv_bridge
    rospy.init_node('viz_object_detections', anonymous=False)
    rospy.loginfo("Node initialized")
    pub = rospy.Publisher('labelled_yolo', Image, queue_size=60)
    box_sub = message_filters.Subscriber('/object_bounding_boxes', String)
    rgb_topic_name = '/camera/color/image_raw'
    rgb_image_subscriber = message_filters.Subscriber(rgb_topic_name, Image, )
    ts = message_filters.TimeSynchronizer([rgb_image_subscriber, box_sub], 60)
    ts.registerCallback(callback)
    cv_bridge = CvBridge()

    
if __name__ == "__main__":
    main()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass