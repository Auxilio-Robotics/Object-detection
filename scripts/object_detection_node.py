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
import base64
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ObjectDetectionNode:

    def __init__(self) -> None:
        
        self.model = YOLO('yolov8n.pt')
        self.model.to('cuda:0')
        rospy.loginfo("Loaded Model")
        self.visualize = True
        self.i = 0

    def runModel(self, img):
        results = self.model(img)
        boxes =  results[0].boxes
        box = boxes.xyxy
        box_cls =boxes.cls
        if self.visualize:
            for (b,cls) in zip(box,box_cls):
                img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
                img = cv2.putText(img, str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0,0,255), 1, cv2.LINE_AA)
                # cv2.imwrite(f"outputs/{self.i}.png", img)

        return box, box_cls, img


    def callback(self,ros_rgb_image):
        
        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
        rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        boxes, classes, annotated_img = self.runModel(rgb_image)
        msg = {
            'boxes' : boxes.cpu().numpy(),
            'box_classes' : classes.cpu().numpy(),
        }
        
        self.data_pub.publish(json.dumps(msg, cls = NumpyEncoder))
        self.annotated_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(annotated_img))


    def main(self):
        rospy.init_node('object_detection', anonymous=False)
        rospy.loginfo("Node initialized")
        self.data_pub = rospy.Publisher('object_bounding_boxes', String, queue_size=1)
        self.annotated_image_pub = rospy.Publisher('annotated_image_body', Image, queue_size=1)
        self.cameraInfoSub = message_filters.Subscriber('/camera/depth/camera_info', Image)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image, )
        cache = message_filters.Cache(self.rgb_image_subscriber, 1)
        cache.registerCallback(self.callback)
        rospy.loginfo("Node Ready...")
        self.cv_bridge = CvBridge()

    
if __name__ == "__main__":
    
    node = ObjectDetectionNode()
    node.main()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass