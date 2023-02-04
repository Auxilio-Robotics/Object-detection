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
        self.i = 0

    def runModel(self, img):
        results = self.model(img)
        boxes =  results[0].boxes
        box = boxes.xyxy
        box_cls =boxes.cls
        # if self.i % 20 == 0:
        #     for (b,cls) in zip(box,box_cls):
        #         img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
        #         img = cv2.putText(img, str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
        #                     1, (0,0,255), 1, cv2.LINE_AA)
            
        #     cv2.imwrite(f"outputs/{self.i}.png", img)
        # self.i+= 1
        return box, box_cls


    def callback(self,ros_rgb_image):
        
        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
        rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        # rgb_image = cv2.resize(rgb_image, (360, 640))
        # rgb_image = cv2.copyMakeBorder(rgb_image, 80, 80, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
        boxes, classes = self.runModel(rgb_image)
        msg = {
            'boxes' : boxes.cpu().numpy(),
            'box_classes' : classes.cpu().numpy(),
        }
        
        # msg = Detections()
        # msg.boxes = list(map(int, list(boxes.cpu().numpy().astype(int))))
        # msg.box_classes = list(map(int, list(classes.cpu().numpy().astype(int))))
        self.pub.publish(json.dumps(msg, cls=NumpyEncoder))
        # rospy.loginfo("Publishing")


    def main(self):
        rospy.init_node('object_detection', anonymous=False)
        rospy.loginfo("Node initialized")
        self.pub = rospy.Publisher('object_bounding_boxes', String, queue_size=60)
        self.rgb_topic_name = '/camera/color/image_raw'
        self.rgb_image_subscriber = message_filters.Subscriber(self.rgb_topic_name, Image, )
        cache = message_filters.Cache(self.rgb_image_subscriber, 10)
        cache.registerCallback(self.callback)
        rospy.loginfo("Node Ready...")
        # synchronizer = message_filters.TimeSynchronizer([rgb_image_subscriber, self.depth_image_subscriber, self.camera_info_subscriber], 10)
        # synchronizer.registerCallback(self.image_callback)
        self.cv_bridge = CvBridge()

    
if __name__ == "__main__":
    
    node = ObjectDetectionNode()
    node.main()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass