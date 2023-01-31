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


class ObjectDetectionNode:

    def __init__(self) -> None:
        
        self.model = YOLO('yolov8n.pt')
        self.model.to('cuda:0')

    def runModel(self, img):
        results = self.model(img)
        boxes =  results[0].boxes
        box = boxes.xyxy
        box_cls =boxes.cls
        # for (b,cls) in zip(box,box_cls):
        #     img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
        #     img = cv2.putText(img, str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
        #                 1, (0,0,255), 1, cv2.LINE_AA)
        return box, box_cls


    def callback(self,ros_rgb_image):

        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
        boxes, classes = self.runModel(rgb_image)
        msg = Detections()
        msg.boxes = boxes
        msg.box_classes = classes
        self.pub.publish(msg)


    def main(self):
        rospy.init_node('object_detection', anonymous=False)
        self.pub = rospy.Publisher('object_bounding_boxes', Detections, queue_size=10)
        self.rgb_topic_name = '/camera/color/image_raw'
        self.rgb_image_subscriber = message_filters.Subscriber(self.rgb_topic_name, Image, self.callback)
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