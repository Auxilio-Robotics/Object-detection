#!/usr/bin/env python

from sklearn.cluster import KMeans
from ultralytics import YOLO
import torch
import time
import numpy as np
import cv2
from sensor_msgs.msg import Image
import rospy
import scipy
from std_msgs.msg import String
from yolo.msg import Detections
import binascii
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from rospy.numpy_msg import numpy_msg
import json
import base64
import sklearn
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ObjectDetectionNode:

    def __init__(self) -> None:
        self.i = 0
        self.annotations = None

    def depthImgCallback(self,ros_rgb_image):
        
        depth_img = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'passthrough')
        self.depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
        loc = np.where(np.array(self.annotations['box_classes']).astype(np.uint8) == 39)[0]
        if self.annotations is not None and len(loc) > 0:
            loc = loc[0]
            x1, y1, x2, y2 = self.annotations['boxes'][loc]
            # TODO : Replace all these hard stuff with rosparams.
            dilationFactors = [10, 10]
            yLow = max(0, int(y1) - dilationFactors[1])
            yHigh = min(420, int(y2) + dilationFactors[1])
            
            xLow = max(0, int(x1) - dilationFactors[0])
            xHigh = min(324, int(x2) + dilationFactors[1])
            # cropped = self.depth_img[yLow:yHigh, xLow:xHigh]
            cropped = self.depth_img[yLow:yHigh, xLow:xHigh]#np.expand_dims(.astype(np.float32), axis = 2)
            clt = KMeans(n_clusters=3, n_init = 'auto').fit(cropped.reshape(-1,1))
            
            centers = clt.cluster_centers_.flatten()
            threshcenters = centers[centers > 20]
            if(len(threshcenters) > 0):
                print(sorted(threshcenters))   
            xc = (xLow + xHigh) / 2
            yc = (yLow + yHigh) / 2
            self.data_pub.publish(json.dumps({'x' : xc, 'y' : yc})) 


    def getAnnotations(self, msg):
        self.annotations = json.loads(msg.data)
        

    def main(self):
        rospy.init_node('depth_parsing', anonymous=False)
        self.data_pub = rospy.Publisher('object_centroids', String, queue_size=1)
        rospy.Subscriber("/object_bounding_boxes", String, self.getAnnotations)
        self.rgb_topic_name = '/camera/aligned_depth_to_color/image_raw'
        self.rgb_image_subscriber = message_filters.Subscriber(self.rgb_topic_name, Image, )
        cache = message_filters.Cache(self.rgb_image_subscriber, 1)
        cache.registerCallback(self.depthImgCallback)
        rospy.loginfo("Node Ready...")
        self.cv_bridge = CvBridge()

    
if __name__ == "__main__":
    
    node = ObjectDetectionNode()
    node.main()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass