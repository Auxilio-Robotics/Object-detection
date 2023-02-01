## Alfred Perception: Object Detection


This repository contains code that uses the ultralyitcs package to run a Yolov8 model on a ROS node. It subscribes to a camera feed, and publishes the classes as a numpy array.



### Progress
- ROS Node set up.
- Will currently be using class 39 (bottle) for integration testing. We will progress to a custom dataset with a particular set of object classes.

### Future Work

- Constantly run Yolov8 and ground 2D detections to 3D map environment for remembering object locations.



