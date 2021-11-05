# object-detection
Object detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage [machine learning](https://www.mathworks.com/discovery/machine-learning.html) or [deep learning](https://www.mathworks.com/discovery/deep-learning.html) to produce meaningful results. When humans look at images or video, we can recognize and locate objects of interest within a matter of moments. The goal of object detection is to replicate this intelligence using a computer.

## Why Object Detection Matters

Object detection is a key technology behind advanced driver assistance systems (ADAS) that enable cars to detect driving lanes or perform pedestrian detection to improve road safety. Object detection is also useful in applications such as video surveillance or image retrieval systems. Using object detection to identify and locate vehicles.

![Using object detection to identify and locate vehicles](https://www.mathworks.com/discovery/object-detection/_jcr_content/mainParsys3/discoverysubsection/mainParsys3/image.adapt.full.medium.jpg/1630396980057.jpg)



Some time ago, I was exploring the exciting world of **convolutional neural networks** and wondered how can we use them for **image classification.**  Beside simple image classification, there’s no shortage of fascinating problems in computer vision, with **object detection**being one of the most interesting. Most commonly it’s associated with self driving cars where systems blend computer vision, LIDAR and other technologies to generate a multidimensional representation of road with all its participants. On the other hand object detection is used in video surveillance.

Ok, so what exactly is object detection? To answer that question let’s start with image classification. In this task we’ve got an image and we want to assign it to one of many different categories (e.g. car, dog, cat, human,…), so basically we want to answer the question **“What is in this picture?”**. Note that one image has only one category assigned to it. After completing this task we do something more difficult and try to locate our object in the image, so our question changes to **“What is it and where it is?”**. This task is called **object localization**. So far so good, but in a real-life scenario, we won’t be interested in locating only one object but rather multiple objects in one image. For example let’s think of a **self-driving car**, that in the real-time video stream has to find the location of other cars, traffic lights, signs, humans and then having this information take appropriate action. It’s a great example of **object detection**. In object detection tasks we are interested in finding all object in the image and drawing so-called **bounding boxes** around them. There are also some situations where we want to find exact boundaries of our objects in the process called **instance segmentation**, but this is a topic for another post.

![img](https://appsilondatascience.com/assets/uploads/2018/08/types.png)

### YOLO algorithm

There are a few different algorithms for object detection and they can be split into two groups:

1. Algorithms based on classification – they work in two stages. In the first step, we’re selecting from the image interesting regions. Then we’re classifying those regions using convolutional neural networks. This solution could be very slow because we have to run prediction for every selected region. Most known example of this type of algorithms is the Region-based convolutional neural network (RCNN) and their cousins Fast-RCNN and Faster-RCNN.

2. Algorithms based on regression – instead of selecting interesting parts of an image, we’re predicting classes and bounding boxes for the whole image **in one run of the algorithm**. Most known example of this type of algorithms is **YOLO (You only look once)** commonly used for real-time object detection.

   

https://www.mathworks.com/discovery/object-detection.html

https://paperswithcode.com/task/object-detection
https://www.datacamp.com/community/tutorials/object-detection-guide
https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html
https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/
