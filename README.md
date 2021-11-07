# Object Detection
Object detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage [machine learning](https://www.mathworks.com/discovery/machine-learning.html) or [deep learning](https://www.mathworks.com/discovery/deep-learning.html) to produce meaningful results. When humans look at images or video, we can recognize and locate objects of interest within a matter of moments. The goal of object detection is to replicate this intelligence using a computer.

## Why Object Detection Matters

Object detection is a key technology behind advanced driver assistance systems (ADAS) that enable cars to detect driving lanes or perform pedestrian detection to improve road safety. Using object detection to identify and locate vehicles.

![Using object detection to identify and locate vehicles](https://www.mathworks.com/discovery/object-detection/_jcr_content/mainParsys3/discoverysubsection/mainParsys3/image.adapt.full.medium.jpg/1630396980057.jpg)

![2021-11-07_144109.png](https://pengfeinie.github.io/images/2021-11-07_144109.png)

[Object Detection in 20 Years: A Survey](https://www.semanticscholar.org/paper/Object-Detection-in-20-Years%3A-A-Survey-Zou-Shi/bd040c9f76d3b0b77e2065089b8d344c9b5d83d6#extracted)

## Two-Stage Networks

The initial stage of two-stage networks, such as [R-CNN and its variants](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html), identifies *region proposals*, or subsets of the image that might contain an object. The second stage classifies the objects within the region proposals. Two-stage networks can achieve very accurate object detection results; however, they are typically slower than single-stage networks. [getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html)

### R-CNN

The R-CNN detector [[2\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_a9cdd2b3-b910-4d3d-90db-b485b415fd9b) first generates region proposals using an algorithm such as Edge Boxes[[1\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_cfbce1ef-74b2-46dd-814f-a0f985c96301). The proposal regions are cropped out of the image and resized. Then, the CNN classifies the cropped and resized regions. Finally, the region proposal bounding boxes are refined by a support vector machine (SVM) that is trained using CNN features.

Use the [`trainRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/trainrcnnobjectdetector.html) function to train an R-CNN object detector. The function returns an [`rcnnObjectDetector`](https://www.mathworks.com/help/vision/ref/rcnnobjectdetector.html) object that detects objects in an image.

![img](https://pengfei.nie.github.io/images/rcnn.png)

### Fast R-CNN

As in the R-CNN detector , the Fast R-CNN[[3\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_7805faa1-3821-45ff-8e7d-bda6ae35a633) detector also uses an algorithm like Edge Boxes to generate region proposals. Unlike the R-CNN detector, which crops and resizes region proposals, the Fast R-CNN detector processes the entire image. Whereas an R-CNN detector must classify each region, Fast R-CNN pools CNN features corresponding to each region proposal. Fast R-CNN is more efficient than R-CNN, because in the Fast R-CNN detector, the computations for overlapping regions are shared.

Use the [`trainFastRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/trainfastrcnnobjectdetector.html) function to train a Fast R-CNN object detector. The function returns a [`fastRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/fastrcnnobjectdetector.html) that detects objects from an image.

![img](https://pengfei.nie.github.io/images/fast.png)

### Faster R-CNN

The Faster R-CNN[[4\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_25d18973-df6c-48ef-aaa9-31a4ec9e6705) detector adds a region proposal network (RPN) to generate region proposals directly in the network instead of using an external algorithm like Edge Boxes. The RPN uses [Anchor Boxes for Object Detection](https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html). Generating region proposals in the network is faster and better tuned to your data.

Use the [`trainFasterRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html) function to train a Faster R-CNN object detector. The function returns a [`fasterRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/fasterrcnnobjectdetector.html) that detects objects from an image.

![img](https://pengfei.nie.github.io/images/faster.png)



## One-Stage Networks

















## How It Works

### Object Detection Using Deep Learning

You can use a variety of techniques to perform object detection. Popular deep learning–based approaches using [convolutional neural networks](https://www.mathworks.com/discovery/convolutional-neural-network-matlab.html) (CNNs), such as R-CNN and YOLO v2, automatically learn to detect objects within images.

You can choose from two key approaches to get started with object detection using deep learning:

- **Create and train a custom object detector.** To train a custom object detector from scratch, you need to design a network architecture to learn the features for the objects of interest. You also need to compile a very large set of labeled data to train the CNN. The results of a custom object detector can be remarkable. That said, you need to manually set up the layers and weights in the CNN, which requires a lot of time and training data.
- **Use a pretrained object detector.** Many object detection workflows using deep learning leverage [transfer learning](https://blogs.mathworks.com/pick/2017/02/24/deep-learning-transfer-learning-in-10-lines-of-matlab-code/), an approach that enables you to start with a pretrained network and then fine-tune it for your application. This method can provide faster results because the object detectors have already been trained on thousands, or even millions, of images.

![Detecting a stop sign using a pretrained R-CNN.](https://www.mathworks.com/discovery/object-detection/_jcr_content/mainParsys3/discoverysubsection_/mainParsys3/image_copy.adapt.full.medium.jpg/1630396980251.jpg)

Detecting a stop sign using a pretrained R-CNN. 

Some time ago, I was exploring the exciting world of **convolutional neural networks** and wondered how can we use them for **image classification.**  Beside simple image classification, there’s no shortage of fascinating problems in computer vision, with **object detection**being one of the most interesting. Most commonly it’s associated with self driving cars where systems blend computer vision, LIDAR and other technologies to generate a multidimensional representation of road with all its participants. On the other hand object detection is used in video surveillance.

Ok, so what exactly is object detection? To answer that question let’s start with image classification. In this task we’ve got an image and we want to assign it to one of many different categories (e.g. car, dog, cat, human,…), so basically we want to answer the question **“What is in this picture?”**. Note that one image has only one category assigned to it. After completing this task we do something more difficult and try to locate our object in the image, so our question changes to **“What is it and where it is?”**. This task is called **object localization**. So far so good, but in a real-life scenario, we won’t be interested in locating only one object but rather multiple objects in one image. For example let’s think of a **self-driving car**, that in the real-time video stream has to find the location of other cars, traffic lights, signs, humans and then having this information take appropriate action. It’s a great example of **object detection**. In object detection tasks we are interested in finding all object in the image and drawing so-called **bounding boxes** around them. There are also some situations where we want to find exact boundaries of our objects in the process called **instance segmentation**, but this is a topic for another post.

![img](https://appsilondatascience.com/assets/uploads/2018/08/types.png)

### YOLO algorithm

There are a few different algorithms for object detection and they can be split into two groups:

1. Algorithms based on classification – they work in two stages. In the first step, we’re selecting from the image interesting regions. Then we’re classifying those regions using convolutional neural networks. This solution could be very slow because we have to run prediction for every selected region. Most known example of this type of algorithms is the Region-based convolutional neural network (RCNN) and their cousins Fast-RCNN and Faster-RCNN.

2. Algorithms based on regression – instead of selecting interesting parts of an image, we’re predicting classes and bounding boxes for the whole image **in one run of the algorithm**. Most known example of this type of algorithms is **YOLO (You only look once)** commonly used for real-time object detection.

   
   
   Before we go into YOLOs details we have to know what we are going to predict. Our task is to predict a class of an object and the bounding box specifying object location. Each bounding box can be described using four descriptors:
   
   1. center of a bounding box (**b****x****b****y**)
   2. width (**b****w**)
   3. height (**b****h**)
   4. value **c** is corresponding to a class of an object (f.e. car, traffic lights,…).
   
   We’ve got also one more predicted value pc which is a probability that there is an object in the bounding box, I will explain in a moment why do we need this.
   
   ![img](https://appsilondatascience.com/assets/uploads/2018/08/bbox-1.png)
   
   
   
   

https://www.mathworks.com/discovery/object-detection.html

https://paperswithcode.com/task/object-detection
https://www.datacamp.com/community/tutorials/object-detection-guide
https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html
https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/
