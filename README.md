# Object Detection
Object detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage [machine learning](https://www.mathworks.com/discovery/machine-learning.html) or [deep learning](https://www.mathworks.com/discovery/deep-learning.html) to produce meaningful results. When humans look at images or video, we can recognize and locate objects of interest within a matter of moments. The goal of object detection is to replicate this intelligence using a computer.

## Why Object Detection Matters

Object detection is a key technology behind advanced driver assistance systems (ADAS) that enable cars to detect driving lanes or perform pedestrian detection to improve road safety. Using object detection to identify and locate vehicles.

![Using object detection to identify and locate vehicles](https://pengfeinie.github.io/images/1630396980057.jpg)

Some time ago, I was exploring the exciting world of **convolutional neural networks** and wondered how can we use them for **image classification.**  Beside simple image classification, there’s no shortage of fascinating problems in computer vision, with **object detection**being one of the most interesting. Most commonly it’s associated with self driving cars where systems blend computer vision, LIDAR and other technologies to generate a multidimensional representation of road with all its participants. On the other hand object detection is used in video surveillance.

Ok, so what exactly is object detection? To answer that question let’s start with image classification. In this task we’ve got an image and we want to assign it to one of many different categories (e.g. car, dog, cat, human,…), so basically we want to answer the question **“What is in this picture?”**. Note that one image has only one category assigned to it. After completing this task we do something more difficult and try to locate our object in the image, so our question changes to **“What is it and where it is?”**. This task is called **object localization**. So far so good, but in a real-life scenario, we won’t be interested in locating only one object but rather multiple objects in one image. For example let’s think of a **self-driving car**, that in the real-time video stream has to find the location of other cars, traffic lights, signs, humans and then having this information take appropriate action. It’s a great example of **object detection**. In object detection tasks we are interested in finding all object in the image and drawing so-called **bounding boxes** around them. There are also some situations where we want to find exact boundaries of our objects in the process called **instance segmentation**, but this is a topic for another post.

![img](https://pengfeinie.github.io/images/types.png)

![2021-11-07_144109.png](https://pengfeinie.github.io/images/2021-11-07_144109.png)



## Two-Stage Networks

The initial stage of two-stage networks, such as [R-CNN and its variants](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html), identifies *region proposals*, or subsets of the image that might contain an object. The second stage classifies the objects within the region proposals. Two-stage networks can achieve very accurate object detection results; however, they are typically slower than single-stage networks. [getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html)

### R-CNN

[R-CNN](https://arxiv.org/abs/1703.06870 ) The R-CNN detector [[2\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_a9cdd2b3-b910-4d3d-90db-b485b415fd9b) first generates region proposals using an algorithm such as Edge Boxes[[1\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_cfbce1ef-74b2-46dd-814f-a0f985c96301). The proposal regions are cropped out of the image and resized. Then, the CNN classifies the cropped and resized regions. Finally, the region proposal bounding boxes are refined by a support vector machine (SVM) that is trained using CNN features.

Use the [`trainRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/trainrcnnobjectdetector.html) function to train an R-CNN object detector. The function returns an [`rcnnObjectDetector`](https://www.mathworks.com/help/vision/ref/rcnnobjectdetector.html) object that detects objects in an image.

![img](https://pengfeinie.github.io/images/rcnn.png)

### Fast R-CNN

[Fast R-CNN](https://arxiv.org/abs/1504.08083)  As in the R-CNN detector , the Fast R-CNN[[3\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_7805faa1-3821-45ff-8e7d-bda6ae35a633) detector also uses an algorithm like Edge Boxes to generate region proposals. Unlike the R-CNN detector, which crops and resizes region proposals, the Fast R-CNN detector processes the entire image. Whereas an R-CNN detector must classify each region, Fast R-CNN pools CNN features corresponding to each region proposal. Fast R-CNN is more efficient than R-CNN, because in the Fast R-CNN detector, the computations for overlapping regions are shared.

Use the [`trainFastRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/trainfastrcnnobjectdetector.html) function to train a Fast R-CNN object detector. The function returns a [`fastRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/fastrcnnobjectdetector.html) that detects objects from an image.

![img](https://pengfeinie.github.io/images/fast.png)

### Faster R-CNN

[Faster R-CNN](https://arxiv.org/abs/1506.01497) The Faster R-CNN[[4\]](https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html#mw_25d18973-df6c-48ef-aaa9-31a4ec9e6705) detector adds a region proposal network (RPN) to generate region proposals directly in the network instead of using an external algorithm like Edge Boxes. The RPN uses [Anchor Boxes for Object Detection](https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html). Generating region proposals in the network is faster and better tuned to your data.

Use the [`trainFasterRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html) function to train a Faster R-CNN object detector. The function returns a [`fasterRCNNObjectDetector`](https://www.mathworks.com/help/vision/ref/fasterrcnnobjectdetector.html) that detects objects from an image.

![img](https://pengfeinie.github.io/images/faster.png)



## One-Stage Networks

### YOLO(You Only Look Once)

There are a few different algorithms for object detection and they can be split into two groups:

1. Algorithms based on classification – they work in two stages. In the first step, we’re selecting from the image interesting regions. Then we’re classifying those regions using convolutional neural networks. This solution could be very slow because we have to run prediction for every selected region. Most known example of this type of algorithms is the Region-based convolutional neural network (RCNN) and their cousins Fast-RCNN and Faster-RCNN.

2. Algorithms based on regression – instead of selecting interesting parts of an image, we’re predicting classes and bounding boxes for the whole image **in one run of the algorithm**. Most known example of this type of algorithms is **YOLO (You only look once)** commonly used for real-time object detection.

   

   Before we go into YOLOs details we have to know what we are going to predict. Our task is to predict a class of an object and the bounding box specifying object location. Each bounding box can be described using four descriptors:

   1. center of a bounding box (**bx**,**by**)
   2. width (**bw**)
   3. height (**bh**)
   4. value **c** is corresponding to a class of an object (f.e. car, traffic lights,…).

   We’ve got also one more predicted value pc which is a probability that there is an object in the bounding box, I will explain in a moment why do we need this.

   ![img](https://pengfeinie.github.io/images/bbox-1.png)

YOLO uses a single bounding box regression to predict the height, width, center, and class of objects. In the image above, represents the probability of an object appearing in the bounding box.



https://scholar.google.com/

![image-20211109124400546](https://pengfeinie.github.io/images/image-20211109124400546.png)

Before diving into YOLO, we need to go through some terms:

**1-Intersect Over Union (IOU):**

![img](https://pengfeinie.github.io/images/iou.png)

*the above image from [source](https://amrokamal-47691.medium.com/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899) .*

IOU can be computed as Area of Intersection divided over Area of Union of two boxes, so IOU must be ≥0 and ≤1.

When predicting bounding boxes, we need the find the IOU between the predicted bounding box and the ground truth box to be ~1.

![img](https://pengfeinie.github.io/images/cat.jpeg)

In the left image, IOU is very low, but in the right image, IOU is ~1.

**2-** **Precision:**

Simply we can define precision as the ratio of **true** positive(true predictions) (TP) and the total number of **predicted** positives(total predictions). The formula is given as such:

![img](https://miro.medium.com/max/60/1*-Y2bLNfGDkcZ5BSPPzHBjQ.png?q=20)

![img](https://miro.medium.com/max/180/1*-Y2bLNfGDkcZ5BSPPzHBjQ.png)

For example, imagine we have 20 images, and we know that there are 120 cars in these 20 images.

Now, let’s suppose we input these images into a model, and it detected 100 cars (here the model said: I’ve found 100 cars in these 20 images, and I’ve drawn bounding boxes around every single car of them).

To calculate the precision of this model, we need to check the 100 boxes the model had drawn, and if we found that 20 of them are incorrect, then the precision will be =80/100=0.8

**3-Recall:**

If we look at the precision example again, we find that it doesn’t consider the total number of cars in the data (120), so if there are 1000 cars instead of 120 and the model output 100 boxes with 80 of them are correct, then the precision will be 0.8 again.

To solve this, we need to define another metric, called the **Recall,** which is the ratio of **true** positive(true predictions) and the total of ground truth positives(total number of cars). The formula is given as such:

![img](https://miro.medium.com/max/60/1*nx6V3Q_EqnGWzcLfW_lL-A.png?q=20)

![img](https://miro.medium.com/max/198/1*nx6V3Q_EqnGWzcLfW_lL-A.png)

For our example, the recall=80/120=0.667.

Now we can notice that the recall measures how well we detect **all** the objects in the data.

![img](https://miro.medium.com/max/350/1*kaqtNALKZujx1FGlbK11OQ.png)

**4- Average Precision and Mean Average Precision(mAP):**

A brief definition for the Average Precision is the **area** under the **precision-recall curve.**

**AP** combines both precision and recall together. It takes a value between 0 and 1 (higher is better). To get **AP** =1 we need both the precision and recall to be equal to 1. The **mAP** is the mean of the AP calculated for all the classes.

#### YOLO v1

##### The Architecture

![image-20211109160501058](https://pengfeinie.github.io/images/image-20211109160501058.png)

**The Model**. Our system models detection as a regression problem. It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes, and C class probabilities. These predictions are encoded as an S × S × (B ∗ 5 + C) tensor. For evaluating YOLO on [PASCAL VOC](https://paperswithcode.com/dataset/pascal-voc), we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. Our final prediction is a 7 × 7 × 30 tensor. We only predict one set of class probabilities per grid cell, regardless of the number of boxes B. 

*The PASCAL Visual Object Classes (VOC) 2012 dataset contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person. Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation, and classification tasks. The PASCAL VOC dataset is split into three subsets: 1,464 images for training, 1,449 images for validation and a private testing set.*

The architecture of YOLO v1 is not complicated, in fact it's just a convolutional backbone with two fully connected layers, much like an image classification network architecture. The clever part of YOLO (the part that makes it an object detector) is in the interpretation of the outputs of those fully connected layers. However, the concepts underlying that interpretation are complex, and can be difficult to grasp on a first reading. As I started writing about YOLO, I found myself repeating lots of ideas that I've already written about in other posts, and I don't want to repeat all that here. Therefore I'm going to refer you to those posts that explain in much more depth the concepts that I think are relevant.

Firstly, I recommend that you understand what anchors are, which I explain in depth in my post on [Faster R-CNN](https://www.harrysprojects.com/articles/fasterrcnn.html). YOLO doesn't technically use anchors, but it uses a similar idea. If you understand anchors then the discussion of the representation presented shortly will feel familiar. Note that there are differences between what YOLO does and the anchors defined in the RPN, so I'll make sure to be clear on those differences at the end. Secondly, it would benefit you to understand what box parameterisation is, in which I go into detail in my post on [Fast R-CNN](https://www.harrysprojects.com/articles/fastrcnn.html). You don't need to understand the specifics of how Fast R-CNN does parameterisation because YOLO does it differently, but it is important that you understand that when an object detector predicts a bounding box, you must always ask, *with respect to what?* If you are familiar with those concepts, then let's continue.

Next, a quick note on the backbone. The authors designed their own convolutional backbone which was inspired by GoogLeNet. But I just want to point that it's just a feature extractor, and you could swap in any backbone you like, and as long as you made the size of the fully connected layers line up, it would all work fine. I won't dwell on the backbone any longer, the object detection is all done in the head. See my post on [Fast R-CNN](https://www.harrysprojects.com/articles/fastrcnn.html) for more detail on the difference between network backbones and heads.

As I said earlier, the network architecture is very simple, it's just a backbone with two fully connected layers. Let's blow up that last layer in a bit more detail. I'm going to refer to it as the *output tensor* to make it easier to refer to.

![img](https://pengfeinie.github.io/images/output_tensor.png)

The first thing you might notice is that I've been calling it a fully connected layer, but it sure doesn't look like one. Don't let the 3D shape fool you, it *is* fully connected, it is *not* produced by a convolution, they just reshape it because it's easier to interpret in 3D. If implemented in PyTorch, you can imagine it being coded as a fully connected layer that is then reshaped into a 3D tensor. Alternatively, you can imagine unrolling the 3D tensor into one long vector of length `1470 (7x7x30)`. However you imagine it, it is fully connected, every output neuron is connected to every neuron in the 4096-vector before it.

So why reshape it into 3D? What do all those outputs mean? Why do those outputs make it an object detector? I'll start with the reason that it's `7x7`. To clarify my notation and make it easier to talk about, I will refer to a *cell*, and what I mean by that is a single position in the `7x7` grid of the output tensor. Therefore each cell is a single vector of length 30, I have highlighted one such cell in the diagram.

YOLO breaks the image up into a grid of size `7x7`. Let me copy in the image from the paper. For the moment, focus on the left part of the diagram, with the `SxS` grid.The images as below come from [YOLO v1 Paper](https://arxiv.org/abs/1506.02640).

![image-20211109125322361](https://pengfeinie.github.io/images/image-20211109125322361.png)

These cells represent prior boxes so that when the network predicts box coordinates, it has something to reference them from. Remember that earlier I said whenever you predict boxes, you have to say *with respect to what?* Well it's with respect to these grid cells. More concretely, the network can detect objects by predicting scales and offsets from those prior boxes. As an illustrative example, take the prior box on the second row down, second in from the right. It's centered on the car, so it seems reasonable that this prior box should be responsible for predicting the car. To predict the best box possible, the network should output values that scaled this box out horizontally so that it fit the car better, just the like pink box in the final detections on the right of the diagram. The `7x7` grid isn't actually drawn on the image, it's just implied, and the thing that implies it is the `7x7` grid in the output tensor. You can imagine overlaying the output tensor on the image, and each cell corresponds to a part of the image. If you understand anchors, this idea should feel famililar to you.

So each cell is responsible for predicting boxes from a single part of the image. More specifically, each cell is responsible for predicting precisely two boxes for each part of the image. Note that there are 49 cells, and each cell is predicting two boxes, so the whole network is only going to predict 98 boxes. That number is fixed.

In order to predict a single box, the network must output a number of things. Firstly it must encode the coordinates of the box which YOLO encodes as `(x, y, w, h)`, where `x` and `y` are the center of the box. Early I suggested you familiarise yourself with box parameterisation, because YOLO does output the actual coordinates of the box. Firstly, the width and height are normalised with respect to the image width, so if the network outputs a value of `1.0` for the width, it's saying the box should span the entire image, likewise `0.5` means it's half the width of the image. Note that the width and height have nothing to do with the actual grid cell itself. The `x and y` values *are* parameterised with respect to the grid cell, they represent offsets from the grid cell position. The grid cell has a width and height which is equal to `1/S` (we've normalised the image to have width and height 1.0). If the network outputs a value of `1.0` for `x`, then it's saying that the `x` value of the box is the `x` position of the grid cell plus the width of the grid cell.

Secondly, YOLO also predicts a confidence score for each box which represents the probability that the box contains an object. Lastly, YOLO predicts a class, which is represented by a vector of `C` values, and the predicted class is the one with the highest value. Now, here's the catch. YOLO does *not* predict a class for every box, it predicts a class *for each cell*. But each cell is associated with two boxes, so those boxes will have the same predicted class, even though they may have different shapes and positions. Let's tie all that together visually, let me copy down my diagram again.

![img](https://pengfeinie.github.io/images/output_tensor.png)

The first five values encode the location and confidence of the first box, the next five encode the location and confidence of the next box, and the final 20 encode the 20 classes (because Pascal VOC has 20 classes). In total, the size of the vector is `5xB + C` where `B` is the number of boxes, and `C` is the number of classes.







For every grid cell, you will get two bounding boxes, which will make up for the starting 10 values of the 1*30 tensor. The remaining 20 denote the number of classes. The values denote the class score, which is the conditional probability of object belongs to class i, if an object is present in the box.

![](https://pengfeinie.github.io/images/yolov1_grid1.jpg)

![](https://pengfeinie.github.io/images/yolo1.gif)

Next, we multiply all these class score with bounding box confidence and get class scores for different bounding boxes. 

![](https://pengfeinie.github.io/images/yolo1_grid.gif)

We do this for all the grid cells. That is equal to 7 x 7 x 2 = 98.

![](https://pengfeinie.github.io/images/yolo1_all_grid.gif)



The Yolo was one of the first deep, one-stage detectors and since the first paper was published in **CVPR 2016**, each year has brought with it a new Yolo paper or tech report. We begin with Yolo v1 [1], but since we are primarily interested in analyzing loss functions, all we really need to know about the Yolo v1 CNN **(Figure 2a)**, is that is takes an RGB image (**448×448×3**) and returns a cube (**7×7×30**), interpreted in **(Figure 2b)**.

![](https://pengfeinie.github.io/images/00adc0adec6423a45a0706a4ce2dc01d.png)

#### YOLO v3

Download: https://pjreddie.com/media/files/yolov3.weights and move to under cfg folder.







### SSD (Single Shot Multibox Detector)

The [SSD architecture](https://arxiv.org/pdf/1512.02325.pdf) was published in 2016 by researchers from Google. It presents an object detection model using a single deep neural network combining regional proposals and feature extraction.

A set of default boxes over different aspect ratios and scales is used and applied to the feature maps. As these feature maps are computed by passing an image through an image classification network, the feature extraction for the bounding boxes can be extracted in a single step. Scores are generated for each object category in every of the default bounding boxes. In order to better fit the ground truth boxes adjustment offsets are calculated for each box.

![img](https://pengfeinie.github.io/images/1_DLdhpsy1CfhSp00AJNa4kg_uxhgv1.png)

Different feature maps in the convolutional network correspond with different receptive fields and are used to naturally handle objects at different scales . As all the computation is encapsulated in a single network and fairly high computational speeds are achieved (for example, for 300 × 300 input 59 FPS).



## How It Works

### Object Detection Using Deep Learning

You can use a variety of techniques to perform object detection. Popular deep learning–based approaches using [convolutional neural networks](https://www.mathworks.com/discovery/convolutional-neural-network-matlab.html) (CNNs), such as R-CNN and YOLO v2, automatically learn to detect objects within images.

You can choose from two key approaches to get started with object detection using deep learning:

- **Create and train a custom object detector.** To train a custom object detector from scratch, you need to design a network architecture to learn the features for the objects of interest. You also need to compile a very large set of labeled data to train the CNN. The results of a custom object detector can be remarkable. That said, you need to manually set up the layers and weights in the CNN, which requires a lot of time and training data.
- **Use a pretrained object detector.** Many object detection workflows using deep learning leverage [transfer learning](https://blogs.mathworks.com/pick/2017/02/24/deep-learning-transfer-learning-in-10-lines-of-matlab-code/), an approach that enables you to start with a pretrained network and then fine-tune it for your application. This method can provide faster results because the object detectors have already been trained on thousands, or even millions, of images. Detecting a stop sign using a pretrained R-CNN. 

![Detecting a stop sign using a pretrained R-CNN.](https://pengfeinie.github.io/images/1630396980251.jpg)

### Object Detection Using Machine Learning

Machine learning techniques are also commonly used for object detection, and they offer different approaches than deep learning. Common machine learning techniques include:

- Aggregate channel features (ACF)
- SVM classification using histograms of oriented gradient (HOG) features
- The Viola-Jones algorithm for human face or upper body detection

Similar to deep learning–based approaches, you can choose to start with a pretrained object detector or create a custom object detector to suit your application. You will need to manually select the identifying features for an object when using machine learning, compared with automatic feature selection in a deep learning–based workflow.

## References

- [https://www.mathworks.com/discovery/object-detection.html](https://www.mathworks.com/discovery/object-detection.html)
- [https://paperswithcode.com/task/object-detection](https://paperswithcode.com/task/object-detection)
- [https://www.datacamp.com/community/tutorials/object-detection-guide](https://www.datacamp.com/community/tutorials/object-detection-guide)
- [https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html](https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html)
- [https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/](https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/)
- [Object Detection in 20 Years: A Survey](https://www.semanticscholar.org/paper/Object-Detection-in-20-Years%3A-A-Survey-Zou-Shi/bd040c9f76d3b0b77e2065089b8d344c9b5d83d6#extracted)  
- [https://arxiv.org/pdf/1905.05055.pdf](https://arxiv.org/pdf/1905.05055.pdf)
- [https://pengfeinie.github.io/files/1905.05055.pdf](https://pengfeinie.github.io/files/1905.05055.pdf) 
- [https://link.springer.com/article/10.1007/s11263-019-01247-4](https://link.springer.com/article/10.1007/s11263-019-01247-4)
- [https://machinelearningmastery.com/object-recognition-with-deep-learning/](https://machinelearningmastery.com/object-recognition-with-deep-learning/)
- [https://viso.ai/deep-learning/object-detection/](https://viso.ai/deep-learning/object-detection/)
- https://paperswithcode.com/dataset/pascal-voc
- https://www.harrysprojects.com/articles/yolov1.html
- https://medium.com/oracledevs/final-layers-and-loss-functions-of-single-stage-detectors-part-1-4abbfa9aa71c
- https://amrokamal-47691.medium.com/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899
- https://blog.csdn.net/hrsstudy/article/details/70305791?spm=1001.2014.3001.5501
