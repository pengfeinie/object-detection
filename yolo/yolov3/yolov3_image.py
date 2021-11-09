import cv2
import yolov3_common


# initialize minimum probability to eliminate weak predictions
p_min = 0.5
# threshold when applying non-max suppression
threshold = 0.3

ln, network, colours, labels = yolov3_common.getLabels()

# Read image with opencv, we get image in BGR
image = cv2.imread('images/901.jpg')
# Slicing and get height, width of the image
h, w = image.shape[:2]

# image preprocessing for deep learning
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
print(image.shape)
print(blob.shape)

# perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities.
bounding_boxes, confidences, class_numbers = yolov3_common.performForward(network, blob, ln, p_min, h, w)
print(bounding_boxes)
print(confidences)
print(class_numbers)

# Implementing non-maximum suppression of given bounding boxes
# With this technique we exclude some of bounding boxes if their
# corresponding confidences are low or there is another
# bounding box for this region with higher confidence
yolov3_common.nonMaximumSuppression(bounding_boxes, confidences, p_min, threshold, colours,
                                    class_numbers, labels, image)

# WINDOW_NORMAL gives window as resizable.
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image)
cv2.waitKey(0)
