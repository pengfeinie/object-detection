import cv2
import yolov3_common

# initialize minimum probability to eliminate weak predictions
p_min = 0.5
# threshold when applying non-maxia suppression
threshold = 0.

# 'VideoCapture' object and reading video from a file
video = cv2.VideoCapture('videos/2.mp4')

# Preparing variable for writer that we will use to write processed frames
writer = None

# Preparing variables for spatial dimensions of the frames
h, w = None, None

ln, network, colours, labels = yolov3_common.getLabels()

# Defining loop for catching frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Getting dimensions of the frame for once as everytime dimensions will be same
    if w is None or h is None:
        # Slicing and get height, width of the image
        h, w = frame.shape[:2]

    # frame preprocessing for deep learning
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # perform a forward pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities.
    bounding_boxes, confidences, class_numbers = yolov3_common.performForward(network, blob, ln, p_min, h, w)
    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence
    yolov3_common.nonMaximumSuppression(bounding_boxes, confidences, p_min, threshold, colours, class_numbers, labels, frame)
    """Store processed frames into result video."""
    # Initialize writer
    if writer is None:
        resultVideo = cv2.VideoWriter_fourcc(*'mp4v')
        # Writing current processed frame into the video file
        writer = cv2.VideoWriter('videos/result2.mp4', resultVideo, 30, (frame.shape[1], frame.shape[0]), True)
    # Write processed current frame to the file
    writer.write(frame)

# Releasing video reader and writer
video.release()
writer.release()
