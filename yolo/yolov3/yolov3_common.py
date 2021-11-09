import numpy as np
import cv2


def getLabels():
    # Create labels into list
    with open('cfg/coco.names') as f:
        labels = [line.strip() for line in f]
        # Initialize colours for representing every detected object
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        # Loading trained YOLO v3 Objects Detector
        # with the help of 'dnn' library from OpenCV
        # Reads a network model stored in Darknet model files.
        network = cv2.dnn.readNetFromDarknet('cfg/yolov3.cfg', 'cfg/yolov3.weights')
        # Getting only output layer names that we need from YOLO
        ln = network.getLayerNames()
        ln = [ln[i - 1] for i in network.getUnconnectedOutLayers()]
        print(ln)
        return ln, network, colours, labels


def performForward(network, blob, ln, p_min, h, w):
    network.setInput(blob)
    output_from_network = network.forward(ln)

    # Preparing lists for detected bounding boxes, confidences and class numbers.
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > p_min:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
    return bounding_boxes, confidences, class_numbers

def nonMaximumSuppression(bounding_boxes, confidences, p_min, threshold, colours, class_numbers, labels, image):
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, p_min, threshold)
    # At-least one detection should exists
    if len(results) > 0:
        for i in results.flatten():
            # Getting current bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            colour_box_current = colours[class_numbers[i]].tolist()

            # Drawing bounding box on the original image
            cv2.rectangle(image, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(image, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)