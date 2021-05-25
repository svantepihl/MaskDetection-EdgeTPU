from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from pycoral.utils.dataset import read_label_file

import argparse
import cv2
import os


def load_image(path):
    image = cv2.imread(path)
    return image

def detect_and_classify_faces(detector,classifier,image,threshold,padding = 10):
    predictions = []
    boxes = []
    faces = []
    height, width, _ = image.shape
    detector_target_size = common.input_size(detector)
    classifier_target_size = common.input_size(classifier)

    scale_x, scale_y = width / detector_target_size[0], height / detector_target_size[1]
    resized_image = cv2.resize(image,detector_target_size)
    run_inference(detector,resized_image.tobytes())
    objects = detect.get_objects(detector,threshold)

    for object in objects:
        bbox = object.bbox.scale(scale_x,scale_y)
        startX, startY = int(bbox.xmin-padding), int(bbox.ymin-padding)
        endX, endY = int(bbox.xmax+padding), int(bbox.ymax+padding)

        # ensure the bounding boxes fall within the dimensions of the image
        (startX, startY) = (max(1, startX), max(1, startY))
        (endX, endY) = (min(width - 1, endX), min(height - 1, endY))
        boxes.append((startX,startY,endX,endY))

        face = image[startY:endY,startX:endX]
        face = cv2.resize(face,classifier_target_size)
        faces.append(face)

    
    for face in faces:
        run_inference(classifier,face.tobytes())
        prediction = classify.get_scores(classifier)
        predictions.append(prediction)


    return (boxes,predictions)

def append_boxes_to_img(cv2_img, boxes, predictions):
    for (box, prediction) in zip(boxes, predictions):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (maskCorrect, maskOnChin, maskOnlyOnMouth, noMask) = prediction

        # determine the class label and color we'll use to draw
        # the bounding box and text
        if max(prediction) == maskCorrect:
            label = "Mask correct"
            color = (0, 255, 0)  # green
        elif max(prediction) == maskOnChin:
            label = "Mask on chin"
            color = (51, 153, 255)
        elif max(prediction) == maskOnlyOnMouth:
            label = "Mask does not cover nose"
            color = (255, 255, 0)
        elif max(prediction) == noMask:
            label = "No mask"
            color = (0, 0, 255)
        else:
            label = ""
            color = (255, 255, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(prediction) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(cv2_img, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(cv2_img, (startX, startY), (endX, endY), color, 2)
    return cv2_img

def main():
    default_model_dir = 'models'
    default_detection_model = 'facedetector_quant_postprocess_edgetpu.tflite'
    default_classification_model = 'classifier_quant_edgetpu.tflite'
    default_labels = 'labels.txt'
    parser = argparse.ArgumentParser()

    parser.add_argument('--detection_model', help='.tflite detection model path',
                        default=os.path.join(default_model_dir, default_detection_model))
    parser.add_argument('--classification_model', help='.tflite classification model path',
                        default=os.path.join(default_model_dir, default_classification_model))
    parser.add_argument('--image', help='Path of the image.',
            default="test_images/test_image_9_faces.jpg")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {}'.format(args.detection_model))
    detection_interpreter = make_interpreter(args.detection_model, device=':0')
    detection_interpreter.allocate_tensors()

    print('Loading {}'.format(args.classification_model))
    classification_interpreter = make_interpreter(args.classification_model, device=':0')
    classification_interpreter.allocate_tensors()

    print('Loading {}'.format(args.image))
    image = load_image(args.image)

    boxes,predictions = detect_and_classify_faces(detection_interpreter,classification_interpreter,image,args.threshold)

    final_image = append_boxes_to_img(image,boxes,predictions)

    cv2.imshow('image',final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()