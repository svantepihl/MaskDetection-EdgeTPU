import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import Object, get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


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
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--camera_idx', type=int,
                        help='Index of which video source to use. ', default=0)
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {}'.format(args.detection_model))
    detection_interpreter = make_interpreter(args.detection_model, device=':0')
    detection_interpreter.allocate_tensors()
    detection_inference_size = input_size(detection_interpreter)

    print('Loading {}'.format(args.classification_model))
    classification_interpreter = make_interpreter(args.classification_model, device=':0')
    classification_interpreter.allocate_tensors()
    classification_labels = read_label_file(args.labels)
    classification_inference_size = input_size(classification_interpreter)

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, detection_inference_size)
        run_inference(detection_interpreter, cv2_im_rgb.tobytes())
        faces = get_objects(detection_interpreter, args.threshold)


        cv2_im = append_faces_to_img(cv2_im, detection_inference_size, faces, classification_labels)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def classify_faces(cv2_im, classification_interpreter, detection_inference_size, objs):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / detection_inference_size[0], height / detection_inference_size[1]

    for obj in objs:
        bbox = obj.bbox.scale(scale_x,scale_y)
        xmin, ymin = int(bbox.xmin), int(bbox.ymin)
        xmax, ymax = int(bbox.xmax), int(bbox.ymax)
        
    return faces 

def append_faces_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


if __name__ == '__main__':
    main()
