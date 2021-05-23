'''
benchmark model on test images
'''
import argparse
import contextlib
import threading
import time
from tqdm import tqdm
from PIL import Image
from statistics import mean

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter


@contextlib.contextmanager
def open_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            yield image


def run_two_models_one_tpu(classification_model, detection_model, image_name,
                           num_inferences, batch_size):
    start_time = time.perf_counter()
    interpreter_a = make_interpreter(classification_model, device=':0')
    interpreter_a.allocate_tensors()
    interpreter_b = make_interpreter(detection_model, device=':0')
    interpreter_b.allocate_tensors()

    identification = []
    classification = []

    with open_image(image_name) as image:
        size_a = common.input_size(interpreter_a)
        common.set_input(interpreter_a, image.resize(size_a, Image.NEAREST))
        _, scale_b = common.set_resized_input(
            interpreter_b, image.size,
            lambda size: image.resize(size, Image.NEAREST))

    num_iterations = (num_inferences + batch_size - 1) // batch_size
    for _ in tqdm(range(num_iterations)):
        for _ in range(batch_size):
            identification_start_time = time.perf_counter()
            interpreter_b.invoke()
            detect.get_objects(interpreter_b, score_threshold=0.,
                               image_scale=scale_b)
            identification.append(time.perf_counter() -
                                  identification_start_time)
        for _ in range(batch_size):
            classification_start_time = time.perf_counter()
            interpreter_a.invoke()
            result1 = classify.get_classes(interpreter_a, top_k=4)
            interpreter_a.invoke()
            result2 = classify.get_classes(interpreter_a, top_k=4)
            interpreter_a.invoke()
            result3 = classify.get_classes(interpreter_a, top_k=4)

            classification.append(time.perf_counter() -
                                  classification_start_time)
    total_time = time.perf_counter() - start_time
    return total_time, identification, classification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--classification_model',
        help='Path of classification model.',
        default="models/classifier_quant_edgetpu.tflite")
    parser.add_argument(
        '--detection_model',
        help='Path of detection model.',
        default="models/facedetector_quant_postprocess_edgetpu.tflite")
    parser.add_argument(
        '--image',
        help='Path of the image.',
        default="test_image_3_faces.jpg")
    parser.add_argument(
        '--num_inferences',
        help='Number of inferences to run.',
        type=int,
        default=2000)
    parser.add_argument(
        '--batch_size',
        help='Runs one model batch_size times before switching to the other.',
        type=int,
        default=1)

    args = parser.parse_args()

    print('Running %s and %s with one Edge TPU, # inferences %d, batch_size %d.' %
          (args.classification_model, args.detection_model, args.num_inferences,
           args.batch_size))
    total_time, identification, classification = run_two_models_one_tpu(args.classification_model,
                                                                        args.detection_model, args.image,
                                                                        args.num_inferences, args.batch_size)

    print('Run took: %.5f. seconds' % total_time)
    print('Average identification time: %.5f seconds' % mean(identification))
    print('Average classification time: %.5f seconds' % mean(classification))
    avg_time = total_time / args.num_inferences
    print('Average total time: %.5f seconds' % avg_time)
    fps = args.num_inferences / total_time
    print('Average FPS: %.5f seconds' % fps)


if __name__ == '__main__':
    main()
