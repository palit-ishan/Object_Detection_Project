from imutils.video import VideoStream
from imutils.video import FPS
import tensorflow as tf
from absl import logging
import numpy as np
import argparse
import imutils
import time
import cv2

tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# labels need to be read and not hard-coded
CLASSES = [
        'Road Work'
        ,'Pedestrian Crossing'
        ,'Bicycle Crossing'
        ,'Stop Sign'
        ,'Traffic Sign - Red'
        ,'Traffic Sign - Green'
        ,'Traffic Sign - Yellow'
    ]
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3))

_OUTPUT_LOCATION = 'location'
_OUTPUT_CATEGORY = 'category'
_OUTPUT_SCORE = 'score'
_OUTPUT_NUMBER = 'number of detections'

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    #print(input_tensor.shape)
    input_tensor[:,:] = rframe

def detect(interpreter, image):
    '''
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    #print(input_tensor.shape)
    input_tensor[:,:] = rframe
    '''
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    count = int(np.squeeze(interpreter.get_tensor(output_indices[_OUTPUT_NUMBER])))
    scores = np.squeeze(interpreter.get_tensor(output_indices[_OUTPUT_SCORE]))
    classes = np.squeeze(interpreter.get_tensor(output_indices[_OUTPUT_CATEGORY]))
    boxes = np.squeeze(interpreter.get_tensor(output_indices[_OUTPUT_LOCATION]))
    
    return count, scores, classes, boxes

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help="Give path to saved model")
    ap.add_argument('-c', '--confidence', type=float, default=0.3,
                    help='minimum probability to filter weak detections')
    args = vars(ap.parse_args())

    print('[INFO] Loading model...')

    interpreter = tf.lite.Interpreter(model_path = args['model'])
    interpreter.allocate_tensors()
    #signature_fn = interpreter.get_signature_runner() #works for tf >= v.2.5

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sorted_output_indices = sorted(
            [output['index'] for output in output_details]
        )

    output_indices = {
            _OUTPUT_LOCATION: sorted_output_indices[0],
            _OUTPUT_CATEGORY: sorted_output_indices[1],
            _OUTPUT_SCORE: sorted_output_indices[2],
            _OUTPUT_NUMBER: sorted_output_indices[3],
        }

    resize_dim = (input_details[0]['shape'][2], input_details[0]['shape'][1])
    input_data_type = input_details[0]['dtype']

    print('[INFO] Starting video stream ...')
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    
    while True:
        frame = vs.read()
        rframe = cv2.resize(frame, resize_dim)
        rframe = np.array(rframe, dtype=input_data_type)
        #rframe = np.expand_dims(rframe, axis=0)
        #print(rframe.shape)

        count, scores, classes, boxes = detect(interpreter, rframe)
        
        results = []
        for i in range(count):
            result = None
            if scores[i] > float(args['confidence']):
                result = {
                        'bounding_box': boxes[i],
                        'class_id': classes[i],
                        'score': scores[i]
                    }
            results.append(result)
            for obj in results:
                if obj is None:
                    continue
                ymin, xmin, ymax, xmax = obj['bounding_box']
                xmin = int(xmin * frame.shape[1])
                xmax = int(xmax * frame.shape[1])
                ymin = int(ymin * frame.shape[0])
                ymax = int(ymax * frame.shape[0])
                
                class_id = int(obj['class_id'])
                
                color = [int(c) for c in COLORS[class_id]]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                label = "{}: {:.0f}%".format(CLASSES[class_id], obj['score']*100)
                cv2.putText(frame, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        fps.update()

    fps.stop()
    print('[INFO] elapsed Time: {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
