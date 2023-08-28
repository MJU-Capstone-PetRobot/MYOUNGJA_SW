# https://github.com/cluangar/YOLOv5-RK3588-Python
# 

import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
from lib.postprocess import yolov5_post_process, letterbox_reverse_box
import lib.config as config
from lib.realsense_depth import *
import time
import serial

py_serial = serial.Serial(
    port = '/dev/ttyUSB0',
    baudrate = 115200,
    timeout=0.1
)



IMG_SIZE = config.IMG_SIZE
CLASSES = config.CLASSES

# decice tree for rk3588
DEVICE_COMPATIBLE_NODE = config.DEVICE_COMPATIBLE_NODE

# rknn mode for rk3588
RK3588_RKNN_MODEL_YOLO5 = config.RK3588_RKNN_MODEL_YOLO5

# Initialize Camera Intel Realsense
dc = DepthCamera()

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def draw(image, boxes, scores, classes, dw, dh, depth_frame):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    is_person = False
    person_cx = 0
    person_cy = 0

    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate x1,y1,x2,down: [{}, {}, {}, {}]'.format(x1, y1, x2, y2))

        # Transform Box to original image
        # x1, y1, x2, y2 = letterbox_reverse_box(x1, y1, x2, y2, config.CAM_WIDTH, config.CAM_HEIGHT, config.IMG_SIZE, config.IMG_SIZE, dw, dh) # 실제 객체보다 box height가 더 크게 그려짐, 여기서 잘못 처리되는 것 같음

        x1 = int(x1)
        y1 = int(y1) - 80
        x2 = int(x2)
        y2 = int(y2) - 80


        cx = (x1 + x2) // 2 # object center point x
        cy = (y1 + y2) // 2 # object center point y

        distance = depth_frame[cy, cx]

        # bounding box, center cross line
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(image, (cx, y1), (cx, y2), (255, 0, 0), 2)
        cv2.line(image, (x1, cy), (x2, cy), (255, 0, 0), 2)

        # class, score, depth text 
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score), (cx + 5, cy - 5), 0, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "{} cm".format(distance / 10), (cx + 5, cy + 20), 0, 0.6, (0, 0, 255), 2)
        
        if CLASSES[cl] is 'person':
            is_person = True
            person_cx = cx
            person_cy = cy

    return is_person, person_cx, person_cy

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    y1, y2 = int(round(dh - 0.1)), int(round(dh + 0.1))
    x1, x2 = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, y1, y2, x1, x2, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL_YOLO5
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite(verbose=False) # using verbose option saves lots of state

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('failed to load RKNN model')
        exit(ret)
    print('done\n')

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    frame_time_cur = 0
    frame_time_prev = 0

    # loop
    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        ori_frame = color_frame

        frame_time_cur = time.time()
        fps = int(1 // (frame_time_cur - frame_time_prev))
        frame_time_prev = frame_time_cur
        fps = str("FPS {}".format(fps))
        # print("fps : ", fps)

        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        color_frame, ratio, (dw, dh) = letterbox(color_frame, new_shape=(IMG_SIZE, IMG_SIZE))

        # Inference
        print('--> Inference')
        outputs = rknn_lite.inference(inputs=[color_frame])

        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        # YOLO post process
        boxes, classes, scores = yolov5_post_process(input_data)
        yawn = 0

        if boxes is not None:
            is_person, person_cx, person_cy = draw(ori_frame, boxes, scores, classes, dw, dh, depth_frame) # center point of person
            if is_person is True:
                print("person, person_cx, person_cy", person_cx, person_cy)
                # TODO: Turn robot head towards the person.
                x_error = 320- person_cx

                
      
                if x_error> 0:
                	yawn+=1
                elif x_error< 0:
                    yawn-=1

                motorinput = "2"+ ","+ "0"+ ","+ "70"+ ","+ str(round(yawn)) +"\n"
                print(motorinput)
                py_serial.write(motorinput.encode())
                

            #cv2.putText(ori_frame, fps, (10, 40), 0, 0.6, (0, 0, 255), 2)
            #cv2.imshow("yolov5 post process result", ori_frame)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    py_serial.close()
    rknn_lite.release()
    cv2.destroyAllWindows()
