# YOLOv5 Parameter
#OBJ_THRESH = 0.25
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = 640
# YOLOv5 Classes
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

# decice tree for rk3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

# rknn mode for rk3588
RK3588_RKNN_MODEL_YOLO5 = 'models/yolov5s-640-640-from_rknpu2.rknn'

#Webcam dev /device/video0, /device/video1 etc.
CAM_DEV = 0
CAM_DEV2 = 2

#Capture Resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480

#Position Display
D1_WIDTH = 0
D1_HEIGHT = 200

D2_WIDTH = 640
D2_HEIGHT = 200
