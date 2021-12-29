import random

import cv2
from torchvision.transforms import transforms

from core.itg_executor import ItgExecutor, ItgJob
from core.raw_dnn import RawDNN
from dnn_models.faster_rcnn import prepare_fasterrcnn

NAMES = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return b, g, r


if __name__ == '__main__':
    raw_dnn = RawDNN(prepare_fasterrcnn())

    capture = cv2.VideoCapture('../media/road.mp4')
    ret, frame_bgr = capture.read()
    if not ret:
        raise Exception("failed to read video")
    frame_bgr = cv2.resize(frame_bgr, (1080, 720))  # x轴长度1080，y轴长度720
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    ipt = preprocess(frame_rgb)

    results = ItgExecutor(raw_dnn).exec(ItgJob(list(range(1, len(raw_dnn.layers))),
                                               [raw_dnn.layers[-1].id_], {0: [ipt]}))
    out = results[raw_dnn.layers[-1].id_][0]
    boxes, labels, scores = out['boxes'], out['labels'], out['scores']
    for idx in range(boxes.shape[0]):
        if scores[idx] >= .8:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            name = NAMES.get(str(labels[idx].item()))
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), random_color(), thickness=2)
            cv2.putText(frame_bgr, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imshow('result', frame_bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()
