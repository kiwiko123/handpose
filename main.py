import cv2
import numpy as np
import torch
from detector import YOLOv2Net, bounding_box



def track(predict_every=5) -> None:
    dimensions = (416, 416)
    channels = 1
    i = 1
    net = YOLOv2Net(restore=True)
    capturer = cv2.VideoCapture(0)

    while capturer.isOpened():
        ret, frame = capturer.read()
        frame = cv2.resize(frame, dsize=dimensions)

        if i % predict_every == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = np.reshape(image, (channels,) + dimensions)
            image = torch.Tensor(image)
            image.unsqueeze_(0)

            outputs = net(image)
            predictions = bounding_box(outputs)

            entry = predictions[0]
            confidence = entry.confidence
            coordinates = entry.bounding_box
            if confidence > 0.45:
                tl_x, tl_y, br_x, br_y = [int(c) for c in coordinates]
                green = (0, 255, 0)
                color = np.reshape(frame, (416, 416, 3))
                cv2.rectangle(color, (tl_x, tl_y), (br_x, br_y), green, 2)
                frame = color

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    capturer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track(predict_every=1)