import cv2
import numpy as np
import torch
from torchvision import transforms
from detector import train_net



def track():
    net = train_net()
    mean = [0.5, 0.5, 0.5]
    std = mean
    capturer = cv2.VideoCapture(0)
    i = 0

    while capturer.isOpened():
        ret, frame = capturer.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if i % 10 == 0:
            image = torch.Tensor(gray)
            image = transforms.functional.normalize(image, mean, std)
            image = image.data.numpy()
            image = cv2.resize(image, (292, 292))
            image = np.reshape(image, (3, 292, 292))
            image = torch.Tensor([image])
            outputs = net(image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted[0].item()
            print(prediction == 0)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    # When everything done, release the capture
    capturer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    track()