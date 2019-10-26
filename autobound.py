import cv2
import os




cap = cv2.VideoCapture('Input/Store.mp4')
os.makedirs('Output/Store.mp4')
count = 0

while cap.isOpened():
    frameWasCaptured, frame = cap.read()

    if frameWasCaptured:
        cv2.imwrite('Output/Store.mp4/frame{:d}.jpg'.format(count), frame)
        count += 30 # i.e. at 30 fps, this advances one second
        cap.set(1, count)
    else:
        cap.release()
        break
