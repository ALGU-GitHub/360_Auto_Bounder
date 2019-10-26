import cv2

cap = cv2.VideoCapture('Input/Store.mp4')
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('frame{:d}.jpg'.format(count), frame)
        count += 30 # i.e. at 30 fps, this advances one second
        cap.set(1, count)
    else:
        cap.release()
        break
