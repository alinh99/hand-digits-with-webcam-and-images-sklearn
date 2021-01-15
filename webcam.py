import cv2
vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 1)
    _, fr_th = cv2.threshold(frame_gray, 140, 255, cv2.THRESH_BINARY_INV)
    ctns, _ = cv2.findContours(fr_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctns]
    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()