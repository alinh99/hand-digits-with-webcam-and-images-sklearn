import cv2
import numpy as np
import joblib


model = joblib.load("digits.pkl")
image = cv2.imread("img.jpg")
image = cv2.resize(image, None, fx=0.3, fy=0.3)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
_, im_th = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

ctns, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctns]

for rect in rects:
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
    try:
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        number = np.array([roi]).reshape(1, 28 * 28)
        # predict = model.predict(number)
        # number = 28x28
        # predict = model.predict(number)
        # predict = model.predict(number)
        # model=joblib.load("hand_digits.pkl")
        theta = np.loadtxt("theta2.txt")
        print(theta.shape[0])
        one = np.ones((number.shape[0], 1))
        number = np.concatenate((number, one), axis=1)
        predict = 1.0 / (1 + np.exp(-np.dot(number, theta.T)))
        print('prediction', str(int(predict[0])))
        cv2.putText(image, str(int(predict[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    except:
        print('error')
cv2.imshow('image', image)
# cv2.imshow('image_gray', image_gray)
# cv2.imshow('image_threshold', im_th)
cv2.waitKey(0)
cv2.destroyWindow()
