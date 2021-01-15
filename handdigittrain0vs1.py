from sklearn.datasets import fetch_openml
import numpy as np

# mnist = fetch_mldata('mnist-original',data_home='./')
mnist = fetch_openml("mnist_784")
N, d = mnist.data.shape
print(N)
print(d)
y_all = mnist.target
x_all = mnist.data
# print(x_all[0])
# print(y_all)

import matplotlib
import matplotlib.pyplot as plt

plt.imshow(x_all.T[:, 3000].reshape(28, 28))
plt.axis("off")
plt.show()

print(y_all[3000])

# chỉ chơi với 2 số 0 và 1
# cần filter dữ liệu của số 0 và số 1
print(np.where(y_all == 0))

x0 = x_all[np.where(y_all == '0')[0]]
x1 = x_all[np.where(y_all == '1')[0]]
print(x0[0])
print(x1[1])

y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])

# ghép lại thành 1 dữ liệu
x = np.concatenate((x0, x1), axis=0)
y = np.concatenate((y0, y1))

# print(x)
# print(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# chia dữ liệu train và test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1000)

model = LogisticRegression(C=1e5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('Accuracy: ' + str(100 * accuracy_score(y_test, y_pred)))

# lưu model lại để dùng
import joblib
model=joblib.load("hand_digits.pkl")
theta=np.loadtxt("theta.txt")
print(theta.shape[0])






