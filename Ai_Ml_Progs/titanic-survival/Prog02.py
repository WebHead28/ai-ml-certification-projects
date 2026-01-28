from sklearn.datasets import load_digits

dataset = load_digits()

print(dataset.data)
print(dataset.target)
print(dataset.data.shape)
print(dataset.images.shape)

dataimagelength = len(dataset.images)
print(dataimagelength)

n = 1000

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
#plt.show()

dataset.images[n]

X = dataset.images.reshape((len(dataset.images), -1))
Y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print(X_train.shape)
print(X_test.shape)

from sklearn import svm
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

n = 118
plt.matshow(dataset.images[n])

result = model.predict(dataset.images[n].reshape((1, -1)))

print(result)
print("\n")

plt.axis('off')
plt.title('%i' % result)
plt.show()
