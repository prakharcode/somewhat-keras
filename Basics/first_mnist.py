import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f'Train images shape {train_images.shape}')
print('Total traning images {0}'.format(len(train_labels)))
print(f'Test images shape {test_images.shape}')
print('Total test images {0}'.format(len(test_labels)))

plt.imshow(train_images[2], cmap=plt.cm.binary)
plt.show()

train_images = train_images.reshape((60000, 28 * 28)) # creating vectors of (28, 28) matrix
train_images = train_images.astype('float32')/255


train_labels = to_categorical(train_labels) # turning into one-hot encoded

test_images = test_images.reshape((10000, 28 * 28))# creating vectors of (28, 28) matrix
test_images = test_images.astype('float32')/255

test_labels = to_categorical(test_labels)# turning into one-hot encoded


net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=( 28 * 28, )))
net.add(layers.Dense(10, activation='softmax'))


net.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(net.summary())

net.fit(train_images, train_labels, epochs = 5, batch_size=64)

test_loss, test_acc = net.evaluate(test_images, test_labels)
print(f'Test Accuracy, {test_acc}')
