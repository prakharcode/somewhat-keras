from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f'Train images shape {train_images.shape}')
print('Total traning images {0}'.format(len(train_labels)))
print(f'Test images shape {test_images.shape}')
print('Total test images {0}'.format(len(test_labels)))

plt.imshow(train_images[2], cmap=plt.cm.binary)
plt.show()

train_images = train_images.reshape((60000, 28 ,28, 1)) # creating vectors of (28, 28, 1) matrix, 1 is color channel
train_images = train_images.astype('float32')/255


train_labels = to_categorical(train_labels) # turning into one-hot encoded

test_images = test_images.reshape((10000, 28, 28, 1))# creating vectors of (28, 28, 1) matrix  1 is color channel
test_images = test_images.astype('float32')/255

test_labels = to_categorical(test_labels)# turning into one-hot encoded



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape= (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2))) # no params
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) # no params added
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(train_images, train_labels, epochs = 5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy, {test_acc}')
