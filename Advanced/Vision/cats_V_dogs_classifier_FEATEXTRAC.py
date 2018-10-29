from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

train_dir = os.path.join(os.getcwd(), 'data/cats_V_dogs/train')
validation_dir = os.path.join(os.getcwd(), 'data/cats_V_dogs/val')
test_dir = os.path.join(os.getcwd(),'data/cats_V_dogs/test')



datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

# Creating Conv Base
# This base is used to get the features from VGG16
# and that goes forward to a TRAINABLE NETWORK and a new model
# is formed with this model being its base.
# This is not changed.
conv_base = VGG16( weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))

print(conv_base.summary())


def extract_features(directory, sample_count):
    features = np.zeros((sample_count, 4, 4, 512))
    labels = np.zeros((sample_count))
    generator = datagen.flow_from_directory(
                        directory,
                        target_size = (150, 150),
                        batch_size=batch_size,
                        class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, -1))
validation_features = np.reshape(validation_features, (1000, -1))
test_features = np.reshape(test_features, (1000, -1))


# Initializing a trainable model and getting the features
# from our pretrained model and training on the accquired feature.

model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', input_dim = 4 * 4 * 512, name = 'Dense_train'))
model.add(layers.Dropout(0.5, name = 'DropOut_Train'))
model.add(layers.Dense(1, activation='sigmoid', name = 'Output'))

model.compile(optimizer = optimizers.RMSprop(lr = 2e-5),
                                    loss = 'binary_crossentropy',
                                    metrics = ['acc'])

history = model.fit(train_features, train_labels,
                    epochs = 30,
                    batch_size = 20,
                    validation_data = (validation_features, validation_labels))


# Polting

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title("Training and Validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_acc, 'b', label = 'Validation loss')
plt.title("Training and Validation loss")
plt.legend()

plt.show()
