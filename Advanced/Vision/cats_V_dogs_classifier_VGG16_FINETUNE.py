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

conv_base = VGG16( weights = 'imagenet',
                    include_top = False,
                    input_shape = (150, 150, 3))

print(conv_base.summary())

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 40,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size = (150, 150),
                batch_size = 20,
                class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
                validation_dir,
                target_size = (150, 150),
                batch_size = 20,
                class_mode = 'binary')



model.compile(loss = 'binary_crossentropy',
                optimizer= optimizers.RMSprop(lr = 1e-5),
                metrics = ['acc'])
history = model.fit_generator(
                train_generator,
                steps_per_epoch = 100,
                epochs = 50,
                validation_data = validation_generator,
                validation_steps = 50)

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
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and Validation loss")
plt.legend()

plt.show()
