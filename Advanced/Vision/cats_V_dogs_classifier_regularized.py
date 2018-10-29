import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


train_dir = os.path.join(os.getcwd(), 'data/cats_V_dogs/train')
validation_dir = os.path.join(os.getcwd(), 'data/cats_V_dogs/val')

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
                                train_dir,
                                target_size = (150, 150),
                                batch_size = 20,
                                class_mode = 'binary')
validation_generators = val_datagen.flow_from_directory(
                                validation_dir,
                                target_size = (150, 150),
                                batch_size = 20,
                                class_mode = 'binary')


print(" S A N I T Y C H E C K ")
for data_batch, labels_batch in train_generator:
    print(f"Data batch shape: {data_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    break


model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3), name="Conv_1"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', name="Conv_2"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv_3"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv_4"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5, name="DropOut"))
model.add(layers.Dense(512, activation='relu', name="fc_1"))
model.add(layers.Dense(1, activation='sigmoid', name="Output"))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc'])


print(model.summary())

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = 100,
                    epochs = 20,
                    validation_data = validation_generators,
                    validation_steps = 50)

model.save('cats_V_dogs_regularized.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = "Training acc")
plt.plot(epochs, val_acc, 'b', label = "Validation acc")
plt.title('Training and Val accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
