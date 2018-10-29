from keras.datasets import imdb
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

max_features = 10000
max_len = 500

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words= max_features)

print(f'Train Sequence: {len(x_train)}')
print(f'Test Sequence: {len(x_test)}')

print('Pad sequence (sample * Time)')
x_train = sequence.pad_sequences(x_train, maxlen = max_len)
x_test = sequence.pad_sequences(x_test, maxlen= max_len)

print(f'x_train_shape: {x_train.shape}')
print(f'x_test_shape: {x_test.shape}')

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length= max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation= 'relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr= 1e-4),
              loss= 'binary_crossentropy',
              metrics= ['acc'])

history = model.fit(x_train, y_train,
                    epochs= 10,
                    batch_size= 128,
                    validation_split= 0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss Conv1D')
plt.legend()

plt.show()
