from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.optimizers import RMSprop
from keras.models import Sequential
import matplotlib.pyplot as plt

# Check whether Bi-directional RNN can be useful

max_features = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words= max_features)

x_train_rev = [x[::-1] for x in x_train]
x_test_rev = [x[::-1] for x  in x_test]

x_train = sequence.pad_sequences(x_train, maxlen= maxlen)
x_test = sequence.pad_sequences(x_test, maxlen= maxlen)

x_train_rev = sequence.pad_sequences(x_train_rev, maxlen= maxlen)
x_test_rev = sequence.pad_sequences(x_test_rev, maxlen= maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc'])

# In linear order

history = model.fit(x_train, y_train,
                    epochs= 10,
                    batch_size= 128,
                    validation_split= 0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']

# In reverse order

history_rev = model.fit(x_train_rev, y_train,
                    epochs= 10,
                    batch_size= 128,
                    validation_split= 0.2)


epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()
plt.figure()

loss_rev = history_rev.history['loss']
val_loss_rev = history_rev.history['val_loss']
epochs = range(1, len(loss_rev) + 1)

plt.plot(epochs, loss_rev, 'bo', label='Training loss')
plt.plot(epochs, val_loss_rev, 'b', label='Validation loss')

plt.title('Training and validation loss Reverse')
plt.legend()

plt.show()

# Bi- Directional Training

model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1), activation='sigmoid')

model.compile(optimizer= RMSprop(), loss= 'binary_crossentropy', metrics= ['acc'])
history_bi = model.fit(x_train, y_train,
                    epochs= 10,
                    batch_size= 128,
                    validation_split= 0.2)

# Plot
loss_bi = history_bi.history['loss']
val_loss_bi = history_bi.history['val_loss']
epochs = range(1, len(loss_bi) + 1)

plt.plot(epochs, loss_bi, 'bo', label='Training loss')
plt.plot(epochs, val_loss_bi, 'b', label='Validation loss')

plt.title('Training and validation loss Bi direction')
plt.legend()

plt.show()
