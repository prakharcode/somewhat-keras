from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), 'data')
fname = os.path.join(data_dir, 'jena_climate/jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

float_data = np.zeros((len(lines), len(header) - 1))# Date-time not required
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

print(len(float_data), float_data[0])

temp_var_over_full_dataset= float_data[:, 1]
temp_var_over_first_10_days = temp_var_over_full_dataset[:1440]
plt.plot(range(len(temp_var_over_full_dataset)), temp_var_over_full_dataset)
plt.figure()
plt.plot(range(1440), temp_var_over_first_10_days)
plt.show()

# Normalizing the data

mean = float_data[:200000].mean(axis= 0)
float_data -= mean
std = float_data[:200000].std(axis= 0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index, shuffle= False, batch_size= 128, step= 6):

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
            min_index + lookback, max_index, size =  batch_size)

        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback//step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size =128

train_gen = generator(float_data,
                    lookback= lookback,
                    delay= delay,
                    min_index= 0,
                    max_index= 200000,
                    shuffle= True,
                    batch_size= batch_size)

val_gen = generator(float_data,
                    lookback= lookback,
                    delay= delay,
                    min_index= 200001,
                    max_index= 300000,
                    step= step,
                    batch_size= batch_size)

test_gen = generator(float_data,
                    lookback= lookback,
                    delay= delay,
                    min_index= 300001,
                    max_index= None,
                    step= step,
                    batch_size= batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size

test_steps = (len(float_data) - 300001 - lookback) // batch_size

# calculating mean over all the temprature, NAIVE

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()

# A basic machine learning approach

model = Sequential()
model.add(layers.Flatten(input_shape= (lookback//step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()

# First recurrent baseline

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                            steps_per_epoch= 500,
                            epochs= 20,
                            validation_data= val_gen,
                            validation_steps= val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()

# Using recurrent dropout

model = Sequential()
model.add(layers.GRU(32,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer= RMSprop(), loss= 'mae')

history = model.fit_generator(train_gen,
                            steps_per_epoch = 500,
                            epochs = 40,
                            validation_data = val_gen,
                            validation_steps= val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()

# Stacked RNN

model = Sequential()
model.add(layers.GRU(32,
                    dropout= 0.1,
                    recurrent_dropout = 0.5,
                    return_sequences= True,
                    input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
            dropout= 0.1,
            recurrent_dropout= 0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history= model.fit_generator(train_gen,
                            steps_per_epoch= 500,
                            epochs=40,
                            validation_data= val_gen,
                            validation_steps= val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()
