from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

print(f"Length of Train data: {len(train_data)}")
print(f"Length of Test data: {len(test_data)}")

print(f"A look at Train data: \n train_data[1] = {train_data[1]}\n")

# indexing

word_index = reuters.get_word_index()
reverse_word_index = dict([ (value, key) for (key, value) in word_index.items() ])

decoded_newswire = lambda x : ' '.join( [ reverse_word_index.get(i - 3, '?') for i in train_data[x]] )

print(f"Encoded News Article: \n {train_data[0]} \n")
print(f"Decoded News Article: \n {decoded_newswire(0)} \n")

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

print(x_train)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# Validation split

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

'''
I know you probably would be thinking, why not this?

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

Cause, I can :P
'''

model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,), name="Input_layer"))
model.add(layers.Dense(64, activation='relu', name="Hidden_layer"))
model.add(layers.Dense(46, activation='softmax', name="Output_layer"))

# Reviewing the model

print(model.summary())

# Compile
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Training

# overfits @ epochs = 20
# Good fit @ epochs = 9

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 9,
                    batch_size = 512,
                    validation_data = (x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

print(f"Results are: {results}")

history_dict = history.history
acc = history_dict["acc"]
val_acc = history_dict["val_acc"]
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.subplot(211)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# inferring

predictions = model.predict(x_test)
print(f"Shape of prediction vector: {predictions[0].shape}")
print(f"Sum of all the predictions in a vector: {sum(predictions[0])}")
print(f"The predicted output class index, argmax: {np.argmax(predictions[0])}")
