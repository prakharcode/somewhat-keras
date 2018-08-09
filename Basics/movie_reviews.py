from keras.datasets import imdb
from keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

'''The argument num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data.
Rare words will be discarded. This allows you to work with vector data of manageable size'''

# Because you’re restricting yourself to the top 10,000 most frequent words, no word index will exceed 10,000

print(f"Max index in train data: {max([max(sequence) for sequence in train_data])}")
print(f"Train label: train_labels[0] = {train_labels[0]}")

print(f"Length of train vector: {len(train_data[0])}")

word_index = imdb.get_word_index() # maps words to index, commonly called as w_to_i
reverse_word_index = dict( [ (value, key) for (key, value) in word_index.items()] ) # this is just opposite, i_to_w

decode_review = ' '.join( [reverse_word_index.get(i - 3, '?') for i in train_data[0] ])

print(f"Encoded Review: \n {train_data[0]} \n")
print(f"Decoded Review: \n {decode_review} \n")

x_train = vectorize_sequence(train_data) # mapping one hot encoding to a particular index
x_test = vectorize_sequence(test_data) # mapping one hot encoding to a particular index

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(f"A data point appears like {x_train[0]}")
print(f"Actual appears as: {y_train[0]}")

# Validation separation

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

'''
The input data is vectors, and the labels are scalars (1s and 0s): this is the easiest setup
you’ll ever encounter. A type of network that performs well on such a problem is
a simple stack of fully connected ( Dense ) layers with relu activations: Dense(16, activation='relu').
The argument being passed to each Dense layer (16) is the number of hidden
units of the layer. A hidden unit is a dimension in the representation space of the layer.

Having 16 hidden units means the weight matrix W will have shape (input_dimension, 16) : the dot product with W will project the input data onto a 16-dimensional represen-
tation space (and then you’ll add the bias vector b and apply the relu operation). You can intuitively understand the dimensionality of your representation space as “how
much freedom you’re allowing the network to have when learning internal representations.” Having more hidden units (a higher-dimensional representation space) allows your
network to learn more-complex representations, but it makes the network more computationally expensive and may lead to learning unwanted patterns (patterns that will improve
performance on the training data but not on the test data, overfitting).
'''

# Model definition

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,), name="Input_Layer"))
model.add(layers.Dense(16, activation='relu', name="Hidden_Layer_1"))
model.add(layers.Dense(1, activation='sigmoid', name="Output_Layer"))

# Review the model

print(model.summary())

# Compiling the model

model.compile(optimizer=optimizers.RMSprop(lr = 0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''
Using custom losses and metrics
from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
'''

# Training the model

# Overfitting @ epochs = 20
# Better fir @ epochs = 4

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 4,
                    batch_size = 128,
                    validation_data = (x_val, y_val))

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

# Running Predict
test = x_test[1].reshape(1, -1)
print(f"Model's Prediction: {model.predict(test)[0]}")
print(f"Actual: {y_test[1]}")
