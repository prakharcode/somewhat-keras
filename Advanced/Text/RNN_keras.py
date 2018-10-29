from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

# Just initialization
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

# Returning Sequential

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences= True))
model.summary()

# Stacking RNNs

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences= True))
model.add(SimpleRNN(32, return_sequences= True))
model.add(SimpleRNN(32, return_sequences= True))
model.add(SimpleRNN(32))
model.summary()
