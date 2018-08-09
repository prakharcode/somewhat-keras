from keras.datasets import boston_housing as Boston
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = Boston.load_data()

print(f"Shape of train data: {train_data.shape}")
print(f"Shape of test_data: {test_data.shape}")

print(f"Real valued targets: {train_targets}")

# Normalize

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

# either denormalize the output or normalize the test-set as well, cause
# train and test set should come from the same distribution.

test_data -= mean
test_data /= std

# Build network

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # single node cause regression.
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# K(4) fold validation
k = 4
num_val_samples = len(train_data) // k #integral part
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate( [train_data[:i * num_val_samples],
                                        train_data[ (i + 1) * num_val_samples:]],
                                        axis=0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]],
                                            axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data = (val_data, val_targets),
                        epochs = num_epochs, batch_size = 1, verbose = 0)
    all_mae_histories.append(history.history["val_mean_absolute_error"])

# Averaging over the scores per epoch

average_mae_history = [ np.mean( [ x[i] for x in all_mae_histories] ) for i in range(num_epochs)]

print(f"Mean of all the scores: {average_mae_history}")

# Plot

plt.plot(range(1, len(average_mae_history) +1), average_mae_history)
plt.ylabel('Validation MAE')
plt.show()

def plot_smooth(points, factor=0.9):
    smoothened_points = []
    for point in points:
        if smoothened_points:
            previous = smoothened_points[-1]
            smoothened_points.append(previous * factor + point * (1 - factor))
        else:
            smoothened_points.append(point)
        return smoothened_points

smooth_mae_history = plot_smooth(average_mae_history[10:])
plt.clf()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(f"Test MSE: {test_mse_score}, Test MAE: {test_mae_score}")
