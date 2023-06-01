import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# EDA - Explore the dataset
print("Training Set:")
print("Number of samples:", x_train.shape[0])
print("Image shape:", x_train.shape[1:])
print("Label shape:", y_train.shape)
print("Unique labels:", set(y_train))

# Visualize some samples from the training set
plt.figure(figsize=(10, 4))
digit_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Labels of digits to display
for i, digit in enumerate(digit_indices):
    indices = np.where(y_train == digit)[0]  # Find indices of samples with the current label
    sample_index = indices[i]  # Get the i-th sample with the current label
    plt.subplot(1, len(digit_indices), i + 1)
    plt.imshow(x_train[sample_index], cmap='gray')
    plt.title(str(y_train[sample_index]))
    plt.axis('off')
plt.tight_layout()
plt.show()

# Preprocess the data
input_shape = (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile and train the model
batch_size = 30
epochs = 10
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(x_test, y_test))
print("The model has been successfully trained")

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('mnist.h5')
print("Saving the model as mnist.h5")
