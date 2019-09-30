import tensorflow as tf
import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Divide each RGB value by 255 to normalize
x_train /= 255
x_test /= 255

model = keras.models.Sequential()

input_shape = (28, 28, 1)
model.add(keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten()) # Flattening the 2D arrays for fully connected layers

model.add(keras.layers.Dense(128, activation=tf.nn.relu))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)

# plt.imshow(x_train[image_index], cmap='Greys')
