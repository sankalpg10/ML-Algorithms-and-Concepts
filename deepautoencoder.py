"""Building an Deep autoencoder for dimensionality reduction.
Inspired by : https://blog.keras.io/building-autoencoders-in-keras.html """


import keras
from keras import layers
from keras import regularizers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# Dimension of encoded representation
encoded_dim = 32  # assuming input is size 784

# input image
input_image = keras.Input(shape=(784,))


# encoded representation of our input
encoded = layers.Dense(128, activation="relu")(input_image)
encoded = layers.Dense(64, activation="relu")(encoded)
encoded = layers.Dense(32, activation="relu")(encoded)

# decoded : reconstruction of the original input
decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(784, activation="sigmoid")(encoded)

# model
autoencoder = keras.Model(input_image, decoded)

# separate encoder model
encoder = keras.Model(input_image, encoded)

# separate decoder model

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoded_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# training our autoencoder
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


# Preparing mnist data

(x_train, _), (x_test, _) = mnist.load_data()

# Normalizing all values between 0 and 1, and flattening the 28x28 image to 784 sized vectors

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(f"x_train shape before flattening : {x_train.shape}")
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
print(f"x_train shape after flattening : {x_train.shape}")

print(f"x_test shape before flattening : {x_test.shape}")
x_test = x_test.reshape((len(x_test)), np.prod(x_test.shape[1:]))
print(f"x_test shape after flattening : {x_test.shape}")

# Training our autoencoder for 50 epochs, our x and y are same because we want to learn to recreate the same image input we dont need the mnist labels
autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)

# Encoding and decoding some images from the test data

encoded_test_images = encoder.predict(x_test)  # using the separate encoder model
decoded_test_images = decoder.predict(
    encoded_test_images
)  # decoding the encoded images


n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display the encoded, shows that the reduced dimensions obv wont look the same as the original but since the decoded images are similar to original, we can safel
    # say that the encodings are a correct represenation of the original images with reduced dimemsion
    # ax = plt.subplot(2, n, i + 1)
    # plt.imshow(encoded_test_images[i].reshape(4, 8))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
