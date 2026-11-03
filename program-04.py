
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# 1. Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalize and flatten images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 3. Define input layer
input_img = layers.Input(shape=(784,))

# 4. Encoder - compress the image
encoded = layers.Dense(32, activation="relu")(input_img)

# 5. Decoder - reconstruct the image
decoded = layers.Dense(784, activation="sigmoid")(encoded)

# 6. Build autoencoder
model = Model(input_img, decoded)

# 7. Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy")

# 8. Train the autoencoder
model.fit(
    x_train, x_train,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# 9. Create encoder model
encoder = Model(input_img, encoded)

# 10. Compress test images
encoded_imgs = encoder.predict(x_test)

# 11. Reconstruct test images
decoded_imgs = model.predict(x_test)

# 12. Display original and reconstructed images
n = 5
plt.figure(figsize=(10, 4))

for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 13. Display dimensions
print("Original size:", x_test.shape)
print("Compressed size:", encoded_imgs.shape)
