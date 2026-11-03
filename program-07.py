
# Class names in CIFAR-10
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Predict the first test image
prediction = model.predict(x_test[:1])

predicted_class = np.argmax(prediction[0])

print("Actual:", classes[y_test[0]])
print("Predicted:", classes[predicted_class])

# Print final training and validation accuracy
train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

print("Training Accuracy:", train_accuracy * 100, "%")
print("Validation Accuracy:", val_accuracy * 100, "%")

# Evaluate test accuracy
loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print("Test Accuracy:", test_accuracy * 100, "%")

# Plot accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()
