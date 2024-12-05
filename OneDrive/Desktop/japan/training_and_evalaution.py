import tensorflow as tf
import matplotlib.pyplot as plt
from model import make_model
from data_generation import make_data, encode_data
from difflib import SequenceMatcher
from data_generation import chars
import numpy as np
from data_generation import ind_to_char

# Dataset dimensions
width, height = 32, 512
min_len, max_len = 4, 16
N_train, N_valid, N_test = 10000, 2000, 2000

# Load data
X_train, y_train = make_data(N_train, min_len=min_len, max_len=max_len)
X_valid, y_valid = make_data(N_valid, min_len=min_len, max_len=max_len)
X_test, y_test = make_data(N_test, min_len=min_len, max_len=max_len)
X_train = X_train.reshape(-1, 32, 512, 1)
X_valid = X_valid.reshape(-1, 32, 512, 1)
X_test = X_test.reshape(-1, 32, 512, 1)

#dataset for training
batch_size = 20
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(encode_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).map(encode_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(encode_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

char_to_ind = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)

# model
model = make_model(width, height, char_to_ind)

## Training loop
epochs = 10
history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, verbose=2)

#training and validation loss graph
plt.figure(figsize=(20, 5))
plt.plot(range(1, len(history.history["loss"]) + 1), history.history["loss"], label="Training Loss")
plt.plot(range(1, len(history.history["val_loss"]) + 1), history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(fontsize=14)
plt.show()

# Evaluate the model
test_loss = model.evaluate(test_ds, verbose=2)
print(f"Test Loss: {test_loss}")

# Prediction Model
prediction_model = tf.keras.models.Model(
    model.get_layer(name="image").input, 
    model.get_layer(name="dense2").output
)


def decode_batch_predictions(preds, max_len):
    
    pred_texts = []
    for pred in preds:
        decoded = tf.argmax(pred, axis=-1).numpy()
        text = "".join([ind_to_char[i].numpy().decode("utf-8") for i in decoded if i != 0])
        pred_texts.append(text[:max_len])
    return pred_texts

y_pred = []
y_true = []

for batch in test_ds:
    batch_images, batch_labels = batch  
    
    # Predict the output
    preds = prediction_model.predict(batch_images)
    
    # Decode predictions
    pred_texts = decode_batch_predictions(preds, max_len=max_len)
    y_pred.extend(pred_texts)
    
    # Decode true labels
    orig_texts = []
    for label in batch_labels:
        label_text = tf.strings.reduce_join(ind_to_char(label))
        orig_texts.append(label_text.numpy().decode("utf-8"))
    y_true.extend(orig_texts)

for i in range(10):
    print(f"True: {y_true[i]} | Predicted: {y_pred[i]}")

def similarity(x, y):
    return SequenceMatcher(None, x, y).ratio()

import tensorflow as tf
import matplotlib.pyplot as plt
from model import make_model
from data_generation import make_data, encode_data
from difflib import SequenceMatcher
from data_generation import chars
import numpy as np
from data_generation import ind_to_char

# Dataset dimensions
width, height = 32, 512
min_len, max_len = 4, 16
N_train, N_valid, N_test = 10000, 2000, 2000

# Load data
X_train, y_train = make_data(N_train, min_len=min_len, max_len=max_len)
X_valid, y_valid = make_data(N_valid, min_len=min_len, max_len=max_len)
X_test, y_test = make_data(N_test, min_len=min_len, max_len=max_len)
X_train = X_train.reshape(-1, 32, 512, 1)
X_valid = X_valid.reshape(-1, 32, 512, 1)
X_test = X_test.reshape(-1, 32, 512, 1)

#dataset for training
batch_size = 20
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(encode_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).map(encode_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(encode_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

char_to_ind = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)

# model
model = make_model(width, height, char_to_ind)

## Training loop
epochs = 1
history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, verbose=2)

#training and validation loss graph
plt.figure(figsize=(20, 5))
plt.plot(range(1, len(history.history["loss"]) + 1), history.history["loss"], label="Training Loss")
plt.plot(range(1, len(history.history["val_loss"]) + 1), history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(fontsize=14)
plt.show()

# Evaluate the model
test_loss = model.evaluate(test_ds, verbose=2)
print(f"Test Loss: {test_loss}")

# Prediction Model
prediction_model = tf.keras.models.Model(
    model.get_layer(name="image").input, 
    model.get_layer(name="dense2").output
)

def decode_batch_predictions(preds, max_len):
    """Decode batch predictions."""
    pred_texts = []
    for pred in preds:
        decoded = tf.argmax(pred, axis=-1).numpy()
        text = "".join([ind_to_char[i].numpy().decode("utf-8") for i in decoded if i != 0])
        pred_texts.append(text[:max_len])
    return pred_texts

def calculate_similarity(true, pred):
    """Calculate similarity between true and predicted text."""
    return SequenceMatcher(None, true, pred).ratio()

def evaluate_model(test_ds, prediction_model, max_len):
    """Evaluate the model on the test dataset."""
    y_pred = []
    y_true = []

    for batch in test_ds:
        batch_images, batch_labels = batch  
        
        # Predict the output
        preds = prediction_model.predict(batch_images)
        
        # Decode predictions
        pred_texts = decode_batch_predictions(preds, max_len=max_len)
        y_pred.extend(pred_texts)
        
        # Decode true labels
        orig_texts = []
        for label in batch_labels:
            label_text = tf.strings.reduce_join(ind_to_char(label))
            orig_texts.append(label_text.numpy().decode("utf-8"))
        y_true.extend(orig_texts)

    similarities = [calculate_similarity(true, pred) for true, pred in zip(y_true, y_pred)]
    average_similarity = np.mean(similarities)
    return y_true, y_pred, average_similarity

y_true, y_pred, average_similarity = evaluate_model(test_ds, prediction_model, max_len)

for i in range(10):
    print(f"True: {y_true[i]} | Predicted: {y_pred[i]}")

print(f"Average Similarity: {average_similarity}")