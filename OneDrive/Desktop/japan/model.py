import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, LSTM, Bidirectional

# CTCLayer for loss calculation
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

#model
def make_model(width, height, char_to_ind):
    imgs = tf.keras.layers.Input(shape=(width, height, 1), name="image", dtype="float32")
    labels = tf.keras.layers.Input(name="label", shape=(None,), dtype="float32")

    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(imgs)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    x = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = (width // 4, (height // 4) * 64)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(1024, activation="relu", name="dense1")(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.5))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(x)

    x = Dense(len(char_to_ind.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = tf.keras.models.Model(inputs=[imgs, labels], outputs=output, name="ocr_model")
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model
