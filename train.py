import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from model.u2net import build_u2net_lite, build_u2net

HEIGHT = 256
WIDTH = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "masks", "*.png")))

    # Split training data into train and validation sets
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=split, random_state=42
    )

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    # Read image file using TensorFlow
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode as RGB
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = image / 255.0  # Normalize to [0, 1]
    return image

def read_mask(path):
    # Read mask file using TensorFlow
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)  # Decode as grayscale
    mask = tf.image.resize(mask, [HEIGHT, WIDTH])
    mask = mask / 255.0  # Normalize to [0, 1]
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.py_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([HEIGHT, WIDTH, 3])
    y.set_shape([HEIGHT, WIDTH, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_device():
    """Check available device (GPU/CPU)"""
    return '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

def plot_loss(csv_log_path):
    """Plot training and validation loss from the CSV log file"""
    data = pd.read_csv(csv_log_path)

    # Plotting the loss and val_loss
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['loss'], label='Training Loss')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")

    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 10

    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "log.csv")

    # dataset root path
    dataset_path = "./"  # Set the working directory as the dataset path

    # Load dataset
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path)
    print(f"Train:    {len(train_x)} - {len(train_y)}")
    print(f"Validate: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH_SIZE)

    """Model"""
    with tf.device(get_device()):
        # Check if model exists and load it, otherwise create a new one
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Loaded pre-trained model from disk.")
        else:
            model = build_u2net((HEIGHT, WIDTH, 3))
            print("Created new model.")

        model.compile(loss="binary_crossentropy", optimizer=Adam(LEARNING_RATE))

        # Callbacks
        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path),
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            TensorBoard(log_dir='./logs')
        ]

        # If log file exists, continue training; otherwise, start fresh
        if os.path.exists(csv_path):
            print(f"Found existing log file: {csv_path}")
        else:
            print(f"Starting new training session, no existing log file found.")

        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=valid_dataset,
            callbacks=callbacks
        )

    # Plot the loss
    plot_loss(csv_path)
