import os 
import pandas as pd 
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

'''
check tf version and cuda version are compatible with this GPU. 
pip install tensorflow-gpu==2.10.* 
'''

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, meta_data, batch_size):
        self.data_paths = meta_data['filename'].values # e.g., list of image file paths
        self.labels = meta_data['label'].values
        self.batch_size = batch_size

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Calculate start and end indices for the current batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_paths = self.data_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # Load and preprocess data for the batch (customize this part)
        # Example: loading images from paths
        imgs_ = np.zeros((self.batch_size, 128,128,3))
        labs_ = np.zeros((self.batch_size, 2)) 

        for j, (path, lab) in enumerate(zip(batch_paths, batch_labels)):
            # print('lab ', lab, type(lab))
            img = Image.open(path).convert('RGB')
            imgs_[j,...] = np.array(img) / 255.0
            labs_[j, lab] = 1.0

        return imgs_, labs_

    def on_epoch_end(self):
        """Optional: method called at the end of every epoch, useful for shuffling data."""
        # For example, shuffle data order after each epoch
        pass
    
def make_simple_cnn(input_shape=(128, 128, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # binary by default
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def ft_resnet(num_classes):
    # load the model    
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(128,128,3))
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)


    model = tf.keras.models.Model(inputs=base.input, outputs=out)

def my_categorical_crossentropy(y_true, y_pred, eps=1e-7):
    # y_true: (batch, C) one-hot
    # y_pred: (batch, C) softmax probs

    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return tf.reduce_mean(loss)

def train_model():
    # split the data sets
    dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/patch_sets/mass1')
    meta_train = os.path.join(dir_, 'train_set.csv')
    meta_test = os.path.join(dir_, 'test_set.csv')
    
    train_df = pd.read_csv(meta_train)
    test_df = pd.read_csv(meta_test)
    
    num_classes = 2

    print(f"Train: {len(train_df)} | Test: {len(test_df)}") 

    model = make_simple_cnn(input_shape=(128, 128, 3), num_classes=2)
    model.compile(optimizer="adam",
                   loss=my_categorical_crossentropy, 
                   metrics=["accuracy"])
    train_loader = CustomDataLoader(train_df, batch_size=5)
    test_loader = CustomDataLoader(test_df, batch_size=5)
    history = model.fit(train_loader, validation_data=test_loader, epochs=5)

if __name__ == "__main__":
    
    train_model()
    
