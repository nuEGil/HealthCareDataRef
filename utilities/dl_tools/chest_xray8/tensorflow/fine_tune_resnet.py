import os 
import pandas as pd 
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

def df_to_dataset(df, image_size=64, batch_size=32, shuffle=True):
    paths, labels = df["filename"].values, df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        lab = tf.one_hot(label, depth=2)
        return preprocess_input(img), label
    
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(len(df))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model():
    # split the data sets
    dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/patch_sets/mass1')
    meta_train = os.path.join(dir_, 'train_set.csv')
    meta_test = os.path.join(dir_, 'test_set.csv')
    
    train_df = pd.read_csv(meta_train)
    test_df = pd.read_csv(meta_test)
    
    num_classes = 2

    print(f"Train: {len(train_df)} | Test: {len(test_df)}") 

    # load the model    
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(128,128,3))
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam",
                   loss='categorical_crossentropy', 
                   metrics=["accuracy"])
    
    train_ds = df_to_dataset(train_df, image_size=128)
    test_ds  = df_to_dataset(test_df, image_size=128, shuffle=False)
    history = model.fit(train_ds, validation_data=test_ds, epochs=5)

if __name__ == "__main__":
    
    train_model()
    
