import tensorflow as tf
import os
import keras
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img

img_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.getcwd(),
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False


)

