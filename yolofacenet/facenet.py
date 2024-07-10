
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
PATH_TO_STORE_MODEL = "./models/"
FILE_NAME = "facenet.h5"


def load_model():
    """Load pretrained facenet model
    """
    print("Preparing facenet model...")
    print(PATH_TO_STORE_MODEL + FILE_NAME)
    return tf.keras.models.load_model(PATH_TO_STORE_MODEL + FILE_NAME)
