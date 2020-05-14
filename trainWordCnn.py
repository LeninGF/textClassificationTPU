"""
This script aims to train the model WordCNN migrated
from its tensorflow v1.X original version in Github
to its equivalent in Keras. Then it must be tested on TPU

Author: Lenin GF
"""
#%% importing libraries
import tensorflow as tf
import os
import numpy as np
from data_utils import *
from sklearn.model_selection import  train_test_split
print(tf.__version__)

#%% These are some constants in the project
NUM_CLASS = 14
BATCH_SIZE = 64
NUM_EPOCHS = 10
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014

#%% Downloading the dataset
if not os.path.exists("dbpedia_csv"):
    print("Downloading dbpedia dataset...")
    download_dbpedia()
print("Creating dataset")
word_dict = build_word_dict()
vocabulary_size = len(word_dict)
x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15)
train_x = np.array(train_x)
valid_x = np.array(valid_x)
train_y = np.array(train_y)
valid_y = np.array(valid_y)

print("train and valid datasets created ...")
print("train x: {}, x[0]: {}, type:{}".format(train_x.shape, train_x[0], type(train_x[0])))
print("valid x: {}".format(np.shape(valid_x)))
print("train y: {}".format(np.shape(train_y)))
#%% Defining the model
"""
This section implements a function that creates WordCNN
model. Some constants are needed to the model work
"""
embedding_size = 128
num_filters = 100
filter_sizes = [3, 4, 5]
num_class = 14


def word_cnn_model_create(embedding_size=128,
                          num_filters=100,
                          filter_sizes=[3, 4, 5],
                          num_classes=14,
                          document_max_len=100):
    x = tf.keras.Input(shape=(100, ))
    embeddings = tf.keras.layers.Embedding(input_dim=vocabulary_size,
                                           output_dim=embedding_size,
                                           input_length=document_max_len,
                                           embeddings_initializer='uniform')(x)
    x_emb = tf.keras.layers.Reshape((100, 128, 1))(embeddings)
    pooled_outputs = []
    for filter_size in filter_sizes:
        conv = tf.keras.layers.Conv2D(input_shape=(None, 100, 128, 1),
                                      filters=num_filters,
                                      kernel_size=[filter_size, embedding_size],
                                      strides=(1, 1),
                                      padding="valid",
                                      activation="relu")(x_emb)
        pool = tf.keras.layers.MaxPooling2D(pool_size=[document_max_len - filter_size + 1, 1],
                                            strides=(1, 1),
                                            padding='valid')(conv)
        pooled_outputs.append(pool)

    h_pool = tf.keras.layers.concatenate(pooled_outputs)
    h_pool_flat = tf.keras.layers.Flatten()(h_pool)
    h_drop = tf.keras.layers.Dropout(rate=0.5)(h_pool_flat)
    output = tf.keras.layers.Dense(units=num_classes, activation="softmax")(h_drop)

    model = tf.keras.Model(inputs=x, outputs=output)
    return model


wordCNNModel = word_cnn_model_create(embedding_size=embedding_size,
                                     num_filters=num_filters,
                                     num_classes=num_class,
                                     filter_sizes=filter_sizes,
                                     document_max_len=WORD_MAX_LEN
                                     )

print("Model to train has structure: ")
wordCNNModel.summary()
# tf.keras.utils.plot_model(model=wordCNNModel, to_file="wordCNNModel.png", show_shapes=True, show_layer_names=True)

wordCNNModel.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                     loss=tf.keras.losses.sparse_categorical_crossentropy,
                     metrics=['acc'])
print("training started")
wordCNNModel.fit(x=train_x,
                 y=train_y,
                 batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS,
                 verbose=1)
print("training finished")
