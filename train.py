import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from data import load_data, tf_dataset
from architecture import build_unet

if __name__ == "__main__":
    """Seeding the environment"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """To call the dataset"""
    path = "E:/MIC_Projects/U-Net_Implementation/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(f"Dataset: Train:{len(train_x)} - Valid:{len(valid_x)} - Test:{len(test_x)}")

    """Hyperparameters"""
    shape = (256, 256, 3)
    num_classes = 3
    lr = 1e-4
    batch_size = 8
    epochs = 10


    """Model"""
    model = build_unet(shape, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr))

    train_dataset=tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset=tf_dataset(valid_x, valid_y, batch=batch_size)

    #number of iteration for the training data and the valid dataset in each epoch
    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size

    callbacks = [ModelCheckpoint("model.h5", verbose=1,save_best_model=True),
                 ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
                 #if continous 3 epochs does not decrease validation loss,then it should decrease the learning rate by the factor of 0.1
                 EarlyStopping(monitor="val_loss", patience=5, verbose=1)
                 #if after certain epochs validation loss does nor decrease, it'll stop the training other model will start overfitting
                 ]

    #training
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_data=valid_dataset,
              validation_steps=valid_steps,
              epochs=epochs,
              callbacks=callbacks)