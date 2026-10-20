"""OPTIONAL: TensorFlow/Keras deep-learning model on the same preprocessing.

pip install tensorflow   then   python -m src.train_keras
The sklearn MLP in train.py is the default so the project runs without TensorFlow.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .config import SEED
from .preprocessing import build_preprocessor
from .train import load_data, split


def main():
    tf.keras.utils.set_random_seed(SEED)
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_data())
    prep = build_preprocessor()
    A_train = prep.fit_transform(X_train).values.astype("float32")
    A_val, A_test = (prep.transform(x).values.astype("float32") for x in (X_val, X_test))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(A_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(A_train, y_train, validation_data=(A_val, y_val), epochs=60, batch_size=256, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_auc", mode="max")])
    print("Keras test AUC:", roc_auc_score(y_test, model.predict(A_test).ravel()))
    model.save("models/keras_churn.keras")


if __name__ == "__main__":
    main()
