AI-Generated Text Detector


An AI-generated text detector in SML project helps to identify whether the text or paragraph was written by a human or copied or inspired by an AI engine. The fundamental idea includes creating a detector on text features that compares the AI-generated content from human-written content.

The basic steps include the:
1.)Data Collection: In this step we collect a dataset with labeled examples of both human-written and AI-generated text.
2.)Feature Extraction: Extract features from text that may help identify AI generation.
3.)Model Training: Use a machine learning classifier (like Logistic Regression, Random Forest, or a neural network) to train on these features.

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import keras_nlp
import matplotlib.pyplot as plt

class Config:
    dataset_path = "path/to/dataset"
    output_dataset_path = "path/to/output"
    label = "generated"
    batch_size = 32
    model_name = "distilbert-base-uncased"
    epochs = 10
    is_training = True

config = Config()

train_df = pd.read_csv(f"{config.dataset_path}/train_essays.csv")
test_df = pd.read_csv(f"{config.dataset_path}/test_essays.csv")
train_prompts = pd.read_csv(f"{config.dataset_path}/train_prompts.csv")

train_df[config.label].value_counts().plot(kind="bar")
plt.title("Label Distribution")
plt.show()

external_dataset_1 = pd.read_csv("/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv")
external_dataset_1[config.label] = 1

external_dataset_2 = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")
external_dataset_2 = external_dataset_2.rename(columns={"label": config.label})

columns = ["text", config.label]
df = pd.concat([train_df[columns], external_dataset_1[columns], external_dataset_2])
df[config.label].value_counts().plot(kind="bar")
plt.title("Combined Label Distribution")
plt.show()

class_weights = compute_class_weight("balanced", classes=np.unique(df[config.label]), y=df[config.label])
class_weights = dict(enumerate(class_weights))

train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df[config.label])

def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, tf.convert_to_tensor(y)))
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=batch_size * 4)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df["text"], train_df[config.label], config.batch_size, "train")
valid_ds = make_dataset(valid_df["text"], valid_df[config.label], config.batch_size, "valid")

def get_model(config):
    encoder = keras_nlp.models.DistilBertBackbone.from_preset(config.model_name)
    encoder.trainable = False
    preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(config.model_name)
    
    inputs = keras.Input(shape=(), dtype=tf.string)
    x = preprocessor(inputs)
    x = encoder(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="text_classification_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

models = []
if config.is_training:
    model = get_model(config)
    model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=valid_ds,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint("model_auc.tf", monitor="val_auc", mode="max", save_best_only=True),
        ]
    )
    models.append(model)
else:
    for model_path in ["model_auc.tf"]:
        model = tf.keras.models.load_model(f"{config.output_dataset_path}/{model_path}")
        models.append(model)

test_ds = tf.data.Dataset.from_tensor_slices(test_df["text"]).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
predictions = np.mean([model.predict(test_ds).flatten() for model in models], axis=0)

sample_submission = pd.read_csv(f"{config.dataset_path}/sample_submission.csv")
sample_submission[config.label] = predictions
sample_submission.to_csv("submission.csv", index=False)

plt.figure()
sample_submission[config.label].plot(kind="kde", label="Predicted")
train_df[config.label].plot(kind="kde", label="Train")
plt.legend()
plt.title("Prediction vs Train Distribution")
plt.show()
