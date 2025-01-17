import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, Sequential

num_skipped = 0
for folder_name in ("Negative", "Positive"):
    folder_path = os.path.join("ConcreteCrackImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
            continue
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Deleted {num_skipped} images")

image_size = (180, 180)
batch_size = 400

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "ConcreteCrackImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
saved = int(input("0 - train model, 1 - use saved model: "))

if saved == 0:
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=(image_size[0], image_size[1], 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(image_size[0], image_size[1], 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    epochs = 6
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save("saved_model/model.keras")

elif saved == 1:
    model = keras.models.load_model('saved_model/model.keras')
    for file in os.listdir("data/test"):
        pth = os.path.join("data/test", file)
        img = tf.keras.utils.load_img(
            pth, target_size=(image_size[0], image_size[1])
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
