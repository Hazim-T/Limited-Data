from matplotlib import pyplot as plt
import numpy as np
import os
import random


def show_images(images, labels, class_names, num_images=32):
    plt.figure(figsize=(12, 12))
    for i in range(min(num_images, len(images))):
        plt.subplot(6, 6, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        true_label = class_names[labels[i].argmax()]
        plt.title(true_label)
    plt.tight_layout()
    plt.show()


def plot_predictions(generator, model, class_names):
    images, labels = next(generator)
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    plt.figure(figsize=(15, 15))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

        true_label = class_names[true_classes[i]]
        predicted_label = class_names[predicted_classes[i]]

        color = 'blue' if predicted_classes[i] == true_classes[i] else 'red'

        # Display the predicted label in the chosen color
        plt.title(predicted_label, color=color)

    plt.tight_layout()
    plt.show()

def shrink_dataset(dataset_path, target_count):
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        if os.path.isdir(folder_path):
            image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                           file.lower().endswith(('.png', '.jpg'))]

            if len(image_files) > target_count:
                images_to_delete = random.sample(image_files, len(image_files) - target_count)
                for img_path in images_to_delete:
                    os.remove(img_path)
                    print(f"Deleted: {img_path}")

            print(folder_name, "Reduced to", min(len(image_files), target_count))