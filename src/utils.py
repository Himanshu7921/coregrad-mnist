from typing import List
from config import config
from coregrad import Scalar
from model import NeuralNetwork

import os
import random
import pickle
import struct
import numpy as np
import matplotlib.pyplot as plt


def softmax(logits: List[Scalar]):
    max_val = max([x.data for x in logits])
    # numerical stability
    exps = [(x - max_val).exp() for x in logits]
    sum_exps = sum(exps, Scalar(0.0))
    # probs
    probs = [e / sum_exps for e in exps]
    return probs

def flatten(x):
    flattened = []
    for batch in x:
        flat = []
        for row in batch:
            flat.extend(row)
        flattened.append(flat)
    return flattened


def normalize_image(images):
    normalized_images = []
    for image in images:
        norm_image = []
        for row in image:
            norm_row = []
            for pixel in row:
                norm_row.append(Scalar(pixel / 255.0))
            norm_image.append(norm_row)
        normalized_images.append(norm_image)
    return normalized_images

def pre_processing(x: np.ndarray):
    x = x / 255.0 # normalize
    # flatten
    B = x.shape[0]
    x = x.reshape(B, -1)
    return x


def cross_entropy_with_logits_loss(logits: List[Scalar], y_true: List[Scalar]):
    # NOTE: y_true must be a one hot encoded vector

    # Get the probability vectors
    y_probs = softmax(logits)

    # Compute the loss
    loss_term = []
    for yp, yt in zip(y_probs, y_true):
        loss_term.append(yt * yp.log())
    loss = -1 * sum(loss_term, Scalar(0.0))
    return loss

def one_hot_encode(labels: List[Scalar], num_classes: str = config["num_classes"]):
    one_hot = []
    for label in labels:
        vec = [0] * num_classes
        vec[label.data] = 1
        one_hot.append(vec)
    return one_hot


# MNIST Dataloading Script taken from: https://www.kaggle.com/code/shyam143/mnist-digit-recognition
def load_mnist_images(path):
    with open(path, "rb") as f:
        # Read Header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read Image Data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape
        images = image_data.reshape(num_images, rows, cols)
        return images

def load_mnist_labels(path):
    with open(path, "rb") as f:
        # Read Header
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read Labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_data(dataset_path: str):
    # Training images and labels
    train_images_pth = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_pth = os.path.join(dataset_path, "train-labels.idx1-ubyte")

    # Testing images and labels
    test_images_pth = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_pth = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")


    train_images = load_mnist_images(train_images_pth)
    train_labels = load_mnist_labels(train_labels_pth)

    test_images = load_mnist_images(test_images_pth)
    test_labels = load_mnist_labels(test_labels_pth)
    return train_images, train_labels, test_images, test_labels



def save_model(model, save_path):
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        "parameters": [
            p.data for p in model.parameters()
        ]
    }
    with open(
        os.path.join(save_path, "model.pkl"),
        "wb"
    ) as f:
        pickle.dump(checkpoint, f)
    print("[INFO] Model saved successfully.")

def load_model(load_path):
    model = NeuralNetwork(
        in_features=config["H"] * config["W"],
        hidden_dim=config["hidden_dim"],
        out_features=config["out_features"],
        n_layers=config["n_layers"]
    )
    
    with open(load_path, "rb") as f:
        checkpoint = pickle.load(f)
    for p, saved_p in zip(
        model.parameters(),
        checkpoint["parameters"]
    ):
        p.data = saved_p
    return model

def test_accuracy(model, x, y):
    correct = 0
    for xi, yi in zip(x, y):
        logits = model(xi)
        probs = softmax(logits)
        pred_class = max(
            range(len(probs)),
            key=lambda i: probs[i].data
        )
        if pred_class == yi:
            correct += 1
    return (correct / len(x)) * 100

def get_batch(x_test, y_test, batch_size = config["evaluation_batch_size"]):
        test_batch_indices = random.sample(
                    range(len(x_test)),
                    batch_size
        )

        x_test_batch = [
                    x_test[i]
                    for i in test_batch_indices
                ]

        y_test_batch = [
                    y_test[i]
                    for i in test_batch_indices
                ]

                # preprocess test batch
        x_test_batch = pre_processing(
                    np.array(x_test_batch)
                )

                # convert to Scalars
        x_test_batch = [
                    [Scalar(v) for v in sample]
                    for sample in x_test_batch
                ]
        return x_test_batch, y_test_batch

def plot_img(img, label):
    """
    Plots one image and it's corresponding Label
    """
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()



if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data(config["dataset_path"])
    # plot_img(train_images[0], train_labels[0])
    normalized_img = normalize_image(train_images[:10])
    flatten_img = flatten(normalized_img)
    print(flatten_img)


