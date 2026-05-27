import os
import random
import numpy as np
from config import config
from coregrad import Scalar
from utils import load_model
import matplotlib.pyplot as plt
from model import NeuralNetwork
from utils import load_data, get_batch, predict, pre_processing


def test_model(x_test, y_test):
    # recreate architecture
    model = NeuralNetwork(
        in_features=config["H"] * config["W"],
        hidden_dim=config["hidden_dim"],
        out_features=config["out_features"],
        n_layers=config["n_layers"]
    )
    # load trained weights
    model = load_model(
        load_path=os.path.join(
            config["save_path"],
            "model.pkl"
        )
    )
    correct = 0
    predictions = []
    for x, y in zip(x_test, y_test):
        # forward pass
        pred_class = predict(x, model)
        predictions.append(pred_class)
        if pred_class == y:
            correct += 1
    acc = (correct / len(y_test)) * 100
    print(f"Predictions: {predictions[:25]}")
    print(f"Actual Labels: {[int(y) for y in y_test[:25]]}")
    print(f"Test Accuracy: {acc:.2f}%")

def visualize_prediction(img, y_label):
    def plot_img(img, label):
        """
        Plots one image and it's corresponding Label
        """
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {label} | Actual: {y_label}")
        plt.show()

    model = NeuralNetwork(
        in_features=config["H"] * config["W"],
        hidden_dim=config["hidden_dim"],
        out_features=config["out_features"],
        n_layers=config["n_layers"]
    )
    # load trained weights
    model = load_model(
        load_path=os.path.join(
            config["save_path"],
            "model.pkl"
        )
    )
    x = img / 255.0
    x = x.reshape(-1)
    x = [Scalar(float(v)) for v in x]
    predicted_class = predict(x, model)
    plot_img(img, predicted_class)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data(dataset_path = config["dataset_path"])
    x_test_batch, y_test_batch = get_batch(x_test, y_test = y_test, batch_size = config["evaluation_batch_size"])
    test_model(x_test_batch, y_test_batch)
    idx = random.choice(range(len(x_test)))
    visualize_prediction(img = x_test[idx], y_label = y_test[idx])