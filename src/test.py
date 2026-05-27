import os

from config import config
from utils import load_model, softmax
from model import NeuralNetwork


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
        model=model,
        load_path=os.path.join(
            config["save_path"],
            "model.pkl"
        )
    )
    correct = 0
    predictions = []
    for x, y in zip(x_test, y_test):
        # forward pass
        logits = model(x)
        # softmax over single sample logits
        y_probs = softmax(logits)
        # argmax prediction
        pred_class = max(
            range(len(y_probs)),
            key=lambda i: y_probs[i].data
        )
        predictions.append(pred_class)
        if pred_class == y:
            correct += 1
    acc = (correct / len(y_test)) * 100
    print(f"Predictions: {predictions[:10]}")
    print(f"Actual Labels: {y_test[:10]}")
    print(f"Test Accuracy: {acc:.2f}%")