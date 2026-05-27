import wandb
from typing import List
from config import config
from coregrad import Scalar
from model import NeuralNetwork
from utils import one_hot_encode, load_data, save_model, test_accuracy, cross_entropy_with_logits_loss, pre_processing

import random
from tqdm import tqdm
import numpy as np
import sys
sys.setrecursionlimit(100000)

def train_model(x_train: List[Scalar], y_train: List[Scalar], model: NeuralNetwork):

    # wandb for experiment tracking
    wandb.init(
        project="coregrad-mnist",
        config=config
    )

    nn = model

    lr = config["lr"]
    EPOCHS = config["epochs"]
    batch_size = config["batch_size"]
    alpha = config["alpha"]

    for epoch in tqdm(range(EPOCHS), desc="Training"):
        # random mini-batch sampling
        batch_indices = random.sample(
            range(len(x_train)),
            batch_size
        )

        x_batch = [x_train[i] for i in batch_indices]
        y_batch = [y_train[i] for i in batch_indices]

        # preprocess ONLY current batch
        x_batch = pre_processing(np.array(x_batch))

        # convert batch to Scalars
        x_batch = [
            [Scalar(v) for v in sample]
            for sample in x_batch
        ]
        y_batch = [y_train[i] for i in batch_indices]

        # forward pass
        y_preds = [nn(x) for x in x_batch]

        batch_losses = []

        for yp, yt in zip(y_preds, y_batch):
            loss = cross_entropy_with_logits_loss(
                logits=yp,
                y_true=yt
            )
            batch_losses.append(loss)

        loss = sum(batch_losses, Scalar(0.0)) * (1 / batch_size)

        # L2 Regularization
        reg_loss = Scalar(0.0)
        for p in nn.parameters():
            reg_loss += p * p
        reg_loss = reg_loss * alpha
        loss += reg_loss

        lr = 1.0 - 0.4 * epoch/100

        # optimizer.zero_grad()
        for p in nn.parameters():
            p.grad = 0

        # backward pass
        loss.backward()

        # optimizer.step()
        for p in nn.parameters():
            p.data -= p.grad * lr

        # wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "loss": loss.data
        })

        if epoch % 2 == 0:
            tqdm.write(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Loss: {loss.data:.6f} |"
            f"lr [{lr}]"
        )
    wandb.finish()
    return nn

def main():
    # Define Model
    nn = NeuralNetwork(
        in_features = config["H"] * config["W"],
        hidden_dim = config["hidden_dim"],
        out_features = config["out_features"],
        n_layers = config["n_layers"]
    )

    # Load x_test, y_test, x_train, y_train
    x_train, y_train, x_test, y_test = load_data(dataset_path = config["dataset_path"])

    # Preprocess to match Scalar Datastructure
    y_train = one_hot_encode([Scalar(x) for x in y_train])
    y_test = one_hot_encode([Scalar(x) for x in y_test])

    # Feed training images and label into the model
    nn = train_model(x_train = x_train, y_train = y_train, model = nn)

    # save the model
    save_model(model = nn, save_path = config["save_path"])

    # test accuracy of the model
    acc = test_accuracy(model = nn, x = x_test, y = y_test)
    print(f"Accuracy of the model on test set is: {acc:.4f}%")

if __name__ == "__main__":
    main()