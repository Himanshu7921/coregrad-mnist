import wandb
from typing import List
from config import config
from optimizer import Adam
from coregrad import Scalar
from test import test_model
from model import NeuralNetwork
from utils import one_hot_encode, load_data, save_model, test_accuracy, cross_entropy_with_logits_loss, pre_processing, get_batch

import random
from tqdm import tqdm
import numpy as np
import sys
sys.setrecursionlimit(100000)

def train_model(x_train: List[Scalar], y_train: List[Scalar], x_test: List[Scalar], y_test: List[Scalar], model: NeuralNetwork, optimizer: Adam):

    # wandb for experiment tracking
    wandb.init(
        project="coregrad-mnist",
        config=config
    )

    nn = model
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

        # lr = config["lr"] * (0.995 ** epoch) # exponential lr scheduler

        optimizer.zero_grad()

        # backward pass
        loss.backward()

        optimizer.step()

        # wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "loss": loss.data
        })

        if epoch % 2 == 0:
            x_test_batch, y_test_batch = get_batch(x_test, y_test, batch_size = config["evaluation_batch_size"])

            # test accuracy
            acc = test_accuracy(
                model=nn,
                x=x_test_batch,
                y=y_test_batch
            )

            # logging
            tqdm.write(
                f"Epoch [{epoch + 1}/{EPOCHS}] | "
                f"Loss: {loss.data:.6f} | "
                f"Acc: {acc:.2f}%"
            )
            
        # Test the model after every 10 epochs
        if epoch % 10 == 0 and epoch > 10:
            test_model(x_test = x_test_batch, y_test = y_test_batch)

            
        # Adding interval checkpoint saving of the model
        if epoch % config["save_interval"] == 0:
            save_model(model = nn, save_path = config["save_path"])

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

    # Define Optimizer
    optimizer = Adam(parameters = nn.parameters(), lr = config["lr"])

    # Load x_test, y_test, x_train, y_train
    x_train, y_train, x_test, y_test = load_data(dataset_path = config["dataset_path"])

    # keeping original labels
    y_train_labels = y_train
    y_test_labels = y_test

    # one-hot encode for training
    y_train = one_hot_encode([Scalar(y) for y in y_train])
    y_test = one_hot_encode([Scalar(y) for y in y_test])

    # Feed training images and label into the model
    nn = train_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test_labels, model = nn, optimizer = optimizer)

    # save the model
    save_model(model = nn, save_path = config["save_path"])

    # random evaluation batch
    x_test_batch, y_test_batch = get_batch(x_test, y_test_labels, batch_size = config["evaluation_batch_size"])

    # test accuracy
    acc = test_accuracy(
                    model=nn,
                    x=x_test_batch,
                    y=y_test_batch
                )
    # Test the model: it internally loads the model for testing
    test_model(x_test = x_test_batch, y_test = y_test_batch)
    print(f"Accuracy of the Trained model on 32 random samples from test set is: {acc:.4f}%")

if __name__ == "__main__":
    main()