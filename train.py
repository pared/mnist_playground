import os
from collections import defaultdict

import torch
import torch.nn as nn
import yaml
from dvclive import Live
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

logger = Live("training_logs", summary=False)
logger._html = False

if torch.cuda.is_available():
    torch.device("cuda")
else:
    torch.device("cpu")
    torch.set_num_threads(int(os.cpu_count() / 2))


def avg(nums):
    return sum(nums) / (len(nums))


def load_params():
    with open("params.yaml") as fd:
        return yaml.safe_load(fd)


def prepare_data_loaders(batch_size, num_workers):
    train_data_path = os.path.join("data", "train")
    test_data_path = os.path.join("data", "test")
    transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    train_dataset = ImageFolder(root=train_data_path, transform=transform)
    test_dataset = ImageFolder(root=test_data_path, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    print(
        f"Prepared the datasets, train size: '{len(train_dataset)}', test size: '{len(test_dataset)}'"
    )
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(8 * 7 * 7, 10)
        print("Model created")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

def write_confusion_matrix_data(actual, predicted, filename, mode="w"):
    import csv
    with open(filename, mode, encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=["actual", "predicted"])
        writer.writeheader()
        for actual, predicted in zip(actual, predicted):
            writer.writerow({"actual": actual, "predicted": predicted})

def write_confusion_matrix_image(actual, predicted, filename):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual, predicted)
        import matplotlib.pyplot as plt
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Confusion matrix')
        plt.savefig(filename)


def train(model, loss_func, optimizer, num_epochs, train_loader, test_loader):
    def single_epoch():
        # train
        train_cf_actual = []
        train_cf_predicted = []
        train_losses = []
        train_accuracies = []
        for images, labels in train_loader:
            model.train()
            b_x = Variable(images)
            b_y = Variable(labels)
            output = model(b_x)
            train_loss = loss_func(output, b_y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

            model.eval()
            with torch.no_grad():
                test_output = model(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()

                train_accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                train_accuracies.append(train_accuracy)

                train_cf_actual.extend(labels.tolist())
                train_cf_predicted.extend(pred_y.tolist())

        write_confusion_matrix_data(actual = train_cf_actual, predicted=train_cf_predicted, filename="train_cm.csv")
        write_confusion_matrix_image(actual=train_cf_actual, predicted=train_cf_predicted, filename="train.jpg")

        # test
        test_cf_actual = []
        test_cf_predicted = []
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for images, labels in test_loader:
                test_output = model(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                test_loss = loss_func(test_output, labels)
                test_losses.append(test_loss.item())

                test_accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                test_accuracies.append(test_accuracy)

                test_cf_actual.extend(labels.tolist())
                test_cf_predicted.extend(pred_y.tolist())

        write_confusion_matrix_data(actual=test_cf_actual, predicted=test_cf_predicted, filename="test_cm.csv")
        write_confusion_matrix_image(actual=test_cf_actual, predicted=test_cf_predicted, filename="test.jpg")

        avg_train_loss = avg(train_losses)
        avg_train_accuracy = avg(train_accuracies)
        avg_test_loss = avg(test_losses)
        avg_test_accuracy = avg(test_accuracies)
        return avg_train_loss, avg_train_accuracy, avg_test_loss, avg_test_accuracy

    def log(name, value):
        logger.log(name, value)

    pbar = tqdm(range(num_epochs), position=0, desc="epoch")
    for epoch in pbar:
        metrics = {}

        train_loss, train_acc, test_loss, test_acc = single_epoch()

        metrics["train_loss"] = train_loss
        metrics["test_loss"] = test_loss
        metrics["train_accuracy"] = train_acc
        metrics["test_accuracy"] = test_acc

        for key, value in metrics.items():
            log(key, value)
        logger.next_step()

        pbar.set_description(
            f"train loss: '{train_loss:.4f}', test_loss: '{test_loss:.4f}, train_acc: '{train_acc:.4f}', test_acc: '{test_acc:.4f}'"
        )
    return metrics


def get_optimizer(model, name, learning_rate):
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif name == "adamax":
        return optim.Adamax(model.parameters(), lr=learning_rate)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    params = load_params()
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]
    optimizer_name = params["optimizer"]

    model = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate)

    train_loader, test_loader = prepare_data_loaders(batch_size, 0)
    metrics = train(model, loss_func, optimizer, num_epochs, train_loader, test_loader)

    torch.save(model.state_dict(), "model")

    with open("metrics.yml", "w+") as fd:
        yaml.safe_dump(metrics, fd)
