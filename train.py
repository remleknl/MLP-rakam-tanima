import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
VAL_SIZE = 10000
SEED = 42


def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_size = len(full_train) - VAL_SIZE
    train_set, val_set = random_split(
        full_train, [train_size, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


def _run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(config_name, config, train_loader, val_loader, test_loader, epochs=EPOCHS):
    model = MLP(
        hidden_layers=config["hidden_layers"],
        use_dropout=config["use_dropout"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
    }

    print(f"\n{'='*60}")
    print(f"  {config['label']}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8}")
    print(f"{'-'*55}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | {vl_loss:>9.4f} | {vl_acc*100:>7.2f}%")

    elapsed = time.time() - start

    _, test_acc = _run_epoch(model, test_loader, criterion)
    print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Eğitim Süresi : {elapsed:.1f} saniye")

    torch.save(model.state_dict(), f"results/{config_name}.pt")
    return history, test_acc, elapsed, model
