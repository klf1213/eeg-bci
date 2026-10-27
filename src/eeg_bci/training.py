from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); total_loss = 0.0; targets = []; predictions = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); logits = model(x); loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * len(y)
        targets.extend(y.detach().cpu().numpy()); predictions.extend(logits.argmax(1).detach().cpu().numpy())
    return total_loss / len(loader.dataset), float(accuracy_score(targets, predictions))


@torch.inference_mode()
def evaluate(model, loader, criterion, device, return_attention=False):
    model.eval(); total_loss = 0.0; targets = []; predictions = []; probabilities = []; attention = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x, return_attention=True) if return_attention else model(x)
        logits, weights = output if return_attention else (output, None)
        total_loss += criterion(logits, y).item() * len(y)
        probs = torch.softmax(logits, dim=1)
        targets.extend(y.cpu().numpy()); predictions.extend(probs.argmax(1).cpu().numpy()); probabilities.extend(probs.cpu().numpy())
        if weights is not None:
            attention.extend(weights.cpu().numpy())
    result = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "targets": np.asarray(targets), "predictions": np.asarray(predictions),
        "probabilities": np.asarray(probabilities),
    }
    if attention:
        result["attention"] = np.asarray(attention)
    return result


def fit(model, train_loader, validation_loader, device, learning_rate, weight_decay,
        max_epochs, patience, checkpoint_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                        "val_macro_f1": validation["macro_f1"]})
        if validation["loss"] < best_loss:
            best_loss, stale = validation["loss"], 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: train loss={train_loss:.4f}, val acc={validation['accuracy']:.4f}")
        if stale >= patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return history
