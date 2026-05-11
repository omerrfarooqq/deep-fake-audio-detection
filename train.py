import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import numpy as np
import os

from data_utils import AudioDataset
from model import CNNLSTM

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

def eval_model(model, loader, loss_fn, device, return_details=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_paths = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, paths = batch
                all_paths.extend(list(paths))
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None
    if return_details:
        return total_loss/total, correct/total, f1, auc, all_labels, all_probs, all_paths
    return total_loss/total, correct/total, f1, auc

def main(args):
    df = pd.read_csv(args.manifest)
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    train_csv = os.path.join(args.workdir, "train_manifest.csv")
    val_csv = os.path.join(args.workdir, "val_manifest.csv")
    test_csv = os.path.join(args.workdir, "test_manifest.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_ds = AudioDataset(train_csv)
    val_ds = AudioDataset(val_csv)
    test_ds = AudioDataset(test_csv, return_path=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    history = []
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc, val_f1, val_auc = eval_model(model, val_loader, loss_fn, device)
        metric_name = "auc" if val_auc is not None else "f1"
        metric_value = val_auc if val_auc is not None else val_f1
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} "
            f"auc={val_auc if val_auc is not None else 'na'}"
        )
        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(model.state_dict(), args.save_model)
            print(f"Saved best model by val_{metric_name}: {metric_value:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc if val_auc is not None else np.nan,
        })

    model.load_state_dict(torch.load(args.save_model, map_location=device))
    test_loss, test_acc, test_f1, test_auc, labels, probs, paths = eval_model(
        model, test_loader, loss_fn, device, return_details=True
    )
    artifacts_dir = args.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    history_path = args.history_csv or os.path.join(artifacts_dir, "training_history.csv")
    preds_path = args.predictions_csv or os.path.join(artifacts_dir, "test_predictions.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    pred_df = pd.DataFrame({
        "filepath": paths,
        "label": labels,
        "prob": probs,
        "pred": [1 if p > 0.5 else 0 for p in probs],
    })
    pred_df.to_csv(preds_path, index=False)
    print(
        f"Test: loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} "
        f"auc={test_auc if test_auc is not None else 'na'}"
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV manifest with filepath,label")
    p.add_argument("--workdir", default=".")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-model", default="best.pth")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--history-csv", default="")
    p.add_argument("--predictions-csv", default="")
    args = p.parse_args()
    main(args)
