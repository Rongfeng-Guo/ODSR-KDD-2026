# main.py
# -*- coding: utf-8 -*-
import argparse
import os
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim

from common import GraphData, seed_everything
from ODSR import ODSR
from utils import evaluate_full_ranking


COMMON: Dict[str, Any] = dict(
    seed=2026,
    lr=0.001,
    emb_size=64,
    n_layer=3,
    tau=0.20,
    batch_size=2048,
    eval_interval=2,
    patience=10,
    eval_batch=2048,
    metric="recall@20",
)

DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "douban-book": dict(reg=0.00, ssl_reg=0.58, eps=0.57, dropout=0.01, max_epoch=60),
    "epinions": dict(reg=0.00, ssl_reg=0.18, eps=0.46, dropout=0.14, max_epoch=30),
    "yelp2018": dict(reg=0.00, ssl_reg=0.43, eps=0.54, dropout=0.17, max_epoch=20),
}

CAP_CKPT = {
    "douban-book": "Douban-Book.pth",
    "epinions": "Epinions.pth",
    "yelp2018": "Yelp2018.pth",
}
LOW_CKPT = {
    "douban-book": "douban-book.pth",
    "epinions": "epinions.pth",
    "yelp2018": "yelp2018.pth",
}


def default_ckpt(dataset_key: str) -> str:
    return os.path.join("checkpoints", CAP_CKPT[dataset_key])


def default_out(dataset_key: str) -> str:
    return os.path.join("checkpoints", LOW_CKPT[dataset_key])


def load_state(model: torch.nn.Module, path: str, device: torch.device) -> None:
    if not path or (not os.path.exists(path)):
        raise FileNotFoundError(f"ckpt not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)


def save_state(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)


def train_one_run(data: GraphData, cfg: Dict[str, Any], device: torch.device, out_path: str) -> Tuple[float, Dict[str, float], int]:
    seed_everything(int(cfg["seed"]))

    model = ODSR(data, cfg, device, ablation_flags={}).to(device)
    opt = optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    users_arr = np.array([p[0] for p in data.train_data], dtype=np.int64)
    pos_arr = np.array([p[1] for p in data.train_data], dtype=np.int64)

    bs = int(cfg["batch_size"])
    n_batch = max(1, len(users_arr) // bs)  # drop tail

    best_val = -1e18
    best_metrics: Dict[str, float] = {}
    best_ep = -1
    bad = 0

    max_epoch = int(cfg["max_epoch"])
    eval_interval = int(cfg["eval_interval"])
    patience = int(cfg["patience"])
    metric_key = str(cfg["metric"]).lower().strip()

    for ep in range(1, max_epoch + 1):
        t0 = time.time()
        model.train()

        idx = np.random.permutation(len(users_arr))
        total = 0.0

        for bi in range(n_batch):
            b = idx[bi * bs:(bi + 1) * bs]
            u = torch.LongTensor(users_arr[b]).to(device)
            p = torch.LongTensor(pos_arr[b]).to(device)
            n = torch.randint(0, data.item_num, (len(b),), device=device)

            loss = model.compute_loss(u, p, n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        avg = total / max(1, n_batch)
        print(f"epoch {ep}/{max_epoch} loss={avg:.6f} time={time.time() - t0:.1f}s", flush=True)

        if ep % eval_interval != 0:
            continue

        metrics = evaluate_full_ranking(model, data, device, k_list=(10, 20), eval_batch=int(cfg["eval_batch"]))
        val = float(metrics.get(metric_key, 0.0))
        print(
            f"eval r20={metrics['recall@20']:.6f} n20={metrics['ndcg@20']:.6f} monitor={metric_key}={val:.6f}",
            flush=True
        )

        if val > best_val:
            best_val = val
            best_metrics = metrics
            best_ep = ep
            bad = 0
            save_state(model, out_path)
            print(f"save {out_path} best={best_val:.6f} ep={best_ep}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop patience={patience}", flush=True)
                break

    return float(best_val), best_metrics, int(best_ep)


@torch.no_grad()
def eval_ckpt(data: GraphData, cfg: Dict[str, Any], device: torch.device, ckpt_path: str) -> Dict[str, float]:
    seed_everything(int(cfg["seed"]))
    model = ODSR(data, cfg, device, ablation_flags={}).to(device)
    load_state(model, ckpt_path, device)
    model.eval()
    return evaluate_full_ranking(model, data, device, k_list=(10, 20), eval_batch=int(cfg["eval_batch"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="douban-book", help="douban-book / epinions / yelp2018")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--social_norm", type=str, default="row", choices=["row", "sym"])

    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint to load for eval (or resume manually)")
    ap.add_argument("--out", type=str, default=None, help="output checkpoint path for training")

    # optional overrides
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--reg", type=float, default=None)
    ap.add_argument("--ssl_reg", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--emb_size", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--max_epoch", type=int, default=None)
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--eval_batch", type=int, default=None)
    ap.add_argument("--metric", type=str, default=None)

    args = ap.parse_args()
    ds = args.dataset.lower().strip()
    if ds not in DATASET_CFG:
        raise ValueError(f"supported: {list(DATASET_CFG.keys())}")

    cfg = dict(COMMON)
    cfg.update(DATASET_CFG[ds])
    cfg["social_norm"] = args.social_norm

    for k in [
        "seed", "lr", "reg", "ssl_reg", "eps", "dropout",
        "batch_size", "n_layer", "emb_size", "tau",
        "max_epoch", "eval_interval", "patience", "eval_batch", "metric",
    ]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    device = torch.device(args.device)

    ckpt_path = args.ckpt if args.ckpt else default_ckpt(ds)
    out_path = args.out if args.out else default_out(ds)

    mode = "train" if args.train else "eval"
    print("mode:", mode, flush=True)
    print("dataset:", args.dataset, flush=True)
    print("device:", args.device, flush=True)
    print("social_norm:", cfg["social_norm"], flush=True)
    if args.train:
        print("out:", out_path, flush=True)
    else:
        print("ckpt:", ckpt_path, flush=True)

    data = GraphData(args.dataset, device, social_norm=cfg["social_norm"], verbose=False)

    if args.train:
        best_val, best_metrics, best_ep = train_one_run(data, cfg, device, out_path)
        print("done", flush=True)
        print("best_epoch:", best_ep, flush=True)
        if best_metrics:
            print(
                f"best r20={best_metrics['recall@20']:.6f} n20={best_metrics['ndcg@20']:.6f} "
                f"r10={best_metrics['recall@10']:.6f} n10={best_metrics['ndcg@10']:.6f}",
                flush=True
            )
        print(f"best_{str(cfg['metric']).lower().strip()}={best_val:.6f}", flush=True)
    else:
        m = eval_ckpt(data, cfg, device, ckpt_path)
        print("eval", flush=True)
        print(
            f"r20={m['recall@20']:.6f} n20={m['ndcg@20']:.6f} r10={m['recall@10']:.6f} n10={m['ndcg@10']:.6f}",
            flush=True
        )


if __name__ == "__main__":
    main()
