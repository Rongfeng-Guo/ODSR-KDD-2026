# -*- coding: utf-8 -*-
import math
from typing import Dict, Tuple, Iterable

import numpy as np
import torch


def _ndcg_at_k(ranked_items: np.ndarray, gt_set: set, k: int) -> float:
    dcg = 0.0
    upto = min(k, len(ranked_items))
    for idx in range(upto):
        if ranked_items[idx] in gt_set:
            dcg += 1.0 / math.log2(idx + 2)
    idcg = 0.0
    upto_i = min(k, len(gt_set))
    for i in range(upto_i):
        idcg += 1.0 / math.log2(i + 2)
    return (dcg / idcg) if idcg > 0 else 0.0


@torch.no_grad()
def evaluate_full_ranking(
    model,
    data,
    device: torch.device,
    k_list: Tuple[int, ...] = (10, 20),
    eval_batch: int = 2048,
) -> Dict[str, float]:
    model.eval()
    u_emb, i_emb = model.forward(current_ssl_mode=None)

    test_users = list(data.test_dict.keys())
    recall = {k: [] for k in k_list}
    ndcg = {k: [] for k in k_list}
    train_items = data.train_items
    max_k = max(k_list)

    for st in range(0, len(test_users), eval_batch):
        b_u = test_users[st:st + eval_batch]
        u_idx = torch.LongTensor(b_u).to(device)
        scores = torch.matmul(u_emb[u_idx], i_emb.t())

        # mask training items
        for row, uid in enumerate(b_u):
            seen = train_items.get(uid, [])
            if seen:
                scores[row, seen] = -1e9

        _, topk = torch.topk(scores, k=max_k, dim=1)
        topk = topk.cpu().numpy()

        for row, uid in enumerate(b_u):
            gt = data.test_dict.get(uid, [])
            if not gt:
                continue
            gt_set = set(gt)
            ranked = topk[row]
            for k in k_list:
                pred_k = ranked[:k]
                hits = sum(1 for it in pred_k if it in gt_set)
                recall[k].append(hits / max(1, len(gt_set)))
                ndcg[k].append(_ndcg_at_k(pred_k, gt_set, k))

    out = {}
    for k in k_list:
        out[f"recall@{k}"] = float(np.mean(recall[k])) if recall[k] else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcg[k])) if ndcg[k] else 0.0
    return out
