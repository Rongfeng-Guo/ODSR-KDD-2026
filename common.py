# -*- coding: utf-8 -*-
"""
common.py
- GraphData: load train/test/social; build adj_ui, norm_adj, adj_uu (row/sym)
- seed_everything
- Safe-export CODE_Unified (alias to ODSR) without breaking GraphData import
"""

import os
import re
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import scipy.sparse as sp


def seed_everything(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_dataset_path(dataset: str) -> str:
    curr = os.path.dirname(os.path.abspath(__file__))

    folder = dataset
    if dataset.lower() == "douban-book":
        folder = "douban-book"
    elif dataset.lower() == "yelp2018":
        folder = "yelp2018"

    roots = []
    env_root = os.environ.get("SELFREC_DATA_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    roots.extend([
        os.path.join(curr, "dataset"),
        os.path.join(curr, "../dataset"),
        "./dataset",
    ])

    cand = [os.path.join(r, folder) for r in roots]
    for p in cand:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"❌ Cannot find dataset path among: {cand}")


def _safe_read_pairs(path: str) -> List[Tuple[str, str]]:
    """
    Only take first 2 columns: u, i
    supports: 'u i', 'u i rating', comma/space separated, etc.
    """
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            if len(parts) < 2:
                continue
            out.append((parts[0], parts[1]))
    return out


def _sp_to_tensor(coo: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    coo = coo.tocoo()
    idx = torch.LongTensor(np.vstack((coo.row, coo.col)))
    val = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse.FloatTensor(idx, val, shape).to(device).coalesce()


class GraphData:
    """
    Fields:
      - user_num, item_num
      - train_data: List[[u, i]]
      - test_dict: Dict[u, List[i]]
      - train_items: Dict[u, List[i]]
      - adj_ui: (U x I) binary sparse
      - norm_adj: (U+I x U+I) sym-normalized bipartite (LightGCN)  [NO SELF-LOOP]
      - adj_uu: (U x U) social normalized (row or sym) + self-loop
    """
    def __init__(self, dataset: str, device: torch.device, social_norm: str = "row", verbose: bool = True):
        self.dataset = dataset
        self.device = device
        self.social_norm = str(social_norm).lower().strip()
        assert self.social_norm in ("row", "sym"), "social_norm must be 'row' or 'sym'"
        self._load_data(verbose=verbose)

    def _load_data(self, verbose: bool = True):
        base_path = _resolve_dataset_path(self.dataset)

        train_path = os.path.join(base_path, "train.txt")
        test_path = os.path.join(base_path, "test.txt")

        trust_path = os.path.join(base_path, "trust.txt")
        links_path = os.path.join(base_path, "links.txt")
        if (not os.path.exists(trust_path)) and os.path.exists(links_path):
            trust_path = links_path

        raw_train = _safe_read_pairs(train_path)
        raw_test = _safe_read_pairs(test_path)
        raw_social = _safe_read_pairs(trust_path) if os.path.exists(trust_path) else []

        u_map: Dict[str, int] = {}
        i_map: Dict[str, int] = {}

        def get_id(m: Dict[str, int], k: str) -> int:
            if k not in m:
                m[k] = len(m)
            return m[k]

        self.train_data = [[get_id(u_map, u), get_id(i_map, i)] for u, i in raw_train]

        self.test_dict: Dict[int, List[int]] = {}
        for u, i in raw_test:
            # keep only seen-in-train users/items (exactly as tuning code)
            if u in u_map and i in i_map:
                uid, iid = u_map[u], i_map[i]
                self.test_dict.setdefault(uid, []).append(iid)

        self.user_num = len(u_map)
        self.item_num = len(i_map)

        if verbose:
            print(
                f"   📊 Loaded: {self.dataset} - {self.user_num} Users, {self.item_num} Items | social_norm={self.social_norm}",
                flush=True
            )

        self.train_items: Dict[int, List[int]] = {}
        for u, i in self.train_data:
            self.train_items.setdefault(u, []).append(i)

        rows = np.array([p[0] for p in self.train_data], dtype=np.int64)
        cols = np.array([p[1] for p in self.train_data], dtype=np.int64)

        # adj_ui (binary user->item)
        ui = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.user_num, self.item_num),
        )
        self.adj_ui = _sp_to_tensor(ui, self.device)

        # bipartite sym-norm on expanded graph (LightGCN)  [NO SELF-LOOP]
        R = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        R[rows, cols + self.user_num] = 1.0
        R[cols + self.user_num, rows] = 1.0
        adj = R.tocoo()

        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=rowsum != 0)
        d_inv[np.isinf(d_inv)] = 0.0
        norm_adj = sp.diags(d_inv).dot(adj).dot(sp.diags(d_inv)).tocoo()
        self.norm_adj = _sp_to_tensor(norm_adj, self.device)

        # social edges (undirected unique) + self-loop
        social_edges = set()
        for u, v in raw_social:
            if u in u_map and v in u_map:
                u_id, v_id = u_map[u], u_map[v]
                if u_id != v_id:
                    if u_id > v_id:
                        u_id, v_id = v_id, u_id
                    social_edges.add((u_id, v_id))

        if social_edges:
            e = list(social_edges)
            s_rows = [a for a, b in e] + [b for a, b in e]
            s_cols = [b for a, b in e] + [a for a, b in e]

            # self-loop
            s_rows += list(range(self.user_num))
            s_cols += list(range(self.user_num))

            A = sp.coo_matrix(
                (np.ones(len(s_rows), dtype=np.float32), (s_rows, s_cols)),
                shape=(self.user_num, self.user_num),
            )
            deg = np.array(A.sum(1)).flatten()

            if self.social_norm == "sym":
                d = np.power(deg, -0.5, where=deg != 0)
                d[np.isinf(d)] = 0.0
                normA = sp.diags(d).dot(A).dot(sp.diags(d)).tocoo()
            else:
                d = np.power(deg, -1.0, where=deg != 0)
                d[np.isinf(d)] = 0.0
                normA = sp.diags(d).dot(A).tocoo()

            self.adj_uu = _sp_to_tensor(normA, self.device)
        else:
            self.adj_uu = torch.sparse.FloatTensor(self.user_num, self.user_num).to(self.device).coalesce()

        if verbose:
            nnz = int(self.adj_uu._nnz())
            print(f"      Social edges(raw unique undirected): {len(social_edges)} | adj_uu nnz={nnz}", flush=True)


# ---- compatibility: scripts expect CODE_Unified in common.py ----
# IMPORTANT: do NOT let an ODSR import failure break GraphData import.
CODE_Unified = None
try:
    from ODSR import ODSR as CODE_Unified  # noqa
except Exception as e:
    print(f"⚠️  [common.py] Failed to import ODSR as CODE_Unified: {e}", flush=True)
