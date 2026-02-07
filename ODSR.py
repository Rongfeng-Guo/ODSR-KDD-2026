# -*- coding: utf-8 -*-
"""
ODSR.py
Unified "keep useful parts" version (code-as-truth):

- social state update: s^{(l+1)} = s_clean^{(l)}
- vector gate (dim-wise)
- micro-interact: m_u_item ⊙ s_orth
- geo evidence: e_cos, e_norm (broadcast scalars)
- risk-aware SSL:
    adaptive noise uses (1 - stopgrad(gate)) as magnitude mask (element-wise)
    direction: normalized s_orth
    gate is stop-grad in SSL path via detach()
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODSRLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # gate input: [m_u_item, s_orth, (m_u_item ⊙ s_orth), e_cos, e_norm]
        dim_input = dim + dim + dim + 2
        self.W_gate = nn.Linear(dim_input, dim)

        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, -0.5)

    def forward(
        self,
        h_i: torch.Tensor,
        s_u: torch.Tensor,
        adj_ui: torch.Tensor,
        adj_uu: torch.Tensor,
        ssl_mode: Optional[str],
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_u_new, s_clean, gate
        """
        # item-view anchor message
        m_u_item = self.dropout(torch.sparse.mm(adj_ui, h_i))
        # social message from social state
        m_u_social = self.dropout(torch.sparse.mm(adj_uu, s_u))

        # projection decomposition (aggregate-then-decompose)
        item_norm_sq = (m_u_item * m_u_item).sum(dim=1, keepdim=True) + 1e-7
        dot = (m_u_social * m_u_item).sum(dim=1, keepdim=True)
        proj_coeff = dot / item_norm_sq
        s_parallel = proj_coeff * m_u_item
        s_orth = m_u_social - s_parallel

        # geo evidence (broadcast scalars)
        social_norm = m_u_social.norm(dim=1, keepdim=True) + 1e-7
        item_norm = torch.sqrt(item_norm_sq)
        e_cos = dot / (social_norm * item_norm)     # [U,1]
        e_norm = torch.tanh(social_norm)            # [U,1]

        # micro-interact
        micro = m_u_item * s_orth                   # [U,D]

        gate_in = torch.cat([m_u_item, s_orth, micro, e_cos, e_norm], dim=1)
        gate = torch.sigmoid(self.W_gate(gate_in))  # vector gate [U,D]

        s_clean = s_parallel + gate * s_orth
        h_u_new = m_u_item + s_clean

        # risk-aware adaptive view: element-wise magnitude mask (1 - sg(g))
        if ssl_mode == "adaptive":
            g_det = gate.detach()
            mag = (1.0 - g_det) * eps              # [U,D]
            orth_norm = s_orth.norm(dim=1, keepdim=True) + 1e-10
            direction = s_orth / orth_norm
            noise = direction * mag
            h_u_new = h_u_new + noise

        return h_u_new, s_clean, gate


class ODSR(nn.Module):
    def __init__(self, data_obj, config: Dict, device: torch.device, ablation_flags: Optional[Dict] = None):
        super().__init__()
        self.device = device
        self.user_num = data_obj.user_num
        self.item_num = data_obj.item_num

        self.emb_size = int(config.get("emb_size", 64))
        self.n_layers = int(config.get("n_layer", 3))
        self.dropout_val = float(config.get("dropout", 0.0))

        self.reg = float(config["reg"])
        self.eps = float(config["eps"])
        self.ssl_reg = float(config["ssl_reg"])
        self.ssl_temp = float(config.get("tau", 0.2))

        # coalesce
        self.adj_uu = data_obj.adj_uu.coalesce()
        self.adj_ui = data_obj.adj_ui.coalesce()
        self.sparse_norm_adj = data_obj.norm_adj.coalesce()

        self.layers = nn.ModuleList([ODSRLayer(self.emb_size, self.dropout_val) for _ in range(self.n_layers)])

        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.user_num, self.emb_size))),
            "item_emb": nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.item_num, self.emb_size))),
        })

        # learnable weighted pooling
        self.alphas = nn.Parameter(torch.ones(self.n_layers + 1))

    def forward(self, current_ssl_mode: Optional[str] = None):
        h_u = self.embedding_dict["user_emb"]
        h_i = self.embedding_dict["item_emb"]

        # social buffer initialized as h_u^(0)
        s_u = h_u.clone()

        h_u_list = [h_u]
        h_i_list = [h_i]

        for k in range(self.n_layers):
            # LightGCN propagation on expanded bipartite norm_adj
            ego = torch.cat([h_u, h_i], 0)
            side = torch.sparse.mm(self.sparse_norm_adj, ego)
            _, h_i_new = torch.split(side, [self.user_num, self.item_num])

            layer_ssl = "adaptive" if current_ssl_mode == "adaptive" else None
            h_u_new, s_clean, _gate = self.layers[k](
                h_i=h_i,
                s_u=s_u,
                adj_ui=self.adj_ui,
                adj_uu=self.adj_uu,
                ssl_mode=layer_ssl,
                eps=self.eps,
            )

            # unified social-state update:
            # s^{(l+1)} = s_clean^{(l)}
            s_u = s_clean

            h_u, h_i = h_u_new, h_i_new
            h_u_list.append(h_u)
            h_i_list.append(h_i)

        w = F.softmax(self.alphas, dim=0)
        final_user = torch.sum(torch.stack(h_u_list, dim=1) * w.view(1, -1, 1), dim=1)
        final_item = torch.sum(torch.stack(h_i_list, dim=1) * w.view(1, -1, 1), dim=1)
        return final_user, final_item

    def compute_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        final_user, final_item = self.forward(current_ssl_mode=None)
        u_e = final_user[users]
        p_e = final_item[pos]
        n_e = final_item[neg]

        rec_loss = -torch.log(1e-8 + torch.sigmoid((u_e * p_e).sum(1) - (u_e * n_e).sum(1))).mean()
        reg_loss = self.reg * (u_e.norm(2) + p_e.norm(2) + n_e.norm(2))

        ssl_loss = torch.tensor(0.0, device=self.device)
        if self.ssl_reg > 0:
            # adaptive view vs clean view
            v1_u, _ = self.forward(current_ssl_mode="adaptive")
            v2_u, _ = self.forward(current_ssl_mode=None)

            v1 = F.normalize(v1_u[users], dim=1)
            v2 = F.normalize(v2_u[users], dim=1)

            pos_sc = torch.exp((v1 * v2).sum(1) / self.ssl_temp)
            ttl_sc = torch.matmul(v1, v2.t()).div(self.ssl_temp).exp().sum(1)
            ssl_loss = self.ssl_reg * (-torch.log(pos_sc / ttl_sc).mean())

        return rec_loss + reg_loss + ssl_loss
