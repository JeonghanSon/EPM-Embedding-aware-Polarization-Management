# src/models/sgcn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SignedConv
from torch_geometric.utils import structured_negative_sampling


class SGCNForSignLinkClass(nn.Module):
    """
    Minimal SGCN for 3-class signed link prediction + (optional) embedding losses.

    - Encoder: SignedConv stack -> z (node embeddings)
    - Head   : Linear([z_u || z_v]) -> 3-class logits (+ / - / none)
    - Loss   : 3-class NLL + lambda * (pos_emb_loss + neg_emb_loss)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        device: str = "cpu",
        triplet_lambda: float = 1.0,   # corresponds to lambda in the original SGCN paper/code
    ):
        super().__init__()
        self.device = torch.device(device)
        self.triplet_lambda = float(triplet_lambda)

        # ----- Encoder (SignedConv stack) -----
        self.conv1 = SignedConv(
            in_channels,
            hidden_channels // 2,
            first_aggr=True,
        )
        self.convs = nn.ModuleList(
            [
                SignedConv(
                    hidden_channels // 2,
                    hidden_channels // 2,
                    first_aggr=False,
                )
                for _ in range(max(0, num_layers - 1))
            ]
        )

        # ----- 3-class head -----
        self.lin = nn.Linear(2 * hidden_channels, 3)

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    # ------------------------------------------------------------
    # Encoder: x, pos/neg edges -> z
    # ------------------------------------------------------------
    def encode(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)

        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z  # [N, H]

    def forward(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(x, pos_edge_index, neg_edge_index)

    # ------------------------------------------------------------
    # Prediction head
    # ------------------------------------------------------------
    def _pair_logits(self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        z = z.to(self.device)
        src = src.to(self.device)
        dst = dst.to(self.device)

        zu = z[src]
        zv = z[dst]

        # Symmetric pair representation for undirected signed edges.
        # This gives the same representation for (u, v) and (v, u).
        emb_pair = torch.cat(
            [
                torch.abs(zu - zv),
                zu * zv,
            ],
            dim=1,
        )  # [E, 2H]

        return self.lin(emb_pair)  # [E, 3]

    def discriminate_z(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        logits = self._pair_logits(z, src, dst)
        return F.log_softmax(logits, dim=1)

    # ------------------------------------------------------------
    # 3-class NLL loss
    # ------------------------------------------------------------
    def nll_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        non_edge_index: torch.Tensor,
        neg_wt: float = 1.0,
        null_wt: float = 1.0,
    ) -> torch.Tensor:
        z = z.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)
        non_edge_index = non_edge_index.to(self.device)

        total_loss = 0.0
        count = 0

        if pos_edge_index.size(1) > 0:
            target = pos_edge_index.new_full((pos_edge_index.size(1),), 0, dtype=torch.long)
            total_loss += F.nll_loss(self.discriminate_z(z, pos_edge_index), target)
            count += 1

        if neg_edge_index.size(1) > 0:
            target = neg_edge_index.new_full((neg_edge_index.size(1),), 1, dtype=torch.long)
            total_loss += neg_wt * F.nll_loss(self.discriminate_z(z, neg_edge_index), target)
            count += 1

        if non_edge_index.size(1) > 0:
            target = non_edge_index.new_full((non_edge_index.size(1),), 2, dtype=torch.long)
            total_loss += null_wt * F.nll_loss(self.discriminate_z(z, non_edge_index), target)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / float(count)

    # ------------------------------------------------------------
    # Embedding losses (core component in the original SGCN objective)
    # ------------------------------------------------------------
    def _pos_embedding_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Original SGCN-style loss:
          out  = ||z_i - z_j||^2 - ||z_i - z_k||^2
          loss = clamp(out, min=0).mean()

        structured_negative_sampling returns (i, j, k) where (i, j) is a positive edge
        and (i, k) is a sampled non-edge.
        """
        if pos_edge_index.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # i, j: positive edge / k: random negative for i
        i, j, k = structured_negative_sampling(pos_edge_index, num_nodes=z.size(0))
        i = i.to(self.device)
        j = j.to(self.device)
        k = k.to(self.device)

        dij = (z[i] - z[j]).pow(2).sum(dim=1)
        dik = (z[i] - z[k]).pow(2).sum(dim=1)
        out = dij - dik
        return torch.clamp(out, min=0.0).mean()

    def _neg_embedding_loss(self, z: torch.Tensor, neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Original SGCN-style loss:
          out  = ||z_i - z_k||^2 - ||z_i - z_j||^2
          loss = clamp(out, min=0).mean()

        (i, j): negative edge, k: sampled non-edge for i w.r.t. neg_edge_index
                (i, k) is not in the negative edge set.
        """
        if neg_edge_index.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        i, j, k = structured_negative_sampling(neg_edge_index, num_nodes=z.size(0))
        i = i.to(self.device)
        j = j.to(self.device)
        k = k.to(self.device)

        dik = (z[i] - z[k]).pow(2).sum(dim=1)
        dij = (z[i] - z[j]).pow(2).sum(dim=1)
        out = dik - dij
        return torch.clamp(out, min=0.0).mean()

    def embedding_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        z = z.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)

        pos_l = self._pos_embedding_loss(z, pos_edge_index)
        neg_l = self._neg_embedding_loss(z, neg_edge_index)
        return pos_l + neg_l

    # ------------------------------------------------------------
    # Total loss = NLL + lambda * embedding_loss
    # ------------------------------------------------------------
    def loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        non_edge_index: torch.Tensor,
        neg_wt: float = 1.0,
        null_wt: float = 1.0,
    ) -> torch.Tensor:
        nll = self.nll_loss(
            z,
            pos_edge_index,
            neg_edge_index,
            non_edge_index,
            neg_wt=neg_wt,
            null_wt=null_wt,
        )

        if self.triplet_lambda <= 0.0:
            return nll

        emb = self.embedding_loss(z, pos_edge_index, neg_edge_index)
        return nll + (self.triplet_lambda * emb)
