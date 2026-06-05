"""Path D: a binary Tree-LSTM over the parse TREE (structure handed in).

It does NOT have to parse — only to compute over a known structure. Leaves embed
their operand value with the same log-magnitude+sign encoding used in M3; internal
nodes combine children with an operator-conditioned binary Tree-LSTM cell
(Tai et al. 2015, N-ary variant: one forget gate per child). The root vector feeds
the SAME RegressionHead / ErrorHead as path C.

`forward` is **depth-batched**: all nodes at the same depth (across the whole
minibatch) are processed in one vectorized cell call, so the heavy compute is a
handful of big matmuls per minibatch instead of one tiny op per node. This is what
makes D tractable (the naive per-node recursion is ~50x slower and Python-bound).
`forward_recursive` is the simple reference; a sanity check asserts they agree.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .exprgen import SYMS
from .models import RegressionHead, ErrorHead

OP2IDX = {s: i for i, s in enumerate(SYMS)}   # + - * / -> 0..3


class BinaryTreeLSTM(nn.Module):
    def __init__(self, h_dim=128, op_dim=8, leaf_scale=4.0):
        super().__init__()
        self.h_dim = h_dim
        self.leaf_scale = leaf_scale   # log1p(|operand|)/leaf_scale; operands in [-20,20]
        self.leaf = nn.Linear(2, 2 * h_dim)
        self.op_emb = nn.Embedding(len(SYMS), op_dim)
        self.gate = nn.Linear(2 * h_dim + op_dim, 5 * h_dim)   # i, f_l, f_r, o, u
        self.reg_head = RegressionHead(h_dim)
        self.err_head = ErrorHead(h_dim)

    # ---- batched cell pieces ----
    def _leaf_batch(self, values, device):
        v = torch.as_tensor(values, dtype=torch.float32, device=device)
        feat = torch.stack([torch.sign(v), torch.log1p(torch.abs(v)) / self.leaf_scale], dim=1)
        h, c = self.leaf(feat).chunk(2, dim=1)
        return torch.tanh(h), c

    def _cell_batch(self, hl, cl, hr, cr, op_ids):
        g = self.gate(torch.cat([hl, hr, self.op_emb(op_ids)], dim=1))
        h = self.h_dim
        i = torch.sigmoid(g[:, 0:h])
        fl = torch.sigmoid(g[:, h:2 * h])
        fr = torch.sigmoid(g[:, 2 * h:3 * h])
        o = torch.sigmoid(g[:, 3 * h:4 * h])
        u = torch.tanh(g[:, 4 * h:5 * h])
        c = i * u + fl * cl + fr * cr
        return o * torch.tanh(c), c

    def forward(self, trees, device):
        # group nodes by depth; assign global ids in ascending-depth order so that
        # children always precede parents in the running state tensor.
        by_depth = {}

        def collect(node):
            by_depth.setdefault(node.depth, []).append(node)
            if node.kind == "op":
                collect(node.left)
                collect(node.right)

        for t in trees:
            collect(t)

        depths = sorted(by_depth)
        gid = {}
        for d in depths:
            for node in by_depth[d]:
                gid[id(node)] = len(gid)

        h_all, c_all = self._leaf_batch([n.value for n in by_depth[0]], device)
        for d in depths:
            if d == 0:
                continue
            internal = by_depth[d]
            li = [gid[id(n.left)] for n in internal]
            ri = [gid[id(n.right)] for n in internal]
            op_ids = torch.tensor([OP2IDX[n.op] for n in internal], device=device)
            hn, cn = self._cell_batch(h_all[li], c_all[li], h_all[ri], c_all[ri], op_ids)
            h_all = torch.cat([h_all, hn], dim=0)
            c_all = torch.cat([c_all, cn], dim=0)

        H = h_all[[gid[id(t)] for t in trees]]
        return self.reg_head(H), self.err_head(H)

    # ---- recursive reference (correctness check only) ----
    def _node_state(self, node, device):
        if node.kind == "leaf":
            h, c = self._leaf_batch([node.value], device)
            return h[0], c[0]
        hl, cl = self._node_state(node.left, device)
        hr, cr = self._node_state(node.right, device)
        op_ids = torch.tensor([OP2IDX[node.op]], device=device)
        h, c = self._cell_batch(hl[None], cl[None], hr[None], cr[None], op_ids)
        return h[0], c[0]

    def forward_recursive(self, trees, device):
        H = torch.stack([self._node_state(t, device)[0] for t in trees])
        return self.reg_head(H), self.err_head(H)
