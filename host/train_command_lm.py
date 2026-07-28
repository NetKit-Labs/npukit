#!/usr/bin/env python3
"""Train tiny command-phrase LM (D=32, T=32, L=6) on Google Speech Commands vocab.

Robot multi-word phrases are synthesized from the GSC word list (and a few
robot glue words). The model is a causal LM: next-token prediction with
teacher forcing. Deploy export matches ``npukit_command_lm.py`` (int8 GEMM,
float Softmax/RMSNorm/GELU).

Usage:
  python3 host/train_command_lm.py
  python3 host/train_command_lm.py --phrases 20000 --epochs 8
  python3 host/gsc_commands.py --download   # optional full audio set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import gsc_commands as gsc
import npukit_command_lm as lm
import npukit_transformer as nt

HOST_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = HOST_DIR / "command_lm_weights.npz"
SAMPLE_PATH = HOST_DIR / "command_lm_sample.npz"


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, qmin: float, qmax: float):
        return torch.clamp(torch.round(x * scale), qmin, qmax) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


def fake_quant(
    x: torch.Tensor,
    scale: float | torch.Tensor,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        s = scale.to(dtype=x.dtype, device=x.device)
        return torch.clamp(torch.round(x * s), qmin, qmax) / s
    return FakeQuantSTE.apply(x, float(scale), qmin, qmax)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, mlp_h: int) -> None:
        super().__init__()
        self.d = d
        self.mlp_h = mlp_h
        self.wq = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wk = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wv = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wo = nn.Parameter(torch.randn(d, d) * 0.12)
        self.w1 = nn.Parameter(torch.randn(d, mlp_h) * 0.12)
        self.w2 = nn.Parameter(torch.randn(mlp_h, d) * 0.12)
        self.gamma1 = nn.Parameter(torch.ones(d))
        self.gamma2 = nn.Parameter(torch.ones(d))
        self.sw_wq = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_wk = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_wv = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_wo = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_w1 = nn.Parameter(torch.full((mlp_h,), 64.0), requires_grad=False)
        self.sw_w2 = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)

    @staticmethod
    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x / rms) * gamma


class TinyCommandLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d, t, mh, v = lm.LM_D, lm.LM_T, lm.LM_MLP, lm.VOCAB_SIZE
        self.t = t
        self.d = d
        self.mlp = mh
        self.vocab = v
        self.n_layers = lm.N_LAYERS
        self.emb = nn.Embedding(v, d, padding_idx=gsc.PAD_ID)
        nn.init.normal_(self.emb.weight, std=0.12)
        with torch.no_grad():
            self.emb.weight[gsc.PAD_ID].zero_()
        self.pos = nn.Parameter(torch.zeros(t, d))
        self.blocks = nn.ModuleList(TransformerBlock(d, mh) for _ in range(self.n_layers))
        self.n_cmd = len(gsc.ROBOT_COMMANDS)
        self.lm_head = nn.Parameter(torch.randn(d, v) * 0.12)
        self.w_cmd = nn.Parameter(torch.randn(d, self.n_cmd) * 0.12)
        self.sw_emb = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_lm = nn.Parameter(torch.full((v,), 64.0), requires_grad=False)
        self.sw_cmd = nn.Parameter(torch.full((self.n_cmd,), 64.0), requires_grad=False)
        self.scale_emb_act = 64.0
        self.scale_blocks = [
            lm.QuantScales(act=64.0, w=64.0, p=127.0) for _ in range(self.n_layers)
        ]
        self.scale_lm = lm.QuantScales(act=64.0, w=64.0, p=127.0)
        self.scale_cmd = lm.QuantScales(act=64.0, w=64.0, p=127.0)

    def _linear(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        *,
        qat: bool,
        scale_act: float,
        scale_w: float | torch.Tensor,
    ) -> torch.Tensor:
        if not qat:
            return x @ w
        xq = fake_quant(x, scale_act)
        if isinstance(scale_w, torch.Tensor):
            wq = fake_quant(w, scale_w.reshape(1, -1))
        else:
            wq = fake_quant(w, scale_w)
        return xq @ wq

    def _block(
        self,
        x: torch.Tensor,
        blk: TransformerBlock,
        *,
        qat: bool,
        li: int,
        key_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sc = self.scale_blocks[li]
        # causal mask (+ optional PAD keys)
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        if key_pad is not None:
            mask = mask | key_pad.unsqueeze(1)

        xn = TransformerBlock.rmsnorm(x, blk.gamma1)
        q = self._linear(xn, blk.wq, qat=qat, scale_act=sc.act, scale_w=blk.sw_wq)
        k = self._linear(xn, blk.wk, qat=qat, scale_act=sc.act, scale_w=blk.sw_wk)
        v = self._linear(xn, blk.wv, qat=qat, scale_act=sc.act, scale_w=blk.sw_wv)
        scale = 1.0 / (self.d**0.5)
        scores = (q @ k.transpose(-1, -2)) * scale
        scores = scores.masked_fill(mask, -1e4)
        p = torch.softmax(scores, dim=-1)
        if qat:
            p = fake_quant(p, sc.p, qmin=0.0, qmax=127.0)
        attn = self._linear(p @ v, blk.wo, qat=qat, scale_act=sc.act, scale_w=blk.sw_wo)
        x = x + attn
        xn = TransformerBlock.rmsnorm(x, blk.gamma2)
        h = self._linear(xn, blk.w1, qat=qat, scale_act=sc.act, scale_w=blk.sw_w1)
        h = F.gelu(h)
        h = self._linear(h, blk.w2, qat=qat, scale_act=sc.act, scale_w=blk.sw_w2)
        return x + h

    def encode(self, token_ids: torch.Tensor, *, qat: bool = False) -> torch.Tensor:
        """token_ids [B,T] → hidden [B,T,D]."""
        key_pad = token_ids.eq(gsc.PAD_ID)
        x = self.emb(token_ids)
        if qat:
            ew = fake_quant(self.emb.weight, self.sw_emb.reshape(1, -1))
            x = F.embedding(token_ids, ew, padding_idx=gsc.PAD_ID)
        x = x + self.pos.unsqueeze(0)
        for li, blk in enumerate(self.blocks):
            x = self._block(x, blk, qat=qat, li=li, key_pad=key_pad)
        return x

    def forward(self, token_ids: torch.Tensor, *, qat: bool = False) -> torch.Tensor:
        """token_ids [B,T] → LM logits [B,T,V]."""
        x = self.encode(token_ids, qat=qat)
        return self._linear(
            x, self.lm_head, qat=qat, scale_act=self.scale_lm.act, scale_w=self.sw_lm
        )

    def classify(self, token_ids: torch.Tensor, *, qat: bool = False) -> torch.Tensor:
        """Mean-pool non-PAD tokens → command logits [B, N_CMD]."""
        x = self.encode(token_ids, qat=qat)
        mask = token_ids.ne(gsc.PAD_ID).unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self._linear(
            pooled, self.w_cmd, qat=qat, scale_act=self.scale_cmd.act, scale_w=self.sw_cmd
        )


@torch.no_grad()
def calibrate_scales(model: TinyCommandLM, xs: torch.Tensor, n: int = 256) -> None:
    model.eval()
    xb = xs[:n]
    # Activation scales from float forward hooks-ish: use abs max of intermediates
    # Simplified: set from embedding / head weight ranges + default 64.
    for blk in model.blocks:
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            w = getattr(blk, name)
            # per-output-channel
            amax = w.detach().abs().amax(dim=0).clamp_min(1e-6)
            getattr(blk, f"sw_{name}").copy_((127.0 / amax).clamp(1.0, 1024.0))
    amax = model.lm_head.detach().abs().amax(dim=0).clamp_min(1e-6)
    model.sw_lm.copy_((127.0 / amax).clamp(1.0, 1024.0))
    amax_c = model.w_cmd.detach().abs().amax(dim=0).clamp_min(1e-6)
    model.sw_cmd.copy_((127.0 / amax_c).clamp(1.0, 1024.0))
    amax_e = model.emb.weight.detach().abs().amax(dim=0).clamp_min(1e-6)
    model.sw_emb.copy_((127.0 / amax_e).clamp(1.0, 1024.0))
    model.scale_cmd.act = 64.0
    # Probe act scales
    logits = model(xb, qat=False)
    _ = logits
    for sc in model.scale_blocks:
        sc.act = 64.0
        sc.p = 127.0
    model.scale_lm.act = 64.0
    model.scale_emb_act = 64.0


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ignore PAD targets."""
    v = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, v),
        targets.reshape(-1),
        ignore_index=gsc.PAD_ID,
    )


@torch.no_grad()
def token_accuracy(model: TinyCommandLM, xs: torch.Tensor, ys: torch.Tensor, *, qat: bool) -> float:
    model.eval()
    logits = model(xs, qat=qat)
    pred = logits.argmax(dim=-1)
    mask = ys != gsc.PAD_ID
    correct = ((pred == ys) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def export_weights(model: TinyCommandLM, path: Path) -> None:
    model.eval()
    d, t, mh, v, L = model.d, model.t, model.mlp, model.vocab, model.n_layers
    payload: dict[str, np.ndarray] = {
        "meta_t": np.int32(t),
        "meta_d": np.int32(d),
        "meta_mlp": np.int32(mh),
        "meta_layers": np.int32(L),
        "meta_vocab": np.int32(v),
        "meta_glue_mode": np.array("float"),
        "pos": nt.to_q12(model.pos.detach().cpu().numpy()),
        "w_emb": nt.quant_weight_to_i8(
            model.emb.weight.detach().cpu().numpy(),
            model.sw_emb.detach().cpu().numpy(),
        ),
        "scale_emb_act": np.float64(model.scale_emb_act),
        "scale_emb_w_ch": model.sw_emb.detach().cpu().numpy().astype(np.float64),
        "w_lm": nt.quant_weight_to_i8(
            model.lm_head.detach().cpu().numpy(),
            model.sw_lm.detach().cpu().numpy(),
        ),
        "scale_lm_act": np.float64(model.scale_lm.act),
        "scale_lm_w_ch": model.sw_lm.detach().cpu().numpy().astype(np.float64),
        "meta_n_cmd": np.int32(model.n_cmd),
        "w_cmd": nt.quant_weight_to_i8(
            model.w_cmd.detach().cpu().numpy(),
            model.sw_cmd.detach().cpu().numpy(),
        ),
        "scale_cmd_act": np.float64(model.scale_cmd.act),
        "scale_cmd_w_ch": model.sw_cmd.detach().cpu().numpy().astype(np.float64),
    }
    for i, blk in enumerate(model.blocks):
        sc = model.scale_blocks[i]
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            w = getattr(blk, name).detach().cpu().numpy()
            sw = getattr(blk, f"sw_{name}").detach().cpu().numpy()
            payload[f"{name}{i}"] = nt.quant_weight_to_i8(w, sw)
            payload[f"scale_block{i}_{name}_w"] = sw.astype(np.float64)
        payload[f"gamma1{i}"] = nt.to_q12(blk.gamma1.detach().cpu().numpy())
        payload[f"gamma2{i}"] = nt.to_q12(blk.gamma2.detach().cpu().numpy())
        payload[f"scale_block{i}_act"] = np.float64(sc.act)
        payload[f"scale_block{i}_w"] = np.float64(64.0)
        payload[f"scale_block{i}_p"] = np.float64(sc.p)
    np.savez_compressed(path, **payload)
    print(f"wrote {path}")


def export_samples(xs: np.ndarray, ys: np.ndarray, path: Path, n: int = 32) -> None:
    n = min(n, xs.shape[0])
    np.savez_compressed(
        path,
        input_ids=xs[:n].astype(np.int32),
        target_ids=ys[:n].astype(np.int32),
        vocab=np.array(gsc.VOCAB),
    )
    print(f"wrote {path} ({n} sequences)")


def make_data(n_phrases: int, seed: int):
    rng = np.random.default_rng(seed)
    cmd_ids = rng.integers(0, len(gsc.ROBOT_COMMANDS), size=n_phrases)
    phrases = [list(gsc.ROBOT_COMMANDS[int(i)]) for i in cmd_ids]
    xs, ys = gsc.phrases_to_tensors(phrases, lm.LM_T, seed=seed)
    # Re-align labels with shuffled tensors: rebuild from phrases order after shuffle
    # phrases_to_tensors shuffles — regenerate labels the same way.
    xs_list, ys_list = xs, ys
    # Recover cmd id from decoded non-special tokens
    inv = {tuple(c): i for i, c in enumerate(gsc.ROBOT_COMMANDS)}
    labels = []
    for xrow in xs_list:
        # x is ids[:-1] padded; reconstruct words between BOS and last content
        words = []
        for tid in xrow:
            if tid in (gsc.PAD_ID, gsc.BOS_ID):
                continue
            if tid == gsc.EOS_ID:
                break
            words.append(gsc.VOCAB[int(tid)])
        labels.append(inv[tuple(words)])
    xs_a = np.asarray(xs_list, dtype=np.int64)
    ys_a = np.asarray(ys_list, dtype=np.int64)
    lab_a = np.asarray(labels, dtype=np.int64)
    n = xs_a.shape[0]
    n_val = max(256, n // 10)
    return (
        xs_a[n_val:],
        ys_a[n_val:],
        lab_a[n_val:],
        xs_a[:n_val],
        ys_a[:n_val],
        lab_a[:n_val],
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    x_tr, y_tr, c_tr, x_va, y_va, c_va = make_data(args.phrases, args.seed)
    print(
        f"geometry T={lm.LM_T} D={lm.LM_D} MLP={lm.LM_MLP} L={lm.N_LAYERS} "
        f"V={lm.VOCAB_SIZE} N_CMD={len(gsc.ROBOT_COMMANDS)}"
    )
    print(f"train={len(x_tr)} val={len(x_va)}")

    ds = TensorDataset(
        torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(c_tr)
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = TinyCommandLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def run_epochs(epochs: int, *, qat: bool, tag: str) -> None:
        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            steps = 0
            for xb, yb, cb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                cb = cb.to(device)
                opt.zero_grad(set_to_none=True)
                lm_logits = model(xb, qat=qat)
                cmd_logits = model.classify(xb, qat=qat)
                loss = ce_loss(lm_logits, yb) + F.cross_entropy(cmd_logits, cb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item())
                steps += 1
            with torch.no_grad():
                xv = torch.from_numpy(x_va).to(device)
                yv = torch.from_numpy(y_va).to(device)
                cv = torch.from_numpy(c_va).to(device)
                acc = token_accuracy(model, xv, yv, qat=qat)
                pred = model.classify(xv, qat=qat).argmax(dim=-1)
                cmd_acc = float((pred == cv).float().mean())
            print(
                f"[{tag}] epoch {ep}/{epochs}  loss={total_loss/max(steps,1):.4f}  "
                f"val_token_acc={100*acc:.1f}%  val_cmd_acc={100*cmd_acc:.1f}%"
            )

    print("--- float warm-up ---")
    run_epochs(args.epochs, qat=False, tag="float")
    print("--- calibrate ---")
    calibrate_scales(model, torch.from_numpy(x_tr).to(device))
    print("--- proxy QAT ---")
    run_epochs(max(1, args.epochs // 2), qat=True, tag="qat")

    export_weights(model, Path(args.weights))
    export_samples(x_va, y_va, Path(args.sample), n=args.sample_n)

    w = lm.CommandLmWeights.load(args.weights)
    acc = lm.next_token_accuracy(
        x_va[:128].astype(np.int32),
        y_va[:128].astype(np.int32),
        w,
    )
    cmd_ok = 0
    for i in range(min(256, len(x_va))):
        pred = lm.classify_command(x_va[i].astype(np.int32), w)
        if pred == int(c_va[i]):
            cmd_ok += 1
    n = min(256, len(x_va))
    print(f"numpy deploy next-token acc (128): {100*acc:.1f}%")
    print(f"numpy deploy command acc ({n}): {100*cmd_ok/n:.1f}%")
    demo = lm.greedy_complete(["go"], w)
    print("greedy 'go' →", " ".join(demo))
    cid = lm.classify_command(gsc.pad_to_t(gsc.encode_words(["go", "left"], add_eos=True), w.t), w)
    print("classify 'go left' →", " ".join(gsc.ROBOT_COMMANDS[cid]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrases", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--sample-n", type=int, default=32)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
