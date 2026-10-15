"""
sonar_harness.py — harness único para os experimentos de classificação mina / não-mina
(dataset público da Marinha Portuguesa: side-scan sonar, 1170 imagens, 2010-2021, MILCO/NOMBO).

Rótulo binário: 1 = imagem contém ao menos um MILCO (mine-like contact); 0 = caso contrário
(inclui imagens só com NOMBO = non-mine bottom object, e fundo sem objeto).

Principais características
- PyTorch + timm, pesos ImageNet carregados de arquivos locais (WEIGHTS_DIR).
- Cache de imagens redimensionadas em .npy (uint8) por resolução.
- Augmentations batched em tensor (rápidas em CPU): padrão, "baseline" (equivalente ao Keras do
  notebook campeão) e "sonar" (speckle multiplicativo + ganho por alcance + random erasing).
- Otimizadores: adam, adamw, sgd (Nesterov), sam (sobre adamw ou sgd). Schedulers: cosine c/ warmup, plateau.
- Loss: BCE com class_weight suavizado, focal loss, label smoothing, mixup/cutmix.
- Treino FATIADO e RETOMÁVEL: `run_slice(budget_s)` treina por no máximo budget_s segundos,
  salva checkpoint (modelo + otimizador + RNG + posição no epoch) e retoma de onde parou na próxima chamada.
  Também funciona de ponta a ponta (budget infinito) em um Jupyter numa máquina com GPU.
- Métricas no teste: ROC-AUC, PR-AUC, F1 com threshold escolhido na VALIDAÇÃO, F1@0.5,
  precisão com recall>=0.85 (threshold da validação), e F1 "oráculo" (threshold escolhido no teste,
  só para comparar com a metodologia do notebook antigo).
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ caminhos (sobrescrevíveis por env)
HERE = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("SONAR_RAW_DIR", Path.home() / "data" / "raw"))
CACHE_DIR = Path(os.environ.get("SONAR_CACHE_DIR", Path.home() / "data" / "cache"))
WORK_DIR = Path(os.environ.get("SONAR_WORK_DIR", Path.home() / "work"))          # checkpoints de trabalho
WEIGHTS_DIR = Path(os.environ.get("SONAR_WEIGHTS_DIR", Path.home() / "weights"))
RESULTS_DIR = Path(os.environ.get("SONAR_RESULTS_DIR", HERE / "resultados"))
MODELS_OUT_DIR = Path(os.environ.get("SONAR_MODELS_DIR", HERE / "modelos"))
QUEUE_FILE = Path(os.environ.get("SONAR_QUEUE", HERE / "fila_experimentos.json"))

WEIGHT_FILES = {
    "mobilenetv2_100": "mobilenetv2_100_ra-b33bc2c4.pth",
    "tf_efficientnetv2_b0": "tf_efficientnetv2_b0-c7cc451f.pth",
    "convnext_pico": "convnext_pico_d1-10ad7f0d.pth",
    "deit_tiny_patch16_224": "deit_tiny_patch16_a1_0-90f89490.pth",
    "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
    "regnety_016": "regnety_016-54367f74.pth",
}
SEED = 42

# =================================================================== DADOS
def build_index(raw_dir: Path = RAW_DIR):
    """Lê as anotações YOLO originais e monta o índice das 1170 imagens."""
    import pandas as pd
    rows = []
    for jpg in sorted(glob.glob(str(raw_dir / "*" / "*.jpg"))):
        txt = jpg[:-4] + ".txt"
        cls = []
        if os.path.exists(txt):
            for line in open(txt).read().strip().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cls.append(int(float(p[0])))
        rows.append({
            "path": jpg, "stem": Path(jpg).stem, "year": Path(jpg).parent.name,
            "n_milco": cls.count(0), "n_nombo": cls.count(1),
            "label": int(cls.count(0) > 0),
        })
    return pd.DataFrame(rows)


def load_cache(size: int):
    """Retorna (X uint8 [N,size,size,3], y int [N], years [N], stems [N]) com cache em disco."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"imgs_{size}.npy"
    meta = CACHE_DIR / "index.csv"
    import pandas as pd
    if meta.exists():
        df = pd.read_csv(meta)
    else:
        df = build_index()
        df.to_csv(meta, index=False)
    if not f.exists():
        from PIL import Image
        X = np.zeros((len(df), size, size, 3), dtype=np.uint8)
        for i, p in enumerate(df["path"]):
            X[i] = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC))
        np.save(f, X)
    X = np.load(f, mmap_mode="r")
    return X, df["label"].to_numpy().astype(np.int64), df["year"].astype(str).to_numpy(), df["stem"].to_numpy()


YEARS_LOYO = ["2010", "2015", "2017", "2018", "2021"]


def get_split(split: str, y: np.ndarray, fold=None, years=None):
    """split='holdout' -> 70/15/15 estratificado fixo. split='cv5' -> fold k de StratifiedKFold(5),
    com 15% do treino separado como validação (early stopping + escolha de threshold)."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    idx = np.arange(len(y))
    if split == "holdout":
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
        return tr, va, te
    if split == "cv5":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr_full, te = list(skf.split(idx, y))[fold]
        tr, va = train_test_split(tr_full, test_size=0.15, stratify=y[tr_full], random_state=SEED + fold)
        return tr, va, te
    if split == "loyo":   # leave-one-year-out: testa num ano (campanha) inteiro nunca visto
        ano = YEARS_LOYO[fold] if isinstance(fold, int) else str(fold)
        te = idx[years == ano]
        tr_full = idx[years != ano]
        tr, va = train_test_split(tr_full, test_size=0.15, stratify=y[tr_full], random_state=SEED)
        return tr, va, te
    raise ValueError(split)


# =================================================================== DATASETS EXTERNOS (SSS, só negativos)
EXT_DIR = Path(os.environ.get("SONAR_EXT_DIR", Path.home() / "data" / "external"))
EXT_SOURCES = {   # nome -> padrão glob (todas as imagens são side-scan sonar reais, sem minas)
    "sctd2_small": "sctd2/small_targets_voc/JPEGImages/*.jpg",   # Klein, alvos pequenos (armadilhas)
    "sctd": "sctd/SCTD/JPEGImages/*.jpg",                         # navios, aeronaves, pessoas
    "klsg": "klsg/**/*.*",                                        # navios e aviões (Klein, EdgeTech...)
}


def load_external(size: int):
    """Cache das imagens externas. Retorna (Xe uint8 [M,size,size,3], fonte [M], particao [M] 'train'/'test')."""
    import pandas as pd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = CACHE_DIR / "ext_index.csv"
    if meta.exists():
        df = pd.read_csv(meta)
    else:
        rows = []
        rng = np.random.default_rng(SEED)
        for src, pat in EXT_SOURCES.items():
            fs = sorted(f for f in glob.glob(str(EXT_DIR / pat), recursive=True)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
            part = np.where(rng.permutation(len(fs)) < len(fs) // 2, "train", "test")
            rows += [{"path": f, "fonte": src, "particao": pt} for f, pt in zip(fs, part)]
        df = pd.DataFrame(rows)
        df.to_csv(meta, index=False)
    f = CACHE_DIR / f"ext_{size}.npy"
    if not f.exists():
        from PIL import Image
        Xe = np.zeros((len(df), size, size, 3), dtype=np.uint8)
        for i, p in enumerate(df["path"]):
            Xe[i] = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC))
        np.save(f, Xe)
    return np.load(f, mmap_mode="r"), df["fonte"].to_numpy(), df["particao"].to_numpy()


class Combo:
    """Concatena (sem copiar) o SIMD e o conjunto externo; índices >= len(A) apontam para B."""
    def __init__(self, A, B):
        self.A, self.B, self.n = A, B, len(A)
        self.shape = (len(A) + len(B),) + A.shape[1:]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx),) + self.A.shape[1:], np.uint8)
        m = idx < self.n
        if m.any():
            out[m] = self.A[idx[m]]
        if (~m).any():
            out[~m] = self.B[idx[~m] - self.n]
        return out


# =================================================================== AUGMENTATION (batched, tensor)
def _affine(x, deg, scale, translate, gen):
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    ang = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(deg)
    sc = 1 + (torch.rand(B, generator=gen) * 2 - 1) * scale
    tx = (torch.rand(B, generator=gen) * 2 - 1) * translate * 2
    ty = (torch.rand(B, generator=gen) * 2 - 1) * translate * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def augment(x, kind: str, gen):
    """x: float tensor [B,3,H,W] em [0,1]. Retorna tensor aumentado em [0,1]."""
    import torch
    if kind in (None, "none"):
        return x
    B = x.shape[0]
    flip = torch.rand(B, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)        # só flip horizontal (sombra acústica)
    if kind == "baseline":   # ~ RandomRotation(0.05) RandomZoom(0.1) RandomContrast(0.2) do Keras
        x = _affine(x, deg=18, scale=0.10, translate=0.0, gen=gen)
        c = 1 + (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1) * 0.2
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - m) * c + m).clamp(0, 1)
    # padrão
    x = _affine(x, deg=10, scale=0.10, translate=0.05, gen=gen)
    b = 1 + (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1) * 0.2
    c = 1 + (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1) * 0.2
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    if kind == "sonar":
        H, W = x.shape[-2:]
        # 1) speckle multiplicativo (Gamma com L "looks" aleatório) em 50% das imagens
        L = torch.randint(4, 16, (B, 1, 1, 1), generator=gen).float()
        g = torch._standard_gamma(L.expand(B, 1, H, W)) / L
        apply = (torch.rand(B, 1, 1, 1, generator=gen) < 0.5).float()
        x = x * (apply * g + (1 - apply))
        # 2) ganho dependente do alcance (gradiente ao longo do eixo across-track = largura)
        ramp = torch.linspace(-1, 1, W).view(1, 1, 1, W)
        k = (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1) * 0.35
        x = x * (1 + k * ramp)
        # 3) random erasing pequeno (oclusão) em 30% das imagens
        for i in range(B):
            if torch.rand(1, generator=gen).item() < 0.3:
                eh, ew = int(H * 0.15), int(W * 0.15)
                y0 = torch.randint(0, H - eh, (1,), generator=gen).item()
                x0 = torch.randint(0, W - ew, (1,), generator=gen).item()
                x[i, :, y0:y0 + eh, x0:x0 + ew] = x[i].mean()
    return x.clamp(0, 1)


def mix_batch(x, y, mode, gen, alpha=0.4):
    import torch
    if mode not in ("mixup", "cutmix") or torch.rand(1, generator=gen).item() > 0.5:
        return x, y
    lam = float(np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen))).beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], generator=gen)
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    H, W = x.shape[-2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, lam * y + (1 - lam) * y[perm]


# =================================================================== MODELO
def build_model(cfg):
    import timm
    import torch
    import torch.nn as nn
    name = cfg["backbone"]
    if cfg.get("head") == "keras_baseline":
        base = timm.create_model(name, pretrained=False, num_classes=0)
        _load_weights(base, name)
        nf = base.num_features
        head = nn.Sequential(nn.Dropout(0.5), nn.Linear(nf, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        model = nn.Sequential()
        model.add_module("backbone", base)
        model.add_module("head", head)
    else:
        kw = {}
        if "drop_path" in cfg:
            kw["drop_path_rate"] = cfg["drop_path"]
        base = timm.create_model(name, pretrained=False, num_classes=0, **kw)
        _load_weights(base, name)
        model = nn.Sequential()
        model.add_module("backbone", base)
        model.add_module("head", nn.Sequential(nn.Dropout(cfg.get("drop", 0.3)), nn.Linear(base.num_features, 1)))
    dc = timm.data.resolve_data_config({}, model=model.backbone)
    model.gray = bool(cfg.get("gray", False))
    model.mean = torch.tensor(dc["mean"]).view(1, 3, 1, 1)
    model.std = torch.tensor(dc["std"]).view(1, 3, 1, 1)
    if cfg.get("freeze") == "frozen":
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model


def _load_weights(net, name):
    import torch
    sd = torch.load(WEIGHTS_DIR / WEIGHT_FILES[name], map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith(("classifier.", "head.fc.", "head.weight", "head.bias", "fc."))}
    if "pos_embed" in sd and hasattr(net, "pos_embed") and sd["pos_embed"].shape != net.pos_embed.shape:
        from timm.layers import resample_abs_pos_embed
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], net.patch_embed.grid_size, num_prefix_tokens=1)
    r = net.load_state_dict(sd, strict=False)
    bad = [k for k in r.missing_keys if not k.startswith(("classifier", "head", "fc"))]
    assert not bad, f"pesos incompletos para {name}: {bad[:5]}"


def set_train_mode(model, cfg):
    model.train()
    if cfg.get("freeze") == "frozen":
        model.backbone.eval()   # BN congelado, como base_model(training=False) no Keras


# =================================================================== OTIMIZAÇÃO
class SAM:
    """Sharpness-Aware Minimization (Foret et al., 2021) sobre um otimizador base."""
    def __init__(self, params, base_opt, rho=0.05):
        self.params = [p for g in params for p in g["params"]]
        self.base = base_opt
        self.rho = rho
        self.param_groups = base_opt.param_groups

    def first_step(self):
        import torch
        grads = [p.grad for p in self.params if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2) + 1e-12
        self._e = []
        for p in self.params:
            if p.grad is None:
                self._e.append(None)
                continue
            e = p.grad * (self.rho / norm)
            p.data.add_(e)
            self._e.append(e)

    def second_step(self):
        for p, e in zip(self.params, self._e):
            if e is not None:
                p.data.sub_(e)
        self.base.step()

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, sd):
        self.base.load_state_dict(sd)


def build_optimizer(model, cfg):
    import torch
    lr = cfg["lr"]
    wd = cfg.get("wd", 0.0)
    head_mult = cfg.get("head_lr_mult", 1.0)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    hd = [p for p in model.head.parameters() if p.requires_grad]
    groups = [g for g in [{"params": bb, "lr": lr}, {"params": hd, "lr": lr * head_mult}] if g["params"]]
    for g in groups:
        g["initial_lr"] = g["lr"]
    name = cfg["optimizer"]
    base_name = cfg.get("sam_base", "adamw") if name == "sam" else name
    if base_name == "adam":
        opt = torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    elif base_name == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd or 0.05)
    elif base_name == "sgd":
        opt = torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd or 1e-4)
    else:
        raise ValueError(name)
    if name == "sam":
        return SAM(groups, opt, rho=cfg.get("sam_rho", 0.05))
    return opt


def lr_at(cfg, step, total_steps):
    """Fator multiplicativo do LR para scheduler 'cosine' (com warmup linear)."""
    warm = int(total_steps * cfg.get("warmup_frac", 0.05))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(1, total_steps - warm)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def loss_fn(logits, target, weights, cfg):
    import torch
    import torch.nn.functional as F
    ls = cfg.get("label_smoothing", 0.0)
    t = target * (1 - ls) + 0.5 * ls
    if cfg.get("loss", "bce") == "focal":
        gamma, alpha = cfg.get("focal_gamma", 2.0), cfg.get("focal_alpha", 0.6)
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        pt = p * t + (1 - p) * (1 - t)
        a = alpha * t + (1 - alpha) * (1 - t)
        return (a * (1 - pt) ** gamma * ce).mean()
    w = weights[1] * target + weights[0] * (1 - target)
    return (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * w).mean()


# =================================================================== MÉTRICAS
def compute_metrics(y_true, p_test, y_val=None, p_val=None, recall_target=0.85):
    from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                                 precision_score, recall_score, roc_auc_score, accuracy_score)
    out = {"auc": float(roc_auc_score(y_true, p_test)), "pr_auc": float(average_precision_score(y_true, p_test))}

    def at(th, pref):
        yp = (p_test >= th).astype(int)
        out[f"{pref}_thr"] = float(th)
        out[f"{pref}_f1"] = float(f1_score(y_true, yp, zero_division=0))
        out[f"{pref}_precision"] = float(precision_score(y_true, yp, zero_division=0))
        out[f"{pref}_recall"] = float(recall_score(y_true, yp, zero_division=0))
        out[f"{pref}_acc"] = float(accuracy_score(y_true, yp))

    at(0.5, "t05")
    if p_val is not None:
        pr, rc, th = precision_recall_curve(y_val, p_val)
        f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
        at(float(th[np.argmax(f1)]), "tval")                 # threshold que maximiza F1 na validação
        ok = np.where(rc[:-1] >= recall_target)[0]
        at(float(th[ok[np.argmax(pr[:-1][ok])]]) if len(ok) else 0.0, "trec85")   # recall alvo na validação
    pr, rc, th = precision_recall_curve(y_true, p_test)      # "oráculo" = metodologia antiga (threshold no teste)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    out["oracle_f1"] = float(f1.max())
    ok = np.where(rc[:-1] >= recall_target)[0]
    out["oracle_prec_at_rec85"] = float(pr[:-1][ok].max()) if len(ok) else 0.0
    return out


# =================================================================== TREINO RETOMÁVEL
DEFAULTS = dict(img_size=224, batch_size=32, epochs=30, patience=8, optimizer="adamw", lr=3e-4, wd=0.05,
                head_lr_mult=10.0, scheduler="cosine", warmup_frac=0.05, freeze="full", aug="standard",
                mix="none", loss="bce", soften=0.6, split="holdout", fold=None, tta=True, save_model=False)


def _to_input(xb_uint8, model):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(xb_uint8)).permute(0, 3, 1, 2).float().div_(255.0)
    if getattr(model, "gray", False):   # entrada em tons de cinza (invariante à paleta de cor do sonar)
        L = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = L.expand(-1, 3, -1, -1).contiguous()
    return x


def _norm(x, model):
    return (x - model.mean) / model.std


def predict(model, X, idx, cfg, bs=64, tta=False):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(idx), bs):
            x = _to_input(X[idx[i:i + bs]], model)
            p = torch.sigmoid(model(_norm(x, model)).squeeze(1))
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(_norm(x.flip(-1), model)).squeeze(1)))
            out.append(p.numpy())
    return np.concatenate(out)


def _paths(job_id):
    d = WORK_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "state.pt", d / "best.pt"


def train_job(job: dict, deadline: float, log=print) -> bool:
    """Treina/retoma um job até `deadline` (time.time()). Retorna True se o job terminou."""
    import torch
    torch.set_num_threads(int(os.environ.get("SONAR_THREADS", os.cpu_count() or 2)))
    cfg = {**DEFAULTS, **job["cfg"]}
    jid = job["id"]
    X, y, years, stems = load_cache(cfg["img_size"])
    tr, va, te = get_split(cfg["split"], y, cfg["fold"], years)
    if cfg.get("extra_neg"):   # negativos externos (metade 'train' de cada fonte) entram só no treino
        Xe, fonte, part = load_external(cfg["img_size"])
        sel = np.where((part == "train") & np.isin(fonte, cfg["extra_neg"]))[0]
        tr = np.concatenate([tr, len(y) + sel])
        X = Combo(X, Xe)
        y = np.concatenate([y, np.zeros(len(Xe), dtype=y.dtype)])
    state_f, best_f = _paths(jid)

    torch.manual_seed(SEED)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    is_sam = isinstance(opt, SAM)
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(tr) / bs)
    total_steps = steps_per_epoch * cfg["epochs"]
    plateau = None
    if cfg["scheduler"] == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt.base if is_sam else opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    # class weights suavizados (igual ao notebook campeão)
    n0, n1 = (y[tr] == 0).sum(), (y[tr] == 1).sum()
    raw = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
    weights = {c: 1 + cfg["soften"] * (w - 1) for c, w in raw.items()}

    st = {"epoch": 0, "batch": 0, "step": 0, "best_auc": -1, "best_loss": 1e9, "bad": 0, "hist": [], "train_time": 0.0,
          "ep_loss": 0.0, "ep_n": 0}
    if state_f.exists():
        ck = torch.load(state_f, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if plateau is not None and ck.get("plateau"):
            plateau.load_state_dict(ck["plateau"])
        st = ck["st"]
        torch.set_rng_state(ck["rng"])

    gen = torch.Generator().manual_seed(SEED + 1000 * st["epoch"] + st["batch"])
    t_start = time.time()
    finished = False
    while True:
        if st["epoch"] >= cfg["epochs"] or st["bad"] >= cfg["patience"]:
            finished = True
            break
        perm = np.random.default_rng(SEED + st["epoch"]).permutation(tr)
        set_train_mode(model, cfg)
        while st["batch"] < steps_per_epoch:
            if time.time() > deadline:
                break
            b = perm[st["batch"] * bs:(st["batch"] + 1) * bs]
            xb = augment(_to_input(X[np.sort(b)], model), cfg["aug"], gen)
            yb = torch.from_numpy(y[np.sort(b)]).float()
            xb, yb = mix_batch(xb, yb, cfg["mix"], gen)
            xb = _norm(xb, model)
            if cfg["scheduler"] == "cosine":
                f = lr_at(cfg, st["step"], total_steps)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * f
            if is_sam:
                loss = loss_fn(model(xb).squeeze(1), yb, weights, cfg)
                loss.backward()
                opt.first_step()
                opt.zero_grad()
                loss_fn(model(xb).squeeze(1), yb, weights, cfg).backward()
                opt.second_step()
            else:
                loss = loss_fn(model(xb).squeeze(1), yb, weights, cfg)
                loss.backward()
                opt.step()
            opt.zero_grad()
            st["ep_loss"] += loss.item() * len(b)
            st["ep_n"] += len(b)
            st["batch"] += 1
            st["step"] += 1
        if st["batch"] < steps_per_epoch:      # fatia acabou no meio do epoch
            break
        # ---- fim do epoch: validação
        pv = predict(model, X, va, cfg)
        from sklearn.metrics import roc_auc_score, log_loss
        vauc = float(roc_auc_score(y[va], pv))
        vloss = float(log_loss(y[va], np.clip(pv, 1e-6, 1 - 1e-6)))
        st["hist"].append({"epoch": st["epoch"] + 1, "train_loss": st["ep_loss"] / max(1, st["ep_n"]),
                           "val_auc": vauc, "val_loss": vloss, "lr": opt.param_groups[0]["lr"]})
        log(f"[{jid}] ep {st['epoch'] + 1}/{cfg['epochs']} loss {st['hist'][-1]['train_loss']:.4f} val_auc {vauc:.4f} val_loss {vloss:.4f}")
        if plateau is not None:
            plateau.step(vloss)
        if vauc > st["best_auc"] + 1e-4 or (abs(vauc - st["best_auc"]) <= 1e-4 and vloss < st["best_loss"]):
            st["best_auc"], st["best_loss"], st["bad"] = vauc, vloss, 0
            torch.save(model.state_dict(), str(best_f) + ".tmp")
            os.replace(str(best_f) + ".tmp", best_f)
        else:
            st["bad"] += 1
        st["epoch"] += 1
        st["batch"] = 0
        st["ep_loss"], st["ep_n"] = 0.0, 0
        gen = torch.Generator().manual_seed(SEED + 1000 * st["epoch"])
    st["train_time"] += time.time() - t_start
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": torch.get_rng_state(),
                "plateau": plateau.state_dict() if plateau is not None else None}, str(state_f) + ".tmp")
    os.replace(str(state_f) + ".tmp", state_f)
    if finished:
        if time.time() > deadline - 60 and not job.get("_final_ok"):
            return False          # pouco tempo na fatia: finaliza (avaliação no teste) na próxima chamada
        finalize_job(job, cfg, model, X, y, years, stems, tr, va, te, st)
    return finished


def finalize_job(job, cfg, model, X, y, years, stems, tr, va, te, st):
    import torch
    _, best_f = _paths(job["id"])
    model.load_state_dict(torch.load(best_f, map_location="cpu"))
    tta = cfg.get("tta", True)
    t0 = time.time()
    pt = predict(model, X, te, cfg, tta=False)
    infer_ms = (time.time() - t0) / len(te) * 1000
    pv = predict(model, X, va, cfg, tta=False)
    res = {"id": job["id"], "group": job.get("group"), "desc": job.get("desc"), "cfg": cfg,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "pos_test": int(y[te].sum()), "epochs_run": len(st["hist"]), "train_time_s": st["train_time"],
           "infer_ms_per_img_cpu": infer_ms, "best_val_auc": st["best_auc"], "hist": st["hist"]}
    res["metrics"] = compute_metrics(y[te], pt, y[va], pv)
    probs = {"test_idx": te, "p_test": pt, "val_idx": va, "p_val": pv}
    if tta:
        ptt, pvt = predict(model, X, te, cfg, tta=True), predict(model, X, va, cfg, tta=True)
        res["metrics_tta"] = compute_metrics(y[te], ptt, y[va], pvt)
        probs.update({"p_test_tta": ptt, "p_val_tta": pvt})
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS_DIR / "runs" / f"{job['id']}.json", "w"), indent=1, default=float)
    np.savez(RESULTS_DIR / "runs" / f"{job['id']}_probs.npz", **probs)
    if cfg.get("save_model"):
        MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
        torch.save({"state_dict": sd, "cfg": cfg}, MODELS_OUT_DIR / f"{job['id']}.pt")


# =================================================================== FILA
def load_queue():
    return json.load(open(QUEUE_FILE)) if QUEUE_FILE.exists() else []


def save_queue(q):
    tmp = QUEUE_FILE.with_suffix(".tmp")
    json.dump(q, open(tmp, "w"), indent=1)
    os.replace(tmp, QUEUE_FILE)


def run_slice(budget_s: float = 150.0, log=print):
    """Executa a fila por até budget_s segundos. Imprime uma linha de status."""
    t_end = time.time() + budget_s
    q = load_queue()
    for job in q:
        if job.get("status") == "done":
            continue
        if time.time() > t_end - 25:
            break
        job["status"] = "running"
        try:
            done = train_job(job, t_end, log=log)
        except Exception as e:  # registra o erro e segue para o próximo job
            import traceback
            job["status"] = "error"
            job["error"] = traceback.format_exc()[-1500:]
            save_queue(q)
            log(f"[{job['id']}] ERRO: {e}")
            continue
        if done:
            job["status"] = "done"
            r = json.load(open(RESULTS_DIR / "runs" / f"{job['id']}.json"))
            m = r["metrics"]
            log(f"[{job['id']}] FIM  AUC {m['auc']:.4f} PR-AUC {m['pr_auc']:.4f} F1(thr val) {m['tval_f1']:.4f} "
                f"F1@0.5 {m['t05_f1']:.4f} oracleF1 {m['oracle_f1']:.4f} epochs {r['epochs_run']}")
        save_queue(q)
        if done and os.environ.get("SONAR_KEEP_STATE") != "1":
            sf, _ = _paths(job["id"])
            if sf.exists():
                os.remove(sf)     # libera disco: mantém só o best.pt do job
        if not done:
            break
    pend = [j["id"] for j in q if j.get("status") not in ("done", "error")]
    log(f"STATUS pendentes={len(pend)} prox={pend[:1]} done={sum(j.get('status') == 'done' for j in q)} "
        f"erros={sum(j.get('status') == 'error' for j in q)}")
    return pend


if __name__ == "__main__":
    import sys
    run_slice(float(sys.argv[1]) if len(sys.argv) > 1 else 150.0)
