"""
inferencia.py — aplica o modelo campeão (ConvNeXt-Pico 320px, média dos 5 folds + TTA) em imagens de sonar.

Uso:
    python inferencia.py caminho/img1.jpg caminho/pasta_de_imagens ... [--modo recall85|f1]

Saída: CSV no stdout com arquivo, probabilidade de conter MILCO e decisão (0/1).
Requer: torch, timm, pillow, numpy (os pesos ImageNet NÃO são necessários — o checkpoint já contém tudo).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import timm
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "modelos" / "campeao_config.json"))


def carregar_modelos():
    modelos = []
    for f in CFG["arquivos"]:
        ck = torch.load(HERE / "modelos" / f, map_location="cpu")
        cfg = ck["cfg"]
        base = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0)
        m = nn.Sequential()
        m.add_module("backbone", base)
        m.add_module("head", nn.Sequential(nn.Dropout(cfg.get("drop", 0.3)), nn.Linear(base.num_features, 1)))
        m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        m.eval()
        dc = timm.data.resolve_data_config({}, model=base)
        m.mean = torch.tensor(dc["mean"]).view(1, 3, 1, 1)
        m.std = torch.tensor(dc["std"]).view(1, 3, 1, 1)
        modelos.append(m)
    return modelos


@torch.inference_mode()
def prever(caminhos, modelos, bs=8):
    S = CFG["img_size"]
    probs = []
    for i in range(0, len(caminhos), bs):
        x = np.stack([np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BICUBIC)) for p in caminhos[i:i + bs]])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
        p = 0
        for m in modelos:
            xn = (x - m.mean) / m.std
            p = p + 0.5 * (torch.sigmoid(m(xn).squeeze(1)) + torch.sigmoid(m(xn.flip(-1)).squeeze(1)))
        probs.append((p / len(modelos)).numpy())
    return np.concatenate(probs)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    modo = "recall85" if "--modo" in sys.argv and "recall85" in sys.argv else "f1"
    thr = CFG["threshold_recall_85"] if modo == "recall85" else CFG["threshold_max_f1"]
    arquivos = []
    for a in args:
        p = Path(a)
        arquivos += sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")) if p.is_dir() else [p]
    modelos = carregar_modelos()
    pr = prever([str(a) for a in arquivos], modelos)
    print("arquivo,prob_milco,decisao")
    for a, v in zip(arquivos, pr):
        print(f"{a},{v:.4f},{int(v >= thr)}")
