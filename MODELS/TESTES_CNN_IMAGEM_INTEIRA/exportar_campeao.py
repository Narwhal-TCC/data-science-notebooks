"""Exporta o campeão (5 modelos dos folds, fp16) e calibra os thresholds nas predições out-of-fold (com TTA)."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

import sonar_harness as h
import analise as A

MODELO = "convnextpico_320_adamw"
OUT = h.MODELS_OUT_DIR
OUT.mkdir(parents=True, exist_ok=True)

_, _, oof = A.cv_resumo("cv_", tta=True)
y = A._labels()
idx = np.concatenate([v[0] for v in oof[MODELO].values()])
s = np.concatenate([v[1] for v in oof[MODELO].values()])
pr, rc, th = precision_recall_curve(y[idx], s)
f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
i_f1 = int(np.argmax(f1))
ok = np.where(rc[:-1] >= 0.85)[0]
i_r = int(ok[np.argmax(pr[:-1][ok])])

cfg = json.load(open(h.RESULTS_DIR / "runs" / f"cv_{MODELO}_f0.json"))["cfg"]
arquivos = []
for k in range(5):
    sd = torch.load(h.WORK_DIR / f"cv_{MODELO}_f{k}" / "best.pt", map_location="cpu")
    sd = {kk: v.half() if v.is_floating_point() else v for kk, v in sd.items()}
    f = OUT / f"campeao_{MODELO}_fold{k}.pt"
    torch.save({"state_dict": sd, "cfg": cfg}, f)
    arquivos.append(f.name)

info = {
    "modelo": MODELO,
    "descricao": "ConvNeXt-Pico (ImageNet) 320px, fine-tuning completo, AdamW+cosine; inferência = média dos 5 modelos dos folds + TTA (flip horizontal)",
    "rotulo": "1 = imagem contém ao menos um MILCO (mine-like contact)",
    "arquivos": arquivos,
    "img_size": cfg["img_size"],
    "threshold_max_f1": float(th[i_f1]),
    "oof_no_threshold_max_f1": {"precisao": float(pr[i_f1]), "recall": float(rc[i_f1]), "f1": float(f1[i_f1])},
    "threshold_recall_85": float(th[i_r]),
    "oof_no_threshold_recall_85": {"precisao": float(pr[i_r]), "recall": float(rc[i_r])},
    "oof_auc": float(roc_auc_score(y[idx], s)),
    "oof_pr_auc": float(average_precision_score(y[idx], s)),
    "obs": "Thresholds calibrados nas 1170 predições out-of-fold (cada imagem prevista por um modelo que não a viu). "
           "A média dos 5 modelos tende a ser ligeiramente mais confiante que um modelo isolado; revalide o threshold com dados novos.",
    "cfg": cfg,
}
json.dump(info, open(OUT / "campeao_config.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in info.items() if k != "cfg"}, indent=1, ensure_ascii=False))
