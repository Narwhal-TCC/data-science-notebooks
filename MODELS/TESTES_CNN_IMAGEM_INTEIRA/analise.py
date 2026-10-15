"""
analise.py — funções de análise e gráficos usadas pelos notebooks 01–04.
Lê os resultados gravados pelo sonar_harness em resultados/runs/*.json e *_probs.npz.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "resultados" / "runs"
FIGS = HERE / "resultados" / "figuras"
FIGS.mkdir(parents=True, exist_ok=True)

# paleta categórica validada (ordem fixa) + neutro para o baseline
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#eda100"]
BASE = "#8a8986"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASELINE_ID = "nb01_baseline_mnv2_96_frozen_adam"
HIST_BASELINE = {"auc": 0.898, "f1_05": 0.787, "precision": 0.881, "recall": 0.712}  # notebook antigo

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
})


def load_runs(prefix: str | None = None) -> dict:
    out = {}
    for f in sorted(RUNS.glob("*.json")):
        r = json.load(open(f))
        if prefix and not r["id"].startswith(prefix):
            continue
        out[r["id"]] = r
    return out


def load_probs(run_id: str) -> dict:
    return dict(np.load(RUNS / f"{run_id}_probs.npz"))


def tabela(runs: dict, tta: bool = False) -> pd.DataFrame:
    rows = []
    for rid, r in runs.items():
        m = r["metrics_tta"] if tta and "metrics_tta" in r else r["metrics"]
        c = r["cfg"]
        rows.append({
            "experimento": rid, "backbone": c["backbone"], "res": c["img_size"], "treino": c.get("freeze"),
            "otimizador": c["optimizer"] + ("+" + c["sam_base"] if c["optimizer"] == "sam" else ""),
            "loss": c.get("loss", "bce"), "aug": c.get("aug") + ("+" + c["mix"] if c.get("mix", "none") != "none" else ""),
            "AUC": m["auc"], "PR-AUC": m["pr_auc"], "F1 (thr val)": m["tval_f1"],
            "Precisão (thr val)": m["tval_precision"], "Recall (thr val)": m["tval_recall"],
            "F1@0.5": m["t05_f1"], "F1 oráculo": m["oracle_f1"],
            "épocas": r["epochs_run"], "treino (min)": r["train_time_s"] / 60,
            "inferência (ms/img CPU)": r["infer_ms_per_img_cpu"],
        })
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def barras(df: pd.DataFrame, metricas=("AUC", "PR-AUC", "F1 (thr val)"), titulo="", arquivo=None):
    """Barras horizontais por experimento, uma faceta por métrica; baseline em cinza."""
    n = len(metricas)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 0.42 * len(df) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    d = df.iloc[::-1]
    for ax, met in zip(axes, metricas):
        cores = [BASE if e == BASELINE_ID else C[0] for e in d["experimento"]]
        ax.barh(d["experimento"], d[met], color=cores, height=0.62)
        lo = max(0.0, d[met].min() - 0.08)
        ax.set_xlim(lo, min(1.0, d[met].max() + 0.04))
        for y, v in enumerate(d[met]):
            ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        if BASELINE_ID in set(d["experimento"]):
            b = float(d.loc[d["experimento"] == BASELINE_ID, met].iloc[0])
            ax.axvline(b, color=BASE, lw=1, ls="--")
        ax.set_title(met)
        ax.grid(axis="y", visible=False)
    fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    if arquivo:
        fig.savefig(FIGS / arquivo, bbox_inches="tight")
    return fig


def curvas_treino(runs: dict, ids, arquivo=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, rid in enumerate(ids):
        h = pd.DataFrame(runs[rid]["hist"])
        cor = BASE if rid == BASELINE_ID else C[i % len(C)]
        axes[0].plot(h["epoch"], h["train_loss"], color=cor, lw=2, label=rid)
        axes[1].plot(h["epoch"], h["val_auc"], color=cor, lw=2, label=rid)
    axes[0].set_title("Loss de treino"); axes[0].set_xlabel("época")
    axes[1].set_title("AUC na validação"); axes[1].set_xlabel("época")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    if arquivo:
        fig.savefig(FIGS / arquivo, bbox_inches="tight")
    return fig


def roc_pr(ids, tta=False, arquivo=None, titulo=""):
    """Curvas ROC e Precisão-Recall no conjunto de teste do holdout (até 4 séries + baseline)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    k = 0
    for rid in ids:
        p = load_probs(rid)
        runs = load_runs()
        y = _labels()[p["test_idx"]]
        s = p["p_test_tta"] if tta and "p_test_tta" in p else p["p_test"]
        cor = BASE if rid == BASELINE_ID else C[k % len(C)]
        k += rid != BASELINE_ID
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, color=cor, lw=2, label=f"{rid} (AUC {roc_auc_score(y, s):.3f})")
        axes[1].plot(rc, pr, color=cor, lw=2, label=f"{rid} (AP {average_precision_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=1)
    axes[0].set_xlabel("Taxa de falsos positivos"); axes[0].set_ylabel("Recall (sensibilidade)"); axes[0].set_title("ROC")
    axes[1].axvline(0.85, color=INK2, lw=1, ls=":")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisão"); axes[1].set_title("Precisão × Recall (linha: recall 0,85)")
    for ax in axes:
        ax.legend(fontsize=7, loc="lower left" if ax is axes[1] else "lower right")
    fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    if arquivo:
        fig.savefig(FIGS / arquivo, bbox_inches="tight")
    return fig


_Y = None


def _labels():
    global _Y
    if _Y is None:
        import sonar_harness as h
        _Y = pd.read_csv(h.CACHE_DIR / "index.csv")["label"].to_numpy()
    return _Y


def metricas_de_probs(y_te, p_te, y_va, p_va):
    import sonar_harness as h
    return h.compute_metrics(y_te, p_te, y_va, p_va)


def ensemble_holdout(ids, tta=True):
    """Média simples das probabilidades de vários modelos no mesmo holdout."""
    y = _labels()
    P = [load_probs(i) for i in ids]
    key_t, key_v = ("p_test_tta", "p_val_tta") if tta else ("p_test", "p_val")
    te, va = P[0]["test_idx"], P[0]["val_idx"]
    assert all((p["test_idx"] == te).all() for p in P)
    pt = np.mean([p[key_t] for p in P], 0)
    pv = np.mean([p[key_v] for p in P], 0)
    return metricas_de_probs(y[te], pt, y[va], pv)


# ------------------------------------------------------------------ validação cruzada
def cv_resumo(prefixo="cv_", tta=False):
    """Retorna (por_fold, resumo) — métricas por fold e média ± desvio por modelo,
    além das métricas sobre as predições out-of-fold (OOF) concatenadas."""
    runs = load_runs(prefixo)
    y = _labels()
    rows, oof = [], {}
    for rid, r in runs.items():
        modelo = rid[len(prefixo):].rsplit("_f", 1)[0]
        fold = r["cfg"]["fold"]
        m = r["metrics_tta"] if tta and "metrics_tta" in r else r["metrics"]
        rows.append({"modelo": modelo, "fold": fold, "AUC": m["auc"], "PR-AUC": m["pr_auc"],
                     "F1 (thr val)": m["tval_f1"], "Precisão (thr val)": m["tval_precision"],
                     "Recall (thr val)": m["tval_recall"], "Prec. c/ recall≥0,85 (thr val)": m["trec85_precision"],
                     "Recall (thr rec85)": m["trec85_recall"], "épocas": r["epochs_run"], "treino (min)": r["train_time_s"] / 60})
        p = load_probs(rid)
        s = p["p_test_tta"] if tta and "p_test_tta" in p else p["p_test"]
        sv = p["p_val_tta"] if tta and "p_val_tta" in p else p["p_val"]
        oof.setdefault(modelo, {})[fold] = (p["test_idx"], s, p["val_idx"], sv)
    por_fold = pd.DataFrame(rows).sort_values(["modelo", "fold"]).reset_index(drop=True)
    agg = por_fold.drop(columns="fold").groupby("modelo").agg(["mean", "std"])
    resumo = pd.DataFrame({k: agg[k]["mean"].map("{:.3f}".format) + " ± " + agg[k]["std"].map("{:.3f}".format)
                           for k in ["AUC", "PR-AUC", "F1 (thr val)", "Precisão (thr val)", "Recall (thr val)",
                                     "Prec. c/ recall≥0,85 (thr val)"]})
    resumo["n_folds"] = por_fold.groupby("modelo").size()
    resumo["AUC_mean"] = agg["AUC"]["mean"]
    resumo = resumo.sort_values("AUC_mean", ascending=False)
    return por_fold, resumo, oof


def oof_metricas(oof_modelo):
    """AUC/PR-AUC sobre todas as predições out-of-fold (1170 imagens, cada uma prevista 1x)."""
    y = _labels()
    idx = np.concatenate([v[0] for v in oof_modelo.values()])
    s = np.concatenate([v[1] for v in oof_modelo.values()])
    return {"AUC OOF": roc_auc_score(y[idx], s), "PR-AUC OOF": average_precision_score(y[idx], s), "n": len(idx)}


def oof_ensemble(oof, modelos):
    """Ensemble (média) de vários modelos fold a fold; threshold escolhido na validação de cada fold."""
    y = _labels()
    rows = []
    for k in sorted(oof[modelos[0]].keys()):
        te = oof[modelos[0]][k][0]; va = oof[modelos[0]][k][2]
        pt = np.mean([oof[m][k][1] for m in modelos], 0)
        pv = np.mean([oof[m][k][3] for m in modelos], 0)
        mm = metricas_de_probs(y[te], pt, y[va], pv)
        rows.append({"fold": k, "AUC": mm["auc"], "PR-AUC": mm["pr_auc"], "F1 (thr val)": mm["tval_f1"],
                     "Precisão (thr val)": mm["tval_precision"], "Recall (thr val)": mm["tval_recall"],
                     "Prec. c/ recall≥0,85 (thr val)": mm["trec85_precision"]})
    return pd.DataFrame(rows)


def cv_boxplot(por_fold, metricas=("AUC", "PR-AUC", "F1 (thr val)"), arquivo=None, extra=None):
    """Pontos por fold + média, por modelo (uma faceta por métrica)."""
    d = por_fold.copy()
    if extra is not None:
        d = pd.concat([d, extra], ignore_index=True)
    ordem = d.groupby("modelo")["AUC"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, len(metricas), figsize=(4.3 * len(metricas), 0.55 * len(ordem) + 1.5), sharey=True)
    for ax, met in zip(np.atleast_1d(axes), metricas):
        for i, mod in enumerate(ordem):
            v = d.loc[d["modelo"] == mod, met].to_numpy()
            cor = BASE if mod.startswith("baseline") else C[0]
            ax.scatter(v, np.full_like(v, i, dtype=float), s=36, color=cor, alpha=0.55, edgecolor="white", lw=1, zorder=3)
            ax.scatter([v.mean()], [i], s=120, marker="|", color=INK, lw=2.5, zorder=4)
            ax.text(v.mean(), i + 0.28, f"{v.mean():.3f}", ha="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(ordem))); ax.set_yticklabels(ordem)
        ax.set_title(met); ax.grid(axis="y", visible=False)
    fig.suptitle("Validação cruzada 5-fold — pontos = folds, traço = média", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    if arquivo:
        fig.savefig(FIGS / arquivo, bbox_inches="tight")
    return fig
