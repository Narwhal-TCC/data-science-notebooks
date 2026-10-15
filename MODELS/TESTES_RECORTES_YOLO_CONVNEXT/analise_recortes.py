"""analise_recortes.py — tabelas e gráficos usados nos notebooks 01–03 e no relatório (só leem resultados/)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"
GRUPOS = {"B1": "Backbone/resolução", "B2": "Otimizador/loss", "B3": "Recorte", "B4": "Fusão", "B5": "3 classes", "F": "Sementes extras"}


def tabela_classificadores(grupos=None):
    fila = {j["id"]: j for j in json.load(open(HERE / "fila_recortes.json"))}
    linhas = []
    for f in sorted((RES / "runs").glob("*.json")):
        r = json.load(open(f))
        j = fila.get(r["id"], {})
        if grupos and j.get("group") not in grupos:
            continue
        t, v = r["test"], r["val"]
        linhas.append({"job": r["id"], "grupo": j.get("group"), "descrição": r.get("desc", ""),
                       "val AUC": v["auc_MILCO_vs_NOMBO"], "teste acc": t.get("acc_objetos_2cls", t["acc"]),
                       "teste F1 macro": t["f1_macro"], "teste AUC": t["auc_MILCO_vs_NOMBO"],
                       "teste acc (TTA)": r["test_tta"].get("acc_objetos_2cls", r["test_tta"]["acc"]),
                       "épocas": r["epocas"], "treino (min, CPU)": round(r["treino_min"], 1),
                       "ms/recorte (CPU)": round(r["infer_ms_recorte_cpu"], 1)})
    return pd.DataFrame(linhas)


def tabela_sementes():
    fc = json.load(open(RES / "pipeline" / "fase_c.json"))
    d = pd.DataFrame(fc["finalistas"])
    d["val AUC"] = d.apply(lambda r: f"{r.val_media:.3f} ± {r.val_dp:.3f}", axis=1)
    d["teste acc"] = d.apply(lambda r: f"{r.test_acc_media:.3f} ± {r.test_acc_dp:.3f}", axis=1)
    d["teste AUC"] = d.apply(lambda r: f"{r.test_auc_media:.3f} ± {r.test_auc_dp:.3f}", axis=1)
    return d[["config", "val AUC", "teste acc", "teste AUC"]]


def tabela_deteccao():
    v = pd.DataFrame(json.load(open(RES / "deteccao" / "varredura.json")))
    return v


def resumo_deteccao(conf=0.25):
    v = tabela_deteccao()
    v = v[(v["conf"] == conf)]
    p = v.pivot_table(index=["tag", "iou_nms"], columns="split", values=["AP50", "recall", "FP_por_img"])
    p.columns = [f"{a} {b}" for a, b in p.columns]
    return p.reset_index().sort_values("AP50 val", ascending=False)


def tabela_pipelines():
    fc = json.load(open(RES / "pipeline" / "fase_c.json"))
    linhas = []
    for k, p in fc["pipelines"].items():
        t, v = p["test"], p["val"]
        linhas.append({"fluxo": k, "descrição": p["desc"], "conf": p["conf"], "aceite MILCO": p["aceita_min"],
                       "val F1 MILCO": v["MILCO_f1"], "teste precisão": t["MILCO_precisao"], "teste recall": t["MILCO_recall"],
                       "teste F1": t["MILCO_f1"], "teste AP50": t["MILCO_AP50"], "TP": t["MILCO_tp"], "FP": t["MILCO_fp"],
                       "FN": t["MILCO_fn"], "imagem AUC": t["img_auc"]})
    return pd.DataFrame(linhas), fc.get("escolhido")


def grafico_pipelines(ax=None):
    import matplotlib.pyplot as plt
    d, _ = tabela_pipelines()
    d = d[d["fluxo"].isin(["P0_antigo", "P1_antigo_ajustado", "P2_detnovo_clsantigo", "P3_detantigo_clsnovo", "P4_novo"])]
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    x = np.arange(len(d)); w = 0.27
    for i, (c, cor) in enumerate([("teste precisão", "#98B4F0"), ("teste recall", "#2F6AAD"), ("teste F1", "#161D40")]):
        ax.bar(x + (i - 1) * w, d[c], w, label=c.replace("teste ", ""), color=cor)
    ax.set_xticks(x); ax.set_xticklabels(d["fluxo"].str.replace("_", "\n", n=1), fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("MILCO por objeto (teste)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
