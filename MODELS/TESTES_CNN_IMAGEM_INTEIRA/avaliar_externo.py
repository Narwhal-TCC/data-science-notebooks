"""
avaliar_externo.py — avalia modelos em imagens de side-scan sonar EXTERNAS (outros sonares/campanhas, sem minas):
  sctd2_small (Klein, alvos pequenos), sctd (navios/aeronaves/pessoas), klsg (navios/aviões).
Usa só a partição 'test' de cada fonte (a partição 'train' pode ter sido usada como negativo no treino).

Mede:
  - FPR (taxa de falso alarme) no threshold operacional do modelo e em 0,5;
  - AUC cruzada: positivos (minas do teste SIMD) × negativos externos — o modelo separa minas reais de objetos
    de outros sonares que NÃO são minas?
Retomável: grava probabilidades por (modelo, fonte) e pula o que já foi feito (cabe em fatias de ~150 s).

Uso: python avaliar_externo.py <run_id | campeao> [budget_s]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sonar_harness as h

OUT = h.RESULTS_DIR / "externo"
OUT.mkdir(parents=True, exist_ok=True)


def _modelos(alvo):
    """Lista de (nome, cfg, fabrica_do_modelo)."""
    if alvo == "campeao":
        info = json.load(open(h.MODELS_OUT_DIR / "campeao_config.json"))
        out = []
        for f in info["arquivos"]:
            def fab(f=f):
                ck = torch.load(h.MODELS_OUT_DIR / f, map_location="cpu")
                m = h.build_model(ck["cfg"])
                m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
                return m
            out.append((f.replace(".pt", ""), info["cfg"], fab))
        return out
    r = json.load(open(h.RESULTS_DIR / "runs" / f"{alvo}.json"))
    cfg = r["cfg"]

    def fab():
        m = h.build_model(cfg)
        m.load_state_dict(torch.load(h.WORK_DIR / alvo / "best.pt", map_location="cpu"))
        return m
    return [(alvo, cfg, fab)]


def rodar(alvo, budget_s=150.0):
    t_end = time.time() + budget_s
    torch.set_num_threads(2)
    arq = OUT / f"{alvo}_probs.npz"
    store = dict(np.load(arq)) if arq.exists() else {}
    for nome, cfg, fab in _modelos(alvo):
        Xe, fonte, part = h.load_external(cfg["img_size"])
        modelo = None
        for src in h.EXT_SOURCES:
            key = f"{nome}|{src}"
            if key in store:
                continue
            if time.time() > t_end - 40:
                np.savez(arq, **store)
                return False
            modelo = modelo or fab()
            idx = np.where((part == "test") & (fonte == src))[0]
            store[key] = h.predict(modelo, Xe, idx, cfg, bs=32)
            np.savez(arq, **store)
    np.savez(arq, **store)
    resumir(alvo, store)
    return True


def resumir(alvo, store):
    from sklearn.metrics import roc_auc_score
    if alvo == "campeao":
        info = json.load(open(h.MODELS_OUT_DIR / "campeao_config.json"))
        thr = info["threshold_max_f1"]
        import analise as A
        _, _, oof = A.cv_resumo("cv_")
        y = A._labels()
        idx = np.concatenate([v[0] for v in oof[info["modelo"]].values()])
        s = np.concatenate([v[1] for v in oof[info["modelo"]].values()])
        pos = s[y[idx] == 1]
        nomes = sorted({k.split("|")[0] for k in store})
    else:
        r = json.load(open(h.RESULTS_DIR / "runs" / f"{alvo}.json"))
        thr = r["metrics"]["tval_thr"]
        p = np.load(h.RESULTS_DIR / "runs" / f"{alvo}_probs.npz")
        import pandas as pd
        lab = pd.read_csv(h.CACHE_DIR / "index.csv")["label"].to_numpy()
        pos = p["p_test"][lab[p["test_idx"]] == 1]
        nomes = [alvo]
    res = {"alvo": alvo, "threshold": thr, "n_pos_simd": int(len(pos)), "fontes": {}}
    todos = []
    for src in h.EXT_SOURCES:
        s = np.mean([store[f"{n}|{src}"] for n in nomes], 0)   # campeão: média dos 5 folds
        todos.append(s)
        res["fontes"][src] = {
            "n": int(len(s)), "fpr_thr": float((s >= thr).mean()), "fpr_05": float((s >= 0.5).mean()),
            "prob_media": float(s.mean()), "prob_p95": float(np.percentile(s, 95)),
            "auc_minas_simd_vs_fonte": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(s))], np.r_[pos, s])),
        }
    s = np.concatenate(todos)
    res["total"] = {"n": int(len(s)), "fpr_thr": float((s >= thr).mean()), "fpr_05": float((s >= 0.5).mean()),
                    "auc_minas_simd_vs_externo": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(s))], np.r_[pos, s]))}
    json.dump(res, open(OUT / f"{alvo}.json", "w"), indent=1)
    print(json.dumps(res["total"]), {k: round(v["fpr_thr"], 3) for k, v in res["fontes"].items()})


if __name__ == "__main__":
    ok = rodar(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 150.0)
    print("EXTERNO", sys.argv[1], "concluido" if ok else "parcial (rode de novo)")
