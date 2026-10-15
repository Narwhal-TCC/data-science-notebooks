"""
fusao_detectores.py — Alternativa 1: combinar detectores (sem novo treino).

Gera caches de detecção "fundidos" a partir dos caches já existentes (val/teste) e avalia:
  (a) detecção (teto de recall, AP50);
  (b) fluxo completo com o ConvNeXt (3 sementes): ponto por F1 e por F2 (recall pesa 4×), escolhidos na VALIDAÇÃO,
      e modo varredura (conf 0,05).
Fusões: WBF (Weighted Boxes Fusion, Solovyev et al. 2021) ou união + NMS.
Uso: python fusao_detectores.py [budget]   → resultados/pipeline/fusao_detectores.json
"""
import json, sys, time
import numpy as np
import crop_harness as CH, fase_c as FC

CH.DET_CONFIGS.update({"E_G5_1024": dict(imgsz=1024), "E_G5_1024_tta": dict(imgsz=1024, augment=True)})
FONTES = {"G5tta": ("E_G5_1024_tta", 0.5), "mosaico": ("D5_tiles2x2_640", 0.5), "Atta": ("D4_640_tta", 0.5)}
FUSOES = {"F_G5tta+mosaico_wbf": (["G5tta", "mosaico"], "wbf"),
          "F_G5tta+mosaico_nms": (["G5tta", "mosaico"], "nms"),
          "F_G5tta+Atta_wbf": (["G5tta", "Atta"], "wbf"),
          "F_G5tta+mosaico+Atta_wbf": (["G5tta", "mosaico", "Atta"], "wbf")}
ENS = "ens:b1_convnextpico_224"
OUT = CH.RES / "pipeline" / "fusao_detectores.json"


def wbf(listas, iou_thr=0.55):
    """listas: uma matriz [N,5] (x1,y1,x2,y2,conf) por modelo. Retorna [M,5] fundido."""
    M = len(listas)
    caixas = [(b, k) for k, L in enumerate(listas) for b in L]
    caixas.sort(key=lambda t: -t[0][4])
    clusters = []   # cada um: [lista de (box, modelo)], caixa fundida
    for b, k in caixas:
        alvo = None
        for c in clusters:
            if CH._iou(b[:4], c["f"][:4]) > iou_thr:
                alvo = c; break
        if alvo is None:
            clusters.append({"m": [(b, k)], "f": b.copy()})
        else:
            alvo["m"].append((b, k))
            B = np.array([x for x, _ in alvo["m"]]); w = B[:, 4:5]
            alvo["f"] = np.r_[(B[:, :4] * w).sum(0) / w.sum(), B[:, 4].mean()]
    out = []
    for c in clusters:
        n_mod = len({k for _, k in c["m"]})
        f = c["f"].copy(); f[4] = f[4] * min(n_mod, M) / M
        out.append(f)
    return np.array(out) if out else np.zeros((0, 5))


def gerar_cache(tag, fontes, metodo):
    f = CH.RES / "deteccao" / f"{tag}.npz"
    if f.exists():
        return
    stores = {n: dict(np.load(CH.RES / "deteccao" / f"{FONTES[n][0]}.npz")) for n in fontes}
    keys = set.intersection(*[set(s) for s in stores.values()])
    out = {}
    for k in keys:
        ls = []
        for n in fontes:
            b = CH.nms(stores[n][k], FONTES[n][1])
            ls.append(b[b[:, 4] >= 0.001] if len(b) else b)
        if metodo == "wbf":
            out[k] = wbf(ls)
        else:
            u = np.concatenate([l for l in ls if len(l)]) if any(len(l) for l in ls) else np.zeros((0, 5))
            out[k] = CH.nms(u, 0.5)
    np.savez(f, **out)


def f2(m):
    p, r = m["MILCO_precisao"], m["MILCO_recall"]; return 5 * p * r / max(4 * p + r, 1e-9)


def main(budget=150):
    t_end = time.time() + budget
    res = json.load(open(OUT)) if OUT.exists() else {}
    for tag, (fontes, metodo) in FUSOES.items():
        if tag in res:
            continue
        gerar_cache(tag, fontes, metodo)
        r = {"fontes": fontes, "metodo": metodo,
             "deteccao": {sp: CH.avaliar_deteccao(tag, sp, iou_nms=0.5, conf=0.25) for sp in ("val", "test")}}
        p = dict(nome=tag, det=tag, iou=0.5, cls=ENS, ajusta=True)
        pv, ok1 = FC.preds_cache(p, "val", t_end); pt, ok2 = FC.preds_cache(p, "test", t_end) if ok1 else ({}, False)
        if not (ok1 and ok2):
            print(f"{tag}: classificando recortes… (continua na próxima chamada)"); json.dump(res, open(OUT, "w"), indent=1); return False
        # teto (varredura)
        for sp, pr in (("val", pv), ("test", pt)):
            m = CH.avaliar_pipeline(FC.filtrar(pr, 0.05), sp, 0.0)
            r[f"varredura_{sp}"] = {k: m[k] for k in ("MILCO_tp", "MILCO_fp", "n_MILCO", "MILCO_recall", "MILCO_precisao")}
        # pontos de operação escolhidos na validação
        for crit, fn in (("F1", lambda m: m["MILCO_f1"]), ("F2", f2)):
            best = None
            for c in FC.CONF_GRID:
                for a in FC.ACEITA_GRID:
                    mv = CH.avaliar_pipeline(FC.filtrar(pv, c), "val", a)
                    k = (fn(mv), mv["MILCO_recall"])
                    if best is None or k > best[0]:
                        best = (k, c, a, mv)
            _, c, a, mv = best
            mt = CH.avaliar_pipeline(FC.filtrar(pt, c), "test", a)
            r[crit] = dict(conf=c, aceita=a, val=FC._enxuto(mv), test=FC._enxuto(mt), val_score=fn(mv), test_score=fn(mt))
        res[tag] = r
        json.dump(res, open(OUT, "w"), indent=1)
        v, t = r["varredura_val"], r["varredura_test"]
        print(f"{tag:28s} varredura val {v['MILCO_tp']}/{v['n_MILCO']} FP {v['MILCO_fp']} | teste {t['MILCO_tp']}/{t['n_MILCO']} FP {t['MILCO_fp']}"
              f" || F2: val {r['F2']['val_score']:.3f} teste {r['F2']['test']['MILCO_tp']}/75 FP {r['F2']['test']['MILCO_fp']}"
              f" || F1: val {r['F1']['val_score']:.3f} teste {r['F1']['test']['MILCO_tp']}/75 FP {r['F1']['test']['MILCO_fp']} F1 {r['F1']['test_score']:.3f}")
        if time.time() > t_end - 20:
            return False
    print("Fusões concluídas →", OUT)
    return True


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 150)
