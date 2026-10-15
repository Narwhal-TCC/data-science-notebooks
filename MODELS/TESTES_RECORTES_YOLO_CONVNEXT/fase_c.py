"""
fase_c.py — Fase C: pipeline ponta a ponta (detector YOLO → recorte → classificador MILCO × NOMBO).

Retomável: cada chamada avança o que der dentro do orçamento de tempo (segundos) e para.
Uso:  python fase_c.py [budget]      → imprime o estado; quando tudo termina grava resultados/pipeline/fase_c.json

Regras (as mesmas do plano aprovado):
  - o split é o do notebook YOLO (819/175/176 imagens); nada é escolhido olhando o TESTE;
  - conf do detector e limiar de aceite de MILCO são escolhidos na VALIDAÇÃO (maior F1 de MILCO por objeto);
  - pesos antigos (YOLO Modelo A e fusão GhostNetV2+MobileNetV2) são apenas lidos.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import crop_harness as CH

OUT = CH.RES / "pipeline"
CONF_MIN = 0.05
CONF_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
ACEITA_GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FINALISTAS = ["b1_convnextpico_224", "b2_label_smoothing", "b3_margem030", "b1_convnextpico_128", "b5_fundo_3classes"]


# ------------------------------------------------------------------ classificadores
def _runs(job):
    f = CH.RES / "runs" / f"{job}.json"
    return json.load(open(f)) if f.exists() else None


def _job42(src):
    """Semente 42: o job original, ou o retreino f_<src>_s42 quando o best.pt original foi apagado (disco)."""
    return src if (CH.WORK / src / "best.pt").exists() else f"f_{src}_s42"


def escolher_finalista():
    """Configuração com maior média da AUC MILCO×NOMBO na validação nas 3 sementes (42, 43, 44)."""
    tab = []
    for src in FINALISTAS:
        jobs = [_job42(src), f"f_{src}_s43", f"f_{src}_s44"]
        rs = [_runs(j) for j in jobs]
        if any(r is None for r in rs):
            return None, None
        v = [r["val"]["auc_MILCO_vs_NOMBO"] for r in rs]   # mesma métrica p/ 2 e 3 classes (AUC MILCO×NOMBO nos objetos)
        t = [r["test"]["acc"] for r in rs]
        ta = [r["test"]["auc_MILCO_vs_NOMBO"] for r in rs]
        tab.append(dict(config=src, jobs=jobs, val_media=float(np.mean(v)), val_dp=float(np.std(v)),
                        test_acc_media=float(np.mean(t)), test_acc_dp=float(np.std(t)),
                        test_auc_media=float(np.mean(ta)), test_auc_dp=float(np.std(ta))))
    tab.sort(key=lambda d: -d["val_media"])
    return tab[0], tab


_CACHE_FN = {}


def classificador(nome):
    """Retorna (função X_uint8 → probs, modo do recorte, tamanho)."""
    if nome in _CACHE_FN:
        return _CACHE_FN[nome]
    if nome == "fusao_antiga":
        fn = (CH.classificador_fn(CH.fusao_antiga()), "m0.15", 224)
    elif nome.startswith("ens:"):           # ens:<config> = média das 3 sementes
        src = nome[4:]
        ms = [CH.carregar_classificador(j) for j in (_job42(src), f"f_{src}_s43", f"f_{src}_s44")]
        fs = [CH.classificador_fn(m) for m, _ in ms]
        cfg = ms[0][1]
        fn = (lambda X: np.mean([f(X) for f in fs], 0), cfg.get("modo", "m0.15"), cfg.get("img_size", 224))
    else:                                    # job único
        m, cfg = CH.carregar_classificador(_job42(nome) if not nome.startswith("f_") else nome)
        fn = (CH.classificador_fn(m), cfg.get("modo", "m0.15"), cfg.get("img_size", 224))
    _CACHE_FN[nome] = fn
    return fn


# ------------------------------------------------------------------ pipelines
def linhas_de_base():
    return [
        # nome, tag de detecção, IoU do NMS, classificador, ajustar conf/aceite na validação?
        dict(nome="P0_antigo", det="D1_640", iou=0.7, cls="fusao_antiga", ajusta=False, conf=0.5, aceita=0.0,
             desc="Fluxo antigo como no notebook: YOLO A 640, conf 0,5 + fusão GhostNetV2+MobileNetV2"),
        dict(nome="P1_antigo_ajustado", det="D1_640", iou=0.7, cls="fusao_antiga", ajusta=True,
             desc="Fluxo antigo, mas com conf/limiar escolhidos na validação"),
        dict(nome="P2_detnovo_clsantigo", det="D4_640_tta", iou=0.5, cls="fusao_antiga", ajusta=True,
             desc="Detector novo (TTA) + classificador antigo"),
    ]


def candidatos(tab):
    """Fluxo novo com cada configuração finalista (ensemble das 3 sementes) — o escolhido é o melhor na VALIDAÇÃO."""
    return [dict(nome=f"C_{t['config']}", det="D4_640_tta", iou=0.5, cls=f"ens:{t['config']}", ajusta=True,
                 desc=f"Candidato: detector TTA + {t['config']} (3 sementes)") for t in tab]


def finais(esc):
    cfg = esc[len("C_"):]
    P = [
        dict(nome="P3_detantigo_clsnovo", det="D1_640", iou=0.7, cls=f"ens:{cfg}", ajusta=True,
             desc=f"Detector antigo + ConvNeXt de recortes ({cfg}, 3 sementes)"),
        dict(nome="P4_novo", det="D4_640_tta", iou=0.5, cls=f"ens:{cfg}", ajusta=True,
             desc=f"Fluxo novo: detector com TTA + ConvNeXt de recortes ({cfg}, 3 sementes)"),
        dict(nome="P4b_novo_1semente", det="D4_640_tta", iou=0.5, cls=cfg, ajusta=True,
             desc=f"Fluxo novo com um único modelo ({cfg}, semente 42) — inferência 3× mais barata"),
        dict(nome="P5_tiles_3classes", det="D5_tiles2x2_640", iou=0.5, cls="ens:b5_fundo_3classes", ajusta=True,
             desc="Detector em mosaico 2×2 (mais recall) + ConvNeXt de 3 classes (3 sementes) que descarta 'fundo'"),
    ]
    return P


def preds_cache(p, split, t_end):
    fn, modo, size = classificador(p["cls"])
    tagc = f"{p['det']}_iou{p['iou']}__{p['cls'].replace(':', '-')}"
    out, ok = CH.recortar_e_classificar(fn, p["det"], split, CONF_MIN, p["iou"], modo, size,
                                        budget=max(t_end - time.time(), 0), cache_tag=tagc)
    return out, ok


def filtrar(preds, conf):
    return {k: [x for x in v if x[4] >= conf] for k, v in preds.items()}


def ajustar(pv):
    melhor = None
    for c in CONF_GRID:
        for a in ACEITA_GRID:
            m = CH.avaliar_pipeline(filtrar(pv, c), "val", aceita_min=a)
            chave = (m["MILCO_f1"], m["MILCO_AP50"])
            if melhor is None or chave > melhor[0]:
                melhor = (chave, c, a)
    return melhor[1], melhor[2]


def _enxuto(m):
    return {k: v for k, v in m.items() if k not in ("img_scores", "img_y")}


def rodar(p, res, t_end):
    """Avalia um pipeline (retomável). Retorna True se terminou."""
    if p["nome"] in res["pipelines"]:
        return True
    pv, ok1 = preds_cache(p, "val", t_end)
    if not ok1:
        print(f"{p['nome']}: recortando/classificando VAL…"); return False
    pt, ok2 = preds_cache(p, "test", t_end)
    if not ok2:
        print(f"{p['nome']}: recortando/classificando TESTE…"); return False
    conf, aceita = ajustar(pv) if p["ajusta"] else (p["conf"], p["aceita"])
    mv = CH.avaliar_pipeline(filtrar(pv, conf), "val", aceita_min=aceita)
    mt = CH.avaliar_pipeline(filtrar(pt, conf), "test", aceita_min=aceita)
    res["pipelines"][p["nome"]] = dict(desc=p["desc"], det=p["det"], iou_nms=p["iou"], cls=p["cls"],
                                       conf=conf, aceita_min=aceita, val=_enxuto(mv), test=_enxuto(mt),
                                       test_img_scores=mt["img_scores"], test_img_y=mt["img_y"])
    v, t = mv, mt
    print(f"{p['nome']:34s} conf {conf} aceita {aceita} | VAL F1 {v['MILCO_f1']:.3f} | TESTE MILCO P {t['MILCO_precisao']:.3f} "
          f"R {t['MILCO_recall']:.3f} F1 {t['MILCO_f1']:.3f} AP50 {t['MILCO_AP50']:.3f} | img AUC {t['img_auc']:.3f}")
    return True


def main(budget=150.0):
    t_end = time.time() + budget
    OUT.mkdir(parents=True, exist_ok=True)
    final, tab = escolher_finalista()
    resf = OUT / "fase_c.json"
    res = json.load(open(resf)) if resf.exists() else {"pipelines": {}}
    res.update(finalistas=tab)

    def salvar():
        json.dump(res, open(resf, "w"), indent=1)

    for p in linhas_de_base():
        ok = rodar(p, res, t_end); salvar()
        if not ok or time.time() > t_end - 20:
            return False
    if tab is None:
        print("Fase C: linhas de base prontas; aguardando as sementes 43/44 dos finalistas")
        return False
    for p in candidatos(tab):
        ok = rodar(p, res, t_end); salvar()
        if not ok or time.time() > t_end - 20:
            return False
    cands = {k: v for k, v in res["pipelines"].items() if k.startswith("C_")}
    esc = max(cands, key=lambda k: (cands[k]["val"]["MILCO_f1"], cands[k]["val"]["MILCO_AP50"]))
    res["escolhido"] = esc
    for p in finais(esc):
        ok = rodar(p, res, t_end); salvar()
        if not ok or time.time() > t_end - 20:
            return False
    res["concluido"] = True
    salvar()
    print("Fase C concluída →", resf, "| escolhido na validação:", esc)
    return True


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 150.0)
