"""
fase_e_detector_gpu.py — integra o YOLO retreinado em GPU (Kaggle, yolo_gpu/saida/) ao fluxo de recortes.

Etapas (retomável, cada chamada avança dentro do orçamento em segundos):
  1. escolhe o melhor detector na VALIDAÇÃO (AP50 + recall de objetos pequenos) entre G1–G6;
  2. guarda as detecções de val/teste em CPU (sem e com TTA, no imgsz de treino);
  3. avalia a detecção (mesmo avaliador de antes) e o fluxo ponta a ponta com o ConvNeXt já escolhido
     (conf e limiar escolhidos na validação), comparando com o P4 atual.
Uso: python fase_e_detector_gpu.py [budget]   → resultados/pipeline/fase_e.json
"""
import csv
import json
import sys
import time
from pathlib import Path

import crop_harness as CH
import fase_c as FC

SAIDA = CH.HERE / "yolo_gpu" / "saida"
OUT = CH.RES / "pipeline" / "fase_e.json"


def melhor_variante():
    linhas = list(csv.DictReader(open(SAIDA / "resultados_yolo_gpu.csv")))
    val = [l for l in linhas if l["split"] == "val"]
    m = max(val, key=lambda l: float(l["AP50"]) + float(l["recall_pequenos@0.25"]))
    return m["variante"], int(m["imgsz"]), linhas


def main(budget=150.0):
    t_end = time.time() + budget
    var, imgsz, linhas = melhor_variante()
    pesos = SAIDA / "pesos" / f"{var}_best.pt"
    res = json.load(open(OUT)) if OUT.exists() else {}
    res.update(variante=var, imgsz=imgsz, tabela_kaggle=linhas)
    cfgs = {f"E_{var}_{imgsz}": dict(imgsz=imgsz), f"E_{var}_{imgsz}_tta": dict(imgsz=imgsz, augment=True)}
    CH.DET_CONFIGS.update(cfgs)
    for tag in cfgs:                                   # 2) cache de detecções (CPU, lento: retomável)
        if not CH.cache_deteccoes(tag, pesos=pesos, budget=max(t_end - time.time(), 0), tag=tag):
            print(f"cache {tag}: em andamento"); json.dump(res, open(OUT, "w"), indent=1); return False
    res.setdefault("deteccao", {})
    for tag in cfgs:                                   # 3a) detecção
        for iou in (0.5, 0.6, 0.7):
            for sp in ("val", "test"):
                res["deteccao"][f"{tag}|{iou}|{sp}"] = CH.avaliar_deteccao(tag, sp, iou_nms=iou, conf=0.25)
    melhor_tag = max(cfgs, key=lambda t: max(res["deteccao"][f"{t}|{i}|val"]["AP50"] for i in (0.5, 0.6, 0.7)))
    melhor_iou = max((0.5, 0.6, 0.7), key=lambda i: res["deteccao"][f"{melhor_tag}|{i}|val"]["AP50"])
    fc = json.load(open(CH.RES / "pipeline" / "fase_c.json"))
    cls = fc["pipelines"]["P4_novo"]["cls"]
    res.setdefault("pipelines", {})
    p = dict(nome="P6_yolo_gpu", det=melhor_tag, iou=melhor_iou, cls=cls, ajusta=True,
             desc=f"YOLO retreinado em GPU ({var}, {imgsz}px{', TTA' if 'tta' in melhor_tag else ''}) + ConvNeXt de recortes (3 sementes)")
    if not FC.rodar(p, res, t_end):                   # 3b) ponta a ponta
        json.dump(res, open(OUT, "w"), indent=1); return False
    res["P4_referencia"] = {k: fc["pipelines"]["P4_novo"][k] for k in ("conf", "aceita_min", "val", "test")}
    res["concluido"] = True
    json.dump(res, open(OUT, "w"), indent=1)
    a, b = res["P4_referencia"]["test"], res["pipelines"]["P6_yolo_gpu"]["test"]
    print(f"P4 (atual)  : MILCO {a['MILCO_tp']}/{a['n_MILCO']}  P {a['MILCO_precisao']:.3f} R {a['MILCO_recall']:.3f} F1 {a['MILCO_f1']:.3f} | img AUC {a['img_auc']:.3f}")
    print(f"P6 (YOLO GPU): MILCO {b['MILCO_tp']}/{b['n_MILCO']}  P {b['MILCO_precisao']:.3f} R {b['MILCO_recall']:.3f} F1 {b['MILCO_f1']:.3f} | img AUC {b['img_auc']:.3f}")
    return True


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 150.0)
