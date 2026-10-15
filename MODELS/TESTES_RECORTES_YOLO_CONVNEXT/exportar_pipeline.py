"""
exportar_pipeline.py — grava em ./modelos/ o fluxo escolhido na validação (resultados/pipeline/fase_c.json):
  - classificador_recortes_s{42,43,44}.pt  (state_dict em fp16 + cfg; ensemble das 3 sementes)
  - pipeline_config.json                   (detector, conf, NMS, recorte, limiar de aceite)
Depois disso, pipeline_inferencia.py roda sozinho (sem crop_harness).
"""
import json
import os
from pathlib import Path

import torch

import crop_harness as CH
from fase_c import _job42

HERE = Path(__file__).resolve().parent
OUT = HERE / "modelos"
OUT.mkdir(exist_ok=True)
fc = json.load(open(CH.RES / "pipeline" / "fase_c.json"))
P = fc["pipelines"]["P4_novo"]
src = fc["escolhido"][len("C_"):]
arquivos = []
for job, s in ((_job42(src), 42), (f"f_{src}_s43", 43), (f"f_{src}_s44", 44)):
    r = json.load(open(CH.RES / "runs" / f"{job}.json"))
    sd = torch.load(CH.WORK / job / "best.pt", map_location="cpu")
    nome = f"classificador_recortes_{src}_s{s}.pt"
    torch.save({"cfg": r["cfg"], "state_dict": {k: v.half() for k, v in sd.items()}, "job": job,
                "val_auc": r["val"]["auc_MILCO_vs_NOMBO"]}, OUT / nome)
    arquivos.append(nome)
cfg0 = json.load(open(CH.RES / "runs" / f"{_job42(src)}.json"))["cfg"]
det = CH.DET_CONFIGS[P["det"]]
config = {
    "descricao": "Fluxo de 2 etapas: YOLO (Modelo A, TTA) localiza → recorte com margem → ConvNeXt-Pico (3 sementes) classifica MILCO × NOMBO",
    "detector": {"pesos": os.path.relpath(CH.YOLO_A, OUT), "imgsz": det["imgsz"], "augment": det.get("augment", False),
                 "conf": P["conf"], "iou_nms": P["iou_nms"]},
    "classificador": {"arquivos": arquivos, "modo": cfg0.get("modo", "m0.15"), "img_size": cfg0.get("img_size", 224),
                      "aceita_min": P["aceita_min"]},
    "metricas_teste": {k: P["test"][k] for k in ("MILCO_precisao", "MILCO_recall", "MILCO_f1", "MILCO_AP50", "img_auc")},
    "origem": "MODELS/TESTES_RECORTES_YOLO_CONVNEXT/resultados/pipeline/fase_c.json (P4_novo; conf e limiar escolhidos na validação)",
}
json.dump(config, open(OUT / "pipeline_config.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(config, indent=1, ensure_ascii=False))
