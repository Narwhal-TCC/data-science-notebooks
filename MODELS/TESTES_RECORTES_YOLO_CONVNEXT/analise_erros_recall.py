"""Diagnóstico para recall: (1) teto de recall de cada detector e da UNIÃO de detectores (só localização, conf ≥ 0,05);
(2) perfil das minas perdidas pelo melhor fluxo em modo varredura (tamanho, ano, perdida no detector × no classificador)."""
import json, collections
import numpy as np
import crop_harness as CH, fase_c as FC
from pathlib import Path
CH.DET_CONFIGS.update({"E_G5_1024": dict(imgsz=1024), "E_G5_1024_tta": dict(imgsz=1024, augment=True)})
DETS = {"YOLO A": ("D1_640", .7), "YOLO A+TTA": ("D4_640_tta", .5), "Mosaico 2x2": ("D5_tiles2x2_640", .5),
        "G5": ("E_G5_1024", .5), "G5+TTA": ("E_G5_1024_tta", .5)}
out = {}
for split in ("val", "test"):
    G = CH.gt_split(split)
    achou = {k: set() for k in DETS}
    for nome, (tag, iou) in DETS.items():
        st = dict(np.load(CH.RES / "deteccao" / f"{tag}.npz"))
        for stem, (p, ann, _) in G.items():
            b = CH.nms(st[f"{split}|{stem}"], iou); b = b[b[:, 4] >= 0.05] if len(b) else b
            for j, a in enumerate(ann):
                if a[0] == 0 and any(CH._iou(x[:4], a[1:]) >= 0.5 for x in b):
                    achou[nome].add((stem, j))
    n = sum(1 for _, (p, ann, _) in G.items() for a in ann if a[0] == 0)
    r = {k: len(v) for k, v in achou.items()}
    r["União G5+TTA ∪ YOLO A+TTA"] = len(achou["G5+TTA"] | achou["YOLO A+TTA"])
    r["União G5+TTA ∪ Mosaico"] = len(achou["G5+TTA"] | achou["Mosaico 2x2"])
    r["União de todos"] = len(set().union(*achou.values()))
    r["n_MILCO"] = n
    out[split] = r
    print(split, json.dumps(r, ensure_ascii=False))
    if split == "test":
        todos = set().union(*achou.values())
        perd = [(s, j) for s, (p, ann, _) in G.items() for j, a in enumerate(ann) if a[0] == 0 and (s, j) not in todos]
        perd_g5 = [(s, j) for s, (p, ann, _) in G.items() for j, a in enumerate(ann) if a[0] == 0 and (s, j) not in achou["G5+TTA"]]
        def perfil(lst):
            area = [np.sqrt((G[s][1][j][3]-G[s][1][j][1])*(G[s][1][j][4]-G[s][1][j][2])) for s, j in lst]
            anos = collections.Counter(s.split("_")[-1] for s, _ in lst)
            return dict(n=len(lst), lado_mediano_px=round(float(np.median(area)), 1) if area else None, anos=dict(anos))
        tod = [(s, j) for s, (p, ann, _) in G.items() for j, a in enumerate(ann) if a[0] == 0]
        print("todas as MILCO do teste:", perfil(tod))
        print("perdidas pelo G5+TTA:", perfil(perd_g5))
        print("perdidas por TODOS os detectores:", perfil(perd))
        out["perdidas"] = dict(todas=perfil(tod), g5tta=perfil(perd_g5), todos=perfil(perd))
# classificador no modo varredura (G5+TTA): minas localizadas mas rejeitadas
pt, _ = FC.preds_cache(dict(nome="x", det="E_G5_1024_tta", iou=0.5, cls="ens:b1_convnextpico_224"), "test", 1e12)
m = CH.avaliar_pipeline(FC.filtrar(pt, 0.05), "test", 0.0)
print("varredura G5+TTA confusão:", m["confusao"])
json.dump(out, open(CH.RES / "pipeline" / "analise_erros_recall.json", "w"), indent=1, ensure_ascii=False)
