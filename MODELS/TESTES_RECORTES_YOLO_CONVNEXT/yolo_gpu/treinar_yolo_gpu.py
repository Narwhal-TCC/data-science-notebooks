"""
treinar_yolo_gpu.py — retreino do detector (etapa de RECORTE) em GPU NVIDIA.

Usa EXATAMENTE o mesmo split do notebook sonar_deteccao_comparacao_modelos_V3_AA.ipynb
(arquivos ordenados por nome, random.Random(42).shuffle, 70/15/15 → 176 imagens de teste),
classe única "object" (o tipo MILCO/NOMBO é decidido depois pelo classificador).

Variantes (cada uma retomável; o que já terminou é pulado):
  G1  yolov8n-p2  640  (referência = Modelo A da apresentação)
  G2  yolov8n-p2 1024
  G3  yolov8s-p2 1024
  G4  yolo11s    1024
  G5  melhor de G2–G4 + augmentação p/ objetos pequenos (mosaic 0.3, copy_paste 0.3, scale 0.3)
  G6  melhor de G2–G4 + AdamW (lr0 0.001)

Uso:
  pip install -r requirements_gpu.txt
  python treinar_yolo_gpu.py --zip C:/caminho/24574879.zip            # tudo
  python treinar_yolo_gpu.py --zip ... --so G1,G2                       # só algumas
  python treinar_yolo_gpu.py --zip ... --smoke --device cpu             # teste rápido (1 época)
  python treinar_yolo_gpu.py --zip PASTA_COM_ZIPS_POR_ANO --so G2 --device 0 --horas 1.5 --cache   # ex.: Kaggle
Saída: ./saida/ (resultados_yolo_gpu.csv, pesos/Gx_best.pt, runs/) — copie a pasta saida/ de volta para
MODELS/TESTES_RECORTES_YOLO_CONVNEXT/yolo_gpu/saida/ no repositório.
"""
import argparse
import csv
import glob
import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
SEED = 42
VARIANTES = {
    "G1": dict(modelo="yolov8n-p2.yaml", base="yolov8n.pt", imgsz=640),
    "G2": dict(modelo="yolov8n-p2.yaml", base="yolov8n.pt", imgsz=1024),
    "G3": dict(modelo="yolov8s-p2.yaml", base="yolov8s.pt", imgsz=1024),
    "G4": dict(modelo="yolo11s.pt", base=None, imgsz=1024),
    "G5": dict(derivada=True, extra=dict(mosaic=0.3, copy_paste=0.3, scale=0.3, flipud=0.0)),
    "G6": dict(derivada=True, extra=dict(optimizer="AdamW", lr0=0.001)),
}


# ------------------------------------------------------------------ dataset
def preparar_dataset(zip_path, work):
    raw = work / "raw"
    if not raw.exists():
        zip_path = Path(zip_path)
        if zip_path.is_dir():          # pasta com os zips por ano (download direto do figshare)
            tmp, apagar = zip_path, False
        else:                          # zip único "baixar tudo" do figshare
            tmp, apagar = work / "_zip", True
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
        for y in ("2010", "2015", "2017", "2018", "2021"):
            with zipfile.ZipFile(tmp / f"{y}.zip") as z:
                z.extractall(raw)
        if apagar:
            shutil.rmtree(tmp)
    imgs = sorted(glob.glob(str(raw / "**" / "*.jpg"), recursive=True), key=os.path.basename)
    assert len(imgs) == 1170, f"esperava 1170 imagens, achei {len(imgs)}"
    sh = imgs[:]
    random.Random(SEED).shuffle(sh)
    n = len(sh)
    ntr, nva = int(n * 0.70), int(n * 0.15)
    splits = {"train": sh[:ntr], "val": sh[ntr:ntr + nva], "test": sh[ntr + nva:]}
    assert len(splits["test"]) == 176
    ds = work / "yolo_dataset"
    for s, lst in splits.items():
        (ds / "images" / s).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / s).mkdir(parents=True, exist_ok=True)
        for p in lst:
            stem = Path(p).stem
            dst = ds / "images" / s / f"{stem}.jpg"
            if not dst.exists():
                shutil.copy(p, dst)
            lab = ds / "labels" / s / f"{stem}.txt"
            if lab.exists():          # já preparado (ex.: 2 processos em GPUs diferentes)
                continue
            linhas = []
            for l in open(p[:-4] + ".txt").read().splitlines():
                q = l.split()
                if len(q) >= 5:
                    linhas.append("0 " + " ".join(q[1:5]))   # classe única; coords já YOLO (centro, normalizado)
            lab.write_text("\n".join(linhas))
    (ds / "data.yaml").write_text(f"path: {ds.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: object\n")
    json.dump({k: [Path(p).name for p in v] for k, v in splits.items()}, open(work / "split.json", "w"), indent=1)
    return ds


# ------------------------------------------------------------------ avaliação (independente do ultralytics)
def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def avaliar(model, ds, split, imgsz, device, conf_op=0.25):
    """AP50 (todos os pontos), recall/precisão no conf operacional, recall de objetos pequenos (<32² px)."""
    from PIL import Image
    dets, gts, n_gt, n_small, tp_small = [], {}, 0, 0, 0
    tp_op = fp_op = 0
    for img in sorted((ds / "images" / split).glob("*.jpg")):
        w, h = Image.open(img).size
        g = []
        for l in (ds / "labels" / split / f"{img.stem}.txt").read_text().splitlines():
            q = l.split()
            if len(q) == 5:
                xc, yc, bw, bh = map(float, q[1:])
                g.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        gts[img.stem] = g
        n_gt += len(g)
        r = model.predict(str(img), imgsz=imgsz, conf=0.001, iou=0.6, device=device, verbose=False)[0]
        bx = r.boxes.xyxy.cpu().numpy().tolist()
        sc = r.boxes.conf.cpu().numpy().tolist()
        dets += [(img.stem, s, b) for s, b in zip(sc, bx)]
        usado = set()
        for s, b in sorted(zip(sc, bx), key=lambda t: -t[0]):
            if s < conf_op:
                break
            j = max(range(len(g)), key=lambda k: _iou(b, g[k]), default=None)
            if j is not None and j not in usado and _iou(b, g[j]) >= 0.5:
                usado.add(j); tp_op += 1
            else:
                fp_op += 1
        for k, gb in enumerate(g):
            if (gb[2] - gb[0]) * (gb[3] - gb[1]) < 32 * 32:
                n_small += 1
                tp_small += k in usado
    dets.sort(key=lambda t: -t[1])
    usados = {k: set() for k in gts}
    tps = []
    for stem, s, b in dets:
        g = gts[stem]
        j = max(range(len(g)), key=lambda k: _iou(b, g[k]), default=None)
        ok = j is not None and j not in usados[stem] and _iou(b, g[j]) >= 0.5
        if ok:
            usados[stem].add(j)
        tps.append(ok)
    tps = np.array(tps, dtype=float)
    ctp = np.cumsum(tps); cfp = np.cumsum(1 - tps)
    rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap50 = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    n_img = len(gts)
    return {"split": split, "AP50": ap50, "recall@0.25": tp_op / max(n_gt, 1),
            "precisao@0.25": tp_op / max(tp_op + fp_op, 1), "FP_por_imagem@0.25": fp_op / max(n_img, 1),
            "recall_pequenos@0.25": tp_small / max(n_small, 1), "n_objetos": n_gt, "n_pequenos": n_small}


# ------------------------------------------------------------------ treino
def treinar(nome, cfg, ds, saida, args):
    from ultralytics import YOLO
    runs = saida / "runs"
    feito = runs / nome / "FEITO.json"
    if feito.exists():
        return json.load(open(feito))
    last = runs / nome / "weights" / "last.pt"
    kw = dict(data=str(ds / "data.yaml"), epochs=1 if args.smoke else args.epochs, imgsz=320 if args.smoke else cfg["imgsz"],
              batch=args.batch or (16 if cfg["imgsz"] <= 640 else 8), device=args.device, single_cls=True,
              mosaic=0.0, flipud=0.0, patience=50, project=str(runs), name=nome, exist_ok=True, seed=SEED,
              workers=0 if args.smoke else args.workers, plots=not args.smoke)
    if args.horas > 0:
        kw["time"] = args.horas          # teto de tempo de treino (h); sobrepõe epochs
    if args.cache:
        kw["cache"] = "ram"
    if args.smoke:
        kw["fraction"] = 0.1
    kw.update(cfg.get("extra", {}))
    t0 = time.time()
    if last.exists() and not args.smoke:
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(cfg["modelo"])
        if cfg.get("base"):
            model.load(cfg["base"])
        model.train(**kw)
    best = runs / nome / "weights" / "best.pt"
    model = YOLO(str(best))
    imgsz = kw["imgsz"]
    res = {"variante": nome, "modelo": cfg["modelo"], "imgsz": imgsz, "extra": cfg.get("extra", {}),
           "treino_min": round((time.time() - t0) / 60, 1),
           "val": avaliar(model, ds, "val", imgsz, args.device), "test": avaliar(model, ds, "test", imgsz, args.device)}
    (saida / "pesos").mkdir(parents=True, exist_ok=True)
    shutil.copy(best, saida / "pesos" / f"{nome}_best.pt")
    json.dump(res, open(feito, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--saida", default=str(AQUI / "saida"))
    ap.add_argument("--work", default=str(AQUI / "work"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--so", default="G1,G2,G3,G4,G5,G6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--horas", type=float, default=0.0, help="teto de tempo por variante, em horas (0 = sem teto)")
    ap.add_argument("--cache", action="store_true", help="cache das imagens em RAM")
    args = ap.parse_args()
    saida, work = Path(args.saida), Path(args.work)
    saida.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    ds = preparar_dataset(args.zip, work)
    resultados = {}
    for nome in args.so.split(","):
        cfg = dict(VARIANTES[nome])
        if cfg.get("derivada"):   # G5/G6 partem da melhor de G2–G4 (escolhida na VALIDAÇÃO)
            base = [resultados.get(g) or (json.load(open(saida / "runs" / g / "FEITO.json"))
                    if (saida / "runs" / g / "FEITO.json").exists() else None) for g in ("G2", "G3", "G4")]
            base = [b for b in base if b]
            if not base:
                print(f"{nome}: precisa de G2–G4 antes; pulando"); continue
            melhor = max(base, key=lambda r: (r["val"]["recall_pequenos@0.25"] + r["val"]["AP50"]))
            cfg = dict(VARIANTES[melhor["variante"]], extra=VARIANTES[nome]["extra"])
            print(f"{nome}: baseada em {melhor['variante']}")
        print(f"=== {nome}: {cfg}")
        resultados[nome] = treinar(nome, cfg, ds, saida, args)
        print(json.dumps(resultados[nome]["test"]))
    linhas = []
    for f in sorted((saida / "runs").glob("G*/FEITO.json")):
        r = json.load(open(f))
        for sp in ("val", "test"):
            linhas.append({"variante": r["variante"], "modelo": r["modelo"], "imgsz": r["imgsz"], "extra": json.dumps(r["extra"]),
                           "treino_min": r["treino_min"], **r[sp]})
    with open(saida / "resultados_yolo_gpu.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader(); w.writerows(linhas)
    print("OK →", saida / "resultados_yolo_gpu.csv")


if __name__ == "__main__":
    main()
