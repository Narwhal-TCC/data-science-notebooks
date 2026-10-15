"""
pipeline_inferencia.py — fluxo de 2 etapas para imagens de sonar de varredura lateral:
    1) YOLO localiza objetos (caixas);
    2) cada caixa é recortada (com margem) e um classificador ConvNeXt decide MILCO (mina) × NOMBO (não-mina).

Uso em linha de comando:
    python pipeline_inferencia.py img1.jpg pasta_de_imagens/ ... [--anotar pasta_saida/]
    → CSV no stdout: arquivo, x1, y1, x2, y2, conf_detector, prob_MILCO, classe
      e, com --anotar, uma cópia de cada imagem com as caixas desenhadas (vermelho = MILCO, azul = NOMBO).

Uso como biblioteca (por exemplo, no Streamlit):
    from pipeline_inferencia import Pipeline
    pipe = Pipeline()                       # carrega uma vez (st.cache_resource)
    deteccoes = pipe.analisar(imagem_pil)   # lista de dicts
    imagem_anotada = pipe.desenhar(imagem_pil, deteccoes)

Requer: torch, timm, ultralytics, pillow, numpy. Os pesos ficam em ./modelos/ (veja pipeline_config.json).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def _caixa_recorte(box, w, h, modo):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if modo.startswith("m"):
        m = float(modo[1:])
        x1, y1, x2, y2 = x1 - m * bw, y1 - m * bh, x2 + m * bw, y2 + m * bh
    else:   # "qK": janela quadrada K× o maior lado
        k = float(modo[1:]); s = max(k * max(bw, bh), 32); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2
    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))   # igual ao crop_harness.caixa_recorte (o PIL arredonda)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _nms(b, iou):
    import torch
    import torchvision
    if len(b) == 0:
        return b
    keep = torchvision.ops.nms(torch.tensor(b[:, :4], dtype=torch.float32), torch.tensor(b[:, 4], dtype=torch.float32), iou).numpy()
    return b[keep]


def _wbf(listas, iou_thr=0.55):
    """Weighted Boxes Fusion (Solovyev et al., 2021): funde as caixas de vários detectores (cada uma [N,5])."""
    M = len(listas)
    caixas = sorted([(b, k) for k, L in enumerate(listas) for b in L], key=lambda t: -t[0][4])
    clusters = []
    for b, k in caixas:
        alvo = next((c for c in clusters if _iou(b[:4], c["f"][:4]) > iou_thr), None)
        if alvo is None:
            clusters.append({"m": [(b, k)], "f": b.copy()})
        else:
            alvo["m"].append((b, k))
            B = np.array([x for x, _ in alvo["m"]]); w = B[:, 4:5]
            alvo["f"] = np.r_[(B[:, :4] * w).sum(0) / w.sum(), B[:, 4].mean()]
    out = []
    for c in clusters:
        f = c["f"].copy(); f[4] = f[4] * min(len({k for _, k in c["m"]}), M) / M
        out.append(f)
    return np.array(out) if out else np.zeros((0, 5))


def _classificador(arquivo):
    import timm
    import torch
    import torch.nn as nn
    ck = torch.load(arquivo, map_location="cpu")
    cfg = ck["cfg"]
    nomes = cfg["backbone"] if isinstance(cfg["backbone"], list) else [cfg["backbone"]]
    nc = 3 if cfg.get("fundo") else 2

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleList([timm.create_model(n, pretrained=False, num_classes=0) for n in nomes])
            stats = []
            for b in self.backbone:
                dc = timm.data.resolve_data_config({}, model=b)
                stats.append((torch.tensor(dc["mean"]).view(1, 3, 1, 1), torch.tensor(dc["std"]).view(1, 3, 1, 1)))
            self.stats = stats
            with torch.no_grad():
                nf = sum(b.eval()(torch.zeros(1, 3, 64, 64)).shape[1] for b in self.backbone)
            if len(nomes) == 1:
                self.head = nn.Sequential(nn.Dropout(0.0), nn.Linear(nf, nc))
            else:
                self.head = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(inplace=True), nn.Dropout(0.0), nn.Linear(256, nc))

        def forward(self, x):
            return self.head(torch.cat([b((x - mu) / sd) for b, (mu, sd) in zip(self.backbone, self.stats)], 1))

    m = Net()
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return m.eval(), cfg


class Pipeline:
    def __init__(self, config=HERE / "modelos" / "pipeline_config.json"):
        from ultralytics import YOLO
        self.cfg = json.load(open(config))
        base = Path(config).parent
        # "detector": um só (formato antigo) | "detectores": vários, fundidos por WBF (ver "fusao")
        self.dets_cfg = self.cfg.get("detectores") or [self.cfg["detector"]]
        self.dets = []
        for d in self.dets_cfg:
            p = Path(d["pesos"])
            self.dets.append(YOLO(str(p if p.is_absolute() else (base / p).resolve())))
        self.cls = [_classificador(base / f) for f in self.cfg["classificador"]["arquivos"]]

    def analisar(self, imagem):
        import torch
        import torchvision
        im = Image.open(imagem).convert("RGB") if not isinstance(imagem, Image.Image) else imagem.convert("RGB")
        w, h = im.size
        listas = []
        for dc, det in zip(self.dets_cfg, self.dets):
            r = det.predict(im, imgsz=dc["imgsz"], conf=0.001 if len(self.dets) > 1 else dc["conf"], iou=0.9,
                            augment=dc.get("augment", False), device="cpu", verbose=False)[0]
            b = np.concatenate([r.boxes.xyxy.numpy(), r.boxes.conf.numpy()[:, None]], 1) if len(r.boxes) else np.zeros((0, 5))
            listas.append(_nms(b, dc["iou_nms"]))
        if len(listas) == 1:
            b = listas[0]
        else:
            fz = self.cfg.get("fusao", {})
            b = _nms(_wbf(listas, fz.get("iou", 0.55)), fz.get("iou_nms", 0.5))
            b = b[b[:, 4] >= fz["conf"]] if len(b) else b
        if not len(b):
            return []
        b, s = b[:, :4], b[:, 4]
        c = self.cfg["classificador"]
        X = np.stack([np.asarray(im.crop(_caixa_recorte(x, w, h, c["modo"])).resize((c["img_size"],) * 2, Image.BILINEAR)) for x in b])
        x = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            P = np.mean([torch.softmax(m(x), 1).numpy() for m, _ in self.cls], 0)
        out = []
        for box, conf, p in zip(b, s, P):
            k = int(np.argmax(p))
            if k == 0 and p[0] < c["aceita_min"]:
                k = 1
            classe = ["MILCO", "NOMBO", "fundo"][k]
            if classe == "fundo":
                continue
            out.append(dict(caixa=[float(v) for v in box], conf_detector=float(conf), prob_MILCO=float(p[0]), classe=classe))
        return out

    @staticmethod
    def desenhar(imagem, deteccoes):
        im = (Image.open(imagem) if not isinstance(imagem, Image.Image) else imagem).convert("RGB").copy()
        dr = ImageDraw.Draw(im)
        for d in deteccoes:
            cor = (230, 40, 40) if d["classe"] == "MILCO" else (40, 120, 230)
            dr.rectangle(d["caixa"], outline=cor, width=2)
            dr.text((d["caixa"][0], max(0, d["caixa"][1] - 11)), f"{d['classe']} {d['prob_MILCO']:.2f}", fill=cor)
        return im


if __name__ == "__main__":
    args = sys.argv[1:]
    saida = None
    if "--anotar" in args:
        i = args.index("--anotar"); saida = Path(args[i + 1]); del args[i:i + 2]
        saida.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for a in args:
        p = Path(a)
        arquivos += sorted(list(p.glob("*.jpg")) + list(p.glob("*.png"))) if p.is_dir() else [p]
    pipe = Pipeline()
    print("arquivo,x1,y1,x2,y2,conf_detector,prob_MILCO,classe")
    for a in arquivos:
        ds = pipe.analisar(a)
        for d in ds:
            print(f"{a},{','.join(f'{v:.0f}' for v in d['caixa'])},{d['conf_detector']:.3f},{d['prob_MILCO']:.3f},{d['classe']}")
        if saida:
            pipe.desenhar(a, ds).save(saida / Path(a).name)
