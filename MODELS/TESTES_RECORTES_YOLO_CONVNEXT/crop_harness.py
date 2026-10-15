"""
crop_harness.py — fluxo de RECORTES: detecção (YOLO) → classificação do recorte (MILCO × NOMBO [× fundo]).

Reaproveita o sonar_harness.py (augmentations, SAM, otimizadores, LR cosine) e acrescenta:
- split idêntico ao notebook do YOLO (arquivos ordenados por nome, Random(42).shuffle, 70/15/15);
- geração de recortes a partir das anotações verdadeiras (margem relativa ou janela quadrada) + recortes de fundo;
- classificadores de recorte: 1 backbone ou FUSÃO de 2 backbones (features concatenadas, como a fusão antiga);
- treino fatiado e retomável (fila_recortes.json), métricas multiclasse;
- detecção: predições do YOLO em cache (por configuração) e avaliação por objeto;
- pipeline ponta a ponta (detecção → recorte → classificação) com métricas por objeto e por imagem.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TESTES_CNN_IMAGEM_INTEIRA"))
import sonar_harness as SH  # noqa: E402

RAW_DIR = Path(os.environ.get("SONAR_RAW_DIR", Path.home() / "data" / "raw"))
CACHE = Path(os.environ.get("RECORTES_CACHE", Path.home() / "data" / "recortes"))
WORK = Path(os.environ.get("RECORTES_WORK", Path.home() / "work_recortes"))
RES = Path(os.environ.get("RECORTES_RESULTS", HERE / "resultados"))
QUEUE = Path(os.environ.get("RECORTES_QUEUE", HERE / "fila_recortes.json"))
MODELS_REPO = HERE.parent
YOLO_A = MODELS_REPO / "TESTE_BOX_YOLO/runs/detect/work/runs/modelA_yolo_p2/weights/best.pt"
FUSAO_ANTIGA = MODELS_REPO / "TESTE_JUNCAO/checkpoints/fusion_melhor.pt"
CLASSES = ["MILCO", "NOMBO", "fundo"]
SEED = 42
SH.WEIGHT_FILES["ghostnet_100"] = "ghostnet_1x-d6787ad5.pth"


# =================================================================== SPLIT E ANOTAÇÕES
def split_imagens():
    """{'train': [paths], 'val': [...], 'test': [...]} — idêntico ao notebook do YOLO (176 de teste)."""
    imgs = sorted(glob.glob(str(RAW_DIR / "**" / "*.jpg"), recursive=True), key=os.path.basename)
    sh = imgs[:]
    random.Random(SEED).shuffle(sh)
    n = len(sh)
    a, b = int(n * 0.70), int(n * 0.15)
    return {"train": sh[:a], "val": sh[a:a + b], "test": sh[a + b:]}


def anotacoes(img_path, w, h):
    """[(classe, xmin, ymin, xmax, ymax) em pixels] — formato original: classe xc yc bw bh normalizados."""
    out = []
    txt = img_path[:-4] + ".txt"
    if os.path.exists(txt):
        for l in open(txt).read().splitlines():
            q = l.split()
            if len(q) >= 5:
                c = int(float(q[0]))
                xc, yc, bw, bh = map(float, q[1:5])
                out.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def caixa_recorte(box, w, h, modo="m0.15"):
    """Caixa expandida para o recorte. 'mX' = margem relativa X de cada lado (como o notebook antigo);
    'q2' = janela quadrada centrada, lado = 2 × maior lado da caixa (mínimo 32 px)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if modo.startswith("m"):
        m = float(modo[1:])
        return (max(0, x1 - bw * m), max(0, y1 - bh * m), min(w, x2 + bw * m), min(h, y2 + bh * m))
    if modo.startswith("q"):
        k = float(modo[1:])
        s = max(32.0, k * max(bw, bh))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (max(0, cx - s / 2), max(0, cy - s / 2), min(w, cx + s / 2), min(h, cy + s / 2))
    raise ValueError(modo)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# =================================================================== CACHE DE RECORTES
def construir_recortes(modo="m0.15", size=224, fundo=True):
    """Recortes das anotações verdadeiras (+ 1 recorte de fundo por objeto, sem sobrepor objetos).
    Salva X uint8 [N,S,S,3], y, split, stem, caixa. Recortes degenerados (<4 px) são descartados, como no notebook."""
    from PIL import Image
    f = CACHE / f"crops_{modo}_{size}.npz"
    if f.exists():
        d = dict(np.load(f, allow_pickle=False))
        return d
    rng = np.random.default_rng(SEED)
    X, y, sp, stem, caixa = [], [], [], [], []
    for split, lst in split_imagens().items():
        for p in lst:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            ann = anotacoes(p, w, h)
            for c, *b in ann:
                cb = caixa_recorte(b, w, h, modo)
                if cb[2] - cb[0] < 4 or cb[3] - cb[1] < 4:
                    continue
                X.append(np.asarray(im.crop(cb).resize((size, size), Image.BILINEAR)))
                y.append(c); sp.append(split); stem.append(Path(p).stem); caixa.append(b)
                if fundo:   # recorte de fundo com o mesmo tamanho, em posição aleatória sem objeto
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    for _ in range(50):
                        x0 = rng.uniform(0, max(1, w - bw)); y0 = rng.uniform(0, max(1, h - bh))
                        fb = (x0, y0, x0 + bw, y0 + bh)
                        if all(_iou(fb, a[1:]) < 0.05 for a in ann):
                            cf = caixa_recorte(fb, w, h, modo)
                            X.append(np.asarray(im.crop(cf).resize((size, size), Image.BILINEAR)))
                            y.append(2); sp.append(split); stem.append(Path(p).stem); caixa.append(fb)
                            break
    d = {"X": np.stack(X).astype(np.uint8), "y": np.array(y), "split": np.array(sp), "stem": np.array(stem),
         "caixa": np.array(caixa, dtype=np.float32)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    return d


# =================================================================== MODELOS
def _backbone(name):
    import timm
    import torch
    b = timm.create_model(name, pretrained=False, num_classes=0)
    SH._load_weights(b, name)
    dc = timm.data.resolve_data_config({}, model=b)
    b.register_buffer("mu", torch.tensor(dc["mean"]).view(1, 3, 1, 1), persistent=False)
    b.register_buffer("sd", torch.tensor(dc["std"]).view(1, 3, 1, 1), persistent=False)
    return b


class Classificador:
    """Fábrica: 1 backbone (head linear) ou fusão de 2 backbones (concat + MLP, como a fusão antiga).
    O modelo recebe imagens em [0,1]; cada ramo normaliza com a média/desvio do seu pré-treino."""
    @staticmethod
    def build(cfg):
        import torch
        import torch.nn as nn
        nc = 3 if cfg.get("fundo") else 2
        nomes = cfg["backbone"] if isinstance(cfg["backbone"], list) else [cfg["backbone"]]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.ModuleList([_backbone(n) for n in nomes])
                with torch.no_grad():   # dimensão real da saída (ex.: GhostNet tem conv_head 960→1280)
                    nf = sum(b.eval()(torch.zeros(1, 3, 64, 64)).shape[1] for b in self.backbone)
                for b in self.backbone:
                    b.train()
                if len(nomes) == 1:
                    self.head = nn.Sequential(nn.Dropout(cfg.get("drop", 0.3)), nn.Linear(nf, nc))
                else:
                    self.head = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(256, nc))
                # o harness original normaliza fora do modelo; aqui cada ramo normaliza por conta própria
                self.mean = torch.zeros(1, 3, 1, 1)
                self.std = torch.ones(1, 3, 1, 1)
                self.gray = False

            def forward(self, x):
                f = [b((x - b.mu) / b.sd) for b in self.backbone]
                return self.head(torch.cat(f, 1))
        return Net()


# =================================================================== MÉTRICAS
def metricas(y, P, nc):
    """y: rótulos verdadeiros; P: probabilidades [N,nc]. Métricas nos objetos (classes 0/1) e, se nc=3, rejeição de fundo."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    pred = P.argmax(1)
    out = {"acc": float(accuracy_score(y, pred)), "f1_macro": float(f1_score(y, pred, average="macro")),
           "matriz": confusion_matrix(y, pred, labels=list(range(nc))).tolist()}
    for k, nome in enumerate(CLASSES[:nc]):
        out[f"f1_{nome}"] = float(f1_score(y == k, pred == k, zero_division=0))
        out[f"prec_{nome}"] = float(precision_score(y == k, pred == k, zero_division=0))
        out[f"rec_{nome}"] = float(recall_score(y == k, pred == k, zero_division=0))
    obj = y < 2
    try:
        out["auc_MILCO_vs_NOMBO"] = float(roc_auc_score(y[obj] == 0, P[obj, 0]))
    except ValueError:
        out["auc_MILCO_vs_NOMBO"] = float("nan")
    # acurácia só nos objetos, decidindo MILCO × NOMBO (comparável à fusão antiga de 2 classes)
    out["acc_objetos_2cls"] = float(((P[obj, 0] >= P[obj, 1]).astype(int) == (y[obj] == 0)).mean())
    out["f1_MILCO_objetos_2cls"] = float(f1_score(y[obj] == 0, P[obj, 0] >= P[obj, 1]))
    return out


def _score_val(y, P, nc):
    """Critério de seleção na validação: AUC MILCO × NOMBO nos objetos + (3 classes) acurácia de fundo."""
    m = metricas(y, P, nc)
    s = m["auc_MILCO_vs_NOMBO"] if not math.isnan(m["auc_MILCO_vs_NOMBO"]) else 0.0
    if nc == 3:
        s = 0.5 * s + 0.5 * m["f1_macro"]
    return s, m


# =================================================================== TREINO FATIADO
DEFAULTS = dict(modo="m0.15", img_size=224, batch_size=32, epochs=30, patience=8, optimizer="adamw", lr=2e-4,
                wd=0.05, head_lr_mult=10.0, scheduler="cosine", warmup_frac=0.05, aug="standard", mix="none",
                loss="ce", soften=0.6, label_smoothing=0.0, fundo=False, seed=SEED, freeze="full")


def _dados(cfg):
    d = construir_recortes(cfg["modo"], cfg["img_size"])
    keep = np.ones(len(d["y"]), bool) if cfg.get("fundo") else d["y"] < 2
    idx = {s: np.where(keep & (d["split"] == s))[0] for s in ("train", "val", "test")}
    return d["X"], d["y"], idx


def _loss(logits, y, pesos, cfg):
    import torch
    import torch.nn.functional as F
    if cfg.get("loss") == "focal":
        ce = F.cross_entropy(logits, y, weight=pesos, reduction="none", label_smoothing=cfg.get("label_smoothing", 0.0))
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return ((1 - pt) ** cfg.get("focal_gamma", 2.0) * ce).mean()
    return F.cross_entropy(logits, y, weight=pesos, label_smoothing=cfg.get("label_smoothing", 0.0))


def _predict(model, X, idx, bs=64, tta=False):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(idx), bs):
            x = torch.from_numpy(np.ascontiguousarray(X[idx[i:i + bs]])).permute(0, 3, 1, 2).float().div_(255)
            p = torch.softmax(model(x), 1)
            if tta:
                p = 0.5 * (p + torch.softmax(model(x.flip(-1)), 1))
            out.append(p.numpy())
    return np.concatenate(out) if out else np.zeros((0, 2))


def train_job(job, deadline, log=print):
    import torch
    torch.set_num_threads(2)
    cfg = {**DEFAULTS, **job["cfg"]}
    nc = 3 if cfg.get("fundo") else 2
    X, y, idx = _dados(cfg)
    tr, va = idx["train"], idx["val"]
    d = WORK / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    state_f, best_f = d / "state.pt", d / "best.pt"
    torch.manual_seed(cfg["seed"])
    model = Classificador.build(cfg)
    opt = SH.build_optimizer(model, cfg)
    is_sam = isinstance(opt, SH.SAM)
    bs = cfg["batch_size"]
    spe = math.ceil(len(tr) / bs)
    total = spe * cfg["epochs"]
    cnt = np.bincount(y[tr], minlength=nc).astype(float)
    raw = len(tr) / (nc * np.maximum(cnt, 1))
    pesos = torch.tensor(1 + cfg["soften"] * (raw - 1), dtype=torch.float32)
    st = {"epoch": 0, "batch": 0, "step": 0, "best": -1.0, "bad": 0, "hist": [], "t": 0.0, "el": 0.0, "en": 0}
    if state_f.exists():
        ck = torch.load(state_f, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); st = ck["st"]; torch.set_rng_state(ck["rng"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 * st["epoch"] + st["batch"])
    t0 = time.time()
    fim = False
    while True:
        if st["epoch"] >= cfg["epochs"] or st["bad"] >= cfg["patience"]:
            fim = True
            break
        perm = np.random.default_rng(cfg["seed"] + st["epoch"]).permutation(tr)
        model.train()
        while st["batch"] < spe and time.time() < deadline:
            b = np.sort(perm[st["batch"] * bs:(st["batch"] + 1) * bs])
            xb = torch.from_numpy(np.ascontiguousarray(X[b])).permute(0, 3, 1, 2).float().div_(255)
            xb = SH.augment(xb, cfg["aug"], gen)
            yb = torch.from_numpy(y[b]).long()
            if cfg["scheduler"] == "cosine":
                f = SH.lr_at(cfg, st["step"], total)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * f
            loss = _loss(model(xb), yb, pesos, cfg)
            loss.backward()
            if is_sam:
                opt.first_step(); opt.zero_grad()
                _loss(model(xb), yb, pesos, cfg).backward()
                opt.second_step()
            else:
                opt.step()
            opt.zero_grad()
            st["el"] += loss.item() * len(b); st["en"] += len(b); st["batch"] += 1; st["step"] += 1
        if st["batch"] < spe:
            break
        Pv = _predict(model, X, va)
        s, m = _score_val(y[va], Pv, nc)
        st["hist"].append({"epoch": st["epoch"] + 1, "train_loss": st["el"] / max(1, st["en"]), "val_score": s,
                           "val_acc": m["acc"], "val_f1_macro": m["f1_macro"]})
        log(f"[{job['id']}] ep {st['epoch'] + 1}/{cfg['epochs']} loss {st['hist'][-1]['train_loss']:.4f} val_score {s:.4f} val_acc {m['acc']:.3f}")
        if s > st["best"] + 1e-4:
            st["best"], st["bad"] = s, 0
            torch.save(model.state_dict(), str(best_f) + ".tmp"); os.replace(str(best_f) + ".tmp", best_f)
        else:
            st["bad"] += 1
        st["epoch"] += 1; st["batch"] = 0; st["el"] = 0.0; st["en"] = 0
        gen = torch.Generator().manual_seed(cfg["seed"] + 1000 * st["epoch"])
    st["t"] += time.time() - t0
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": torch.get_rng_state()}, str(state_f) + ".tmp")
    os.replace(str(state_f) + ".tmp", state_f)
    if fim:
        if time.time() > deadline - 45:
            return False
        model.load_state_dict(torch.load(best_f, map_location="cpu"))
        t1 = time.time()
        Pt = _predict(model, X, idx["test"])
        ms = (time.time() - t1) / max(1, len(idx["test"])) * 1000
        Pv = _predict(model, X, va)
        res = {"id": job["id"], "grupo": job.get("group"), "desc": job.get("desc"), "cfg": cfg, "epocas": len(st["hist"]),
               "treino_min": st["t"] / 60, "infer_ms_recorte_cpu": ms, "best_val_score": st["best"], "hist": st["hist"],
               "n": {k: int(len(v)) for k, v in idx.items()},
               "val": metricas(y[va], Pv, nc), "test": metricas(y[idx["test"]], Pt, nc),
               "test_tta": metricas(y[idx["test"]], _predict(model, X, idx["test"], tta=True), nc)}
        (RES / "runs").mkdir(parents=True, exist_ok=True)
        json.dump(res, open(RES / "runs" / f"{job['id']}.json", "w"), indent=1, default=float)
        np.savez(RES / "runs" / f"{job['id']}_probs.npz", test_idx=idx["test"], p_test=Pt, val_idx=va, p_val=Pv)
    return fim


def run_slice(budget=150.0, log=print):
    t_end = time.time() + budget
    q = json.load(open(QUEUE))
    for job in q:
        if job.get("status") == "done":
            continue
        if time.time() > t_end - 30:
            break
        job["status"] = "running"
        try:
            done = train_job(job, t_end, log=log)
        except Exception as e:
            import traceback
            job["status"] = "error"; job["error"] = traceback.format_exc()[-1500:]
            json.dump(q, open(QUEUE, "w"), indent=1); log(f"[{job['id']}] ERRO {e}"); continue
        if done:
            job["status"] = "done"
            r = json.load(open(RES / "runs" / f"{job['id']}.json"))
            t = r["test"]
            log(f"[{job['id']}] FIM acc {t['acc']:.3f} f1_macro {t['f1_macro']:.3f} f1_MILCO {t['f1_MILCO']:.3f} "
                f"auc {t['auc_MILCO_vs_NOMBO']:.3f} | val {r['best_val_score']:.3f} ep {r['epocas']}")
            sf = WORK / job["id"] / "state.pt"
            if sf.exists():
                sf.unlink()
        tmp = QUEUE.with_suffix(".tmp"); json.dump(q, open(tmp, "w"), indent=1); os.replace(tmp, QUEUE)
        if not done:
            break
    pend = [j["id"] for j in q if j.get("status") not in ("done", "error")]
    log(f"STATUS pendentes={len(pend)} prox={pend[:1]} done={sum(j.get('status') == 'done' for j in q)} "
        f"erros={sum(j.get('status') == 'error' for j in q)}")


# =================================================================== FUSÃO ANTIGA (linha de base)
def fusao_antiga():
    import timm
    import torch
    import torch.nn as nn
    from torchvision import models

    class Fusion(nn.Module):
        def __init__(s):
            super().__init__()
            s.ghostnet = timm.create_model("ghostnetv2_100", pretrained=False, num_classes=0)
            s.mobilenet = models.mobilenet_v2(weights=None); s.mobilenet.classifier = nn.Identity()
            s.classifier = nn.Sequential(nn.Linear(2560, 256), nn.ReLU(True), nn.Dropout(0.4), nn.Linear(256, 2))
            s.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
            s.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        def forward(s, x):   # x em [0,1], como os demais classificadores daqui
            x = (x - s.mu) / s.sd
            return s.classifier(torch.cat([s.ghostnet(x), s.mobilenet(x)], 1))
    m = Fusion()
    m.load_state_dict(torch.load(FUSAO_ANTIGA, map_location="cpu"))
    return m.eval()


def carregar_classificador(job_id):
    import torch
    r = json.load(open(RES / "runs" / f"{job_id}.json"))
    m = Classificador.build(r["cfg"])
    m.load_state_dict(torch.load(WORK / job_id / "best.pt", map_location="cpu"))
    return m.eval(), r["cfg"]


# =================================================================== DETECÇÃO (cache de predições)
DET_CONFIGS = {
    "D1_640": dict(imgsz=640), "D2_960": dict(imgsz=960), "D3_1280": dict(imgsz=1280),
    "D4_640_tta": dict(imgsz=640, augment=True), "D5_tiles2x2_640": dict(imgsz=640, tiles=2),
}


def _predizer_imagem(det, p, c):
    """Caixas [x1,y1,x2,y2,conf] com conf ≥ 0.001 e NMS mínimo (IoU 0.9); NMS final é refeito depois."""
    from PIL import Image
    if not c.get("tiles"):
        r = det.predict(p, imgsz=c["imgsz"], conf=0.001, iou=0.9, augment=c.get("augment", False), device="cpu", verbose=False)[0]
        return np.concatenate([r.boxes.xyxy.numpy(), r.boxes.conf.numpy()[:, None]], 1) if len(r.boxes) else np.zeros((0, 5))
    im = Image.open(p).convert("RGB")
    W, H = im.size
    k, ov = c["tiles"], 0.2
    tw, th = W / (k - (k - 1) * ov), H / (k - (k - 1) * ov)
    out = []
    for i in range(k):
        for j in range(k):
            x0, y0 = i * tw * (1 - ov), j * th * (1 - ov)
            tile = im.crop((x0, y0, x0 + tw, y0 + th))
            r = det.predict(tile, imgsz=c["imgsz"], conf=0.001, iou=0.9, device="cpu", verbose=False)[0]
            if len(r.boxes):
                b = r.boxes.xyxy.numpy() + np.array([x0, y0, x0, y0])
                out.append(np.concatenate([b, r.boxes.conf.numpy()[:, None]], 1))
    full = det.predict(p, imgsz=c["imgsz"], conf=0.001, iou=0.9, device="cpu", verbose=False)[0]   # + imagem inteira
    if len(full.boxes):
        out.append(np.concatenate([full.boxes.xyxy.numpy(), full.boxes.conf.numpy()[:, None]], 1))
    return np.concatenate(out) if out else np.zeros((0, 5))


def cache_deteccoes(nome_cfg, splits=("val", "test"), pesos=None, budget=150.0, tag=None):
    """Prediz e guarda por imagem (retomável). Retorna True quando terminou."""
    from ultralytics import YOLO
    t_end = time.time() + budget
    tag = tag or nome_cfg
    f = RES / "deteccao" / f"{tag}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    store = dict(np.load(f)) if f.exists() else {}
    det = None
    sp = split_imagens()
    for s in splits:
        for p in sp[s]:
            k = f"{s}|{Path(p).stem}"
            if k in store:
                continue
            if time.time() > t_end - 10:
                np.savez(f, **store)
                return False
            det = det or YOLO(str(pesos or YOLO_A))
            store[k] = _predizer_imagem(det, p, DET_CONFIGS[nome_cfg])
    np.savez(f, **store)
    return True


def nms(b, iou):
    import torch
    import torchvision
    if len(b) == 0:
        return b
    keep = torchvision.ops.nms(torch.tensor(b[:, :4], dtype=torch.float32), torch.tensor(b[:, 4], dtype=torch.float32), iou).numpy()
    return b[keep]


def gt_split(split):
    from PIL import Image
    out = {}
    for p in split_imagens()[split]:
        w, h = Image.open(p).size
        out[Path(p).stem] = (p, anotacoes(p, w, h), (w, h))
    return out


def avaliar_deteccao(tag, split, iou_nms=0.6, conf=0.25):
    """AP50 (classe única), recall/precisão/FP por imagem no conf dado, recall de objetos pequenos (<32² px)."""
    store = dict(np.load(RES / "deteccao" / f"{tag}.npz"))
    G = gt_split(split)
    dets, n_gt, tp, fp, n_small, tp_small = [], 0, 0, 0, 0, 0
    for stem, (p, ann, _) in G.items():
        b = nms(store[f"{split}|{stem}"], iou_nms)
        g = [a[1:] for a in ann]
        n_gt += len(g)
        usado = set()
        for x in b[np.argsort(-b[:, 4])] if len(b) else []:
            j = max(range(len(g)), key=lambda k: _iou(x[:4], g[k]), default=None)
            ok = j is not None and j not in usado and _iou(x[:4], g[j]) >= 0.5
            if ok:
                usado.add(j)
            dets.append((x[4], ok))
            if x[4] >= conf:
                tp += ok; fp += (not ok)
        n_small += sum(1 for gb in g if (gb[2] - gb[0]) * (gb[3] - gb[1]) < 32 * 32)
    # recall de pequenos no conf operacional
    for stem, (p, ann, _) in G.items():
        b = nms(store[f"{split}|{stem}"], iou_nms)
        b = b[b[:, 4] >= conf] if len(b) else b
        for a in ann:
            gb = a[1:]
            if (gb[2] - gb[0]) * (gb[3] - gb[1]) < 32 * 32 and any(_iou(x[:4], gb) >= 0.5 for x in b):
                tp_small += 1
    dets.sort(key=lambda t: -t[0])
    oks = np.array([o for _, o in dets], float)
    ctp, cfp = np.cumsum(oks), np.cumsum(1 - oks)
    rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.r_[0, rec, 1], np.r_[1, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return {"tag": tag, "split": split, "iou_nms": iou_nms, "conf": conf, "AP50": float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])),
            "recall": tp / max(n_gt, 1), "precisao": tp / max(tp + fp, 1), "FP_por_img": fp / len(G),
            "recall_pequenos": tp_small / max(n_small, 1), "n_obj": n_gt}


# =================================================================== PIPELINE PONTA A PONTA
def recortar_e_classificar(classificar, tag, split, conf, iou_nms, modo, size, budget=1e9, cache_tag=None):
    """Para cada detecção (conf ≥ conf após NMS): recorte (modo/size) → probabilidades do classificador.
    Retorna {stem: [(x1,y1,x2,y2,conf_det, p0, p1[, p2])]} (cache em disco, retomável)."""
    from PIL import Image
    t_end = time.time() + budget
    f = RES / "pipeline" / f"{cache_tag}_{split}.json" if cache_tag else None
    out = json.load(open(f)) if f and f.exists() else {}
    store = dict(np.load(RES / "deteccao" / f"{tag}.npz"))
    for p in split_imagens()[split]:
        stem = Path(p).stem
        if stem in out:
            continue
        if time.time() > t_end - 10:
            break
        b = nms(store[f"{split}|{stem}"], iou_nms)
        b = b[b[:, 4] >= conf] if len(b) else b
        res = []
        if len(b):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            X = np.stack([np.asarray(im.crop(caixa_recorte(x[:4], w, h, modo)).resize((size, size), Image.BILINEAR)) for x in b])
            P = classificar(X)
            res = [list(map(float, x)) + list(map(float, pp)) for x, pp in zip(b, P)]
        out[stem] = res
    if f:
        f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(f, "w"))
    return out, len(out) == len(split_imagens()[split])


def classificador_fn(model, tta=False, bs=32):
    import torch

    def f(X):
        model.eval()
        P = []
        with torch.inference_mode():
            for i in range(0, len(X), bs):
                x = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).permute(0, 3, 1, 2).float().div_(255)
                p = torch.softmax(model(x), 1)
                if tta:
                    p = 0.5 * (p + torch.softmax(model(x.flip(-1)), 1))
                P.append(p.numpy())
        return np.concatenate(P)
    return f


def avaliar_pipeline(preds, split, aceita_min=0.0):
    """Métricas ponta a ponta.
    Por objeto (IoU ≥ 0.5): uma detecção vira 'MILCO' se p(MILCO) é a maior prob. (e ≥ aceita_min);
    'fundo' (3 classes) descarta a detecção. AP50 de MILCO usa score = conf_det × p(MILCO).
    Por imagem: score = max(conf_det × p(MILCO)); rótulo = imagem tem MILCO."""
    from sklearn.metrics import f1_score, roc_auc_score
    G = gt_split(split)
    tp = fp = fn = 0
    conf_m = {"MILCO→MILCO": 0, "MILCO→NOMBO": 0, "MILCO→perdido": 0, "NOMBO→NOMBO": 0, "NOMBO→MILCO": 0, "NOMBO→perdido": 0}
    ap_list, n_milco = [], 0
    img_y, img_s = [], []
    for stem, (p, ann, _) in G.items():
        d = preds.get(stem, [])
        g = [(a[0], a[1:]) for a in ann]
        n_milco += sum(1 for c, _ in g if c == 0)
        # atribuição gulosa por conf de detecção
        usado = {}
        for i, x in sorted(enumerate(d), key=lambda t: -t[1][4]):
            j = max(range(len(g)), key=lambda k: _iou(x[:4], g[k][1]), default=None)
            if j is not None and j not in usado.values() and _iou(x[:4], g[j][1]) >= 0.5:
                usado[i] = j
        for i, x in enumerate(d):
            probs = x[5:]
            cls = int(np.argmax(probs))
            if cls == 0 and probs[0] < aceita_min:
                cls = 1
            casou = i in usado
            gc = g[usado[i]][0] if casou else None
            if cls == 0:
                if casou and gc == 0:
                    tp += 1
                else:
                    fp += 1
            ap_list.append((x[4] * probs[0], casou and gc == 0))
            if casou:
                key = f"{CLASSES[gc]}→{CLASSES[cls] if cls < 2 else 'perdido'}"
                if key in conf_m:
                    conf_m[key] += 1
        achados = set(usado.values())
        for k, (c, _) in enumerate(g):
            if k not in achados:
                conf_m[f"{CLASSES[c]}→perdido"] += 1
        img_y.append(int(any(c == 0 for c, _ in g)))
        img_s.append(max([x[4] * x[5] for x in d], default=0.0))
    # contar MILCO perdidos/mal classificados como FN
    fn = n_milco - tp
    ap_list.sort(key=lambda t: -t[0])
    oks = np.array([o for _, o in ap_list], float)
    ctp, cfp = np.cumsum(oks), np.cumsum(1 - oks)
    rec, prec = ctp / max(n_milco, 1), ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.r_[0, rec, 1], np.r_[1, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])) if len(oks) else 0.0
    P_ = tp / max(tp + fp, 1); R_ = tp / max(n_milco, 1)
    img_y, img_s = np.array(img_y), np.array(img_s)
    return {"split": split, "MILCO_precisao": P_, "MILCO_recall": R_, "MILCO_f1": 2 * P_ * R_ / max(P_ + R_, 1e-9),
            "MILCO_AP50": ap, "MILCO_tp": tp, "MILCO_fp": fp, "MILCO_fn": fn, "n_MILCO": n_milco, "confusao": conf_m,
            "img_auc": float(roc_auc_score(img_y, img_s)) if 0 < img_y.sum() < len(img_y) else float("nan"),
            "img_f1@0.5": float(f1_score(img_y, img_s >= 0.5)), "img_scores": img_s.tolist(), "img_y": img_y.tolist()}


if __name__ == "__main__":
    run_slice(float(sys.argv[1]) if len(sys.argv) > 1 else 150.0)
