"""figuras_pipeline.py — imagens de teste anotadas lado a lado: verdade | fluxo antigo (P0) | fluxo novo (P4)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import crop_harness as CH

OUT = CH.RES / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
fc = json.load(open(CH.RES / "pipeline" / "fase_c.json"))
CORES = {"MILCO": (230, 40, 40), "NOMBO": (40, 120, 230)}


def preds(nome):
    p = fc["pipelines"][nome]
    cache = json.load(open(CH.RES / "pipeline" / f"{p['det']}_iou{p['iou_nms']}__{p['cls'].replace(':', '-')}_test.json"))
    out = {}
    for st, v in cache.items():
        r = []
        for x in v:
            if x[4] < p["conf"]:
                continue
            k = int(np.argmax(x[5:]))
            if k == 2:
                continue
            r.append((x[:4], "MILCO" if (k == 0 and x[5] >= p["aceita_min"]) else "NOMBO", x[5]))
        out[st] = r
    return out


def painel(p, caixas, titulo, W=420):
    im = Image.open(p).convert("RGB")
    s = W / im.width
    im = im.resize((W, int(im.height * s)))
    d = ImageDraw.Draw(im)
    for b, c, pr in caixas:
        b = [v * s for v in b]
        d.rectangle(b, outline=CORES[c], width=3)
        d.text((b[0], max(0, b[1] - 12)), c if pr is None else f"{c} {pr:.2f}", fill=CORES[c])
    top = Image.new("RGB", (W, 22), (22, 29, 64))
    ImageDraw.Draw(top).text((6, 5), titulo, fill=(255, 255, 255))
    out = Image.new("RGB", (W, im.height + 22), "white")
    out.paste(top, (0, 0)); out.paste(im, (0, 22))
    return out


def main(n=6):
    G = CH.gt_split("test")
    antigo, novo = preds("P0_antigo"), preds("P4_novo")

    def acertos(pr, ann):
        return sum(1 for a in ann if a[0] == 0 and any(c == "MILCO" and CH._iou(b, a[1:]) >= 0.5 for b, c, _ in pr))
    ganho = []
    for st, (p, ann, _) in G.items():
        nm = sum(1 for a in ann if a[0] == 0)
        if nm:
            ganho.append((acertos(novo.get(st, []), ann) - acertos(antigo.get(st, []), ann), nm, st))
    ganho.sort(key=lambda t: (-t[0], -t[1]))
    escolha = [g[2] for g in ganho[:n - 1]] + [g[2] for g in ganho[::-1] if g[0] <= 0][:1]   # + 1 caso difícil
    linhas = []
    for st in escolha:
        p, ann, _ = G[st]
        gt = [(a[1:], "MILCO" if a[0] == 0 else "NOMBO", None) for a in ann]
        linhas.append([painel(p, gt, f"{st} - verdade"), painel(p, antigo.get(st, []), "Fluxo antigo"),
                       painel(p, novo.get(st, []), "Fluxo novo")])
    W = sum(c.width for c in linhas[0]) + 20
    H = sum(max(c.height for c in l) for l in linhas) + 10 * len(linhas)
    fig = Image.new("RGB", (W, H), "white")
    y = 0
    for l in linhas:
        x = 0
        for c in l:
            fig.paste(c, (x, y)); x += c.width + 10
        y += max(c.height for c in l) + 10
    fig.save(OUT / "antigo_vs_novo_teste.png")
    json.dump(escolha, open(OUT / "antigo_vs_novo_teste.json", "w"))
    print("OK", OUT / "antigo_vs_novo_teste.png", fig.size, escolha)


if __name__ == "__main__":
    main()
