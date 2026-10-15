"""Reúne os números para a comparação "apresentação antiga (12/09) × melhor fluxo atual". Saída: resultados/figuras/comparacao_final/dados.json"""
import csv, json
import numpy as np
from pathlib import Path
import crop_harness as CH, fase_c as FC

OUT = CH.RES / "figuras" / "comparacao_final" / "dados.json"
CH.DET_CONFIGS.update({"E_G5_1024": dict(imgsz=1024), "E_G5_1024_tta": dict(imgsz=1024, augment=True)})


def ap_det(tag, split, iou_nms, iou_thr=0.5, area=None):
    """AP (todos os pontos) estilo COCO por faixa de área (GT fora da faixa = ignorado)."""
    store = dict(np.load(CH.RES / "deteccao" / f"{tag}.npz"))
    G = CH.gt_split(split)
    dets, n_gt = [], 0
    for stem, (p, ann, _) in G.items():
        b = CH.nms(store[f"{split}|{stem}"], iou_nms)
        g = [a[1:] for a in ann]
        ar = [(x[2] - x[0]) * (x[3] - x[1]) for x in g]
        ign = [area is not None and not (area[0] <= a < area[1]) for a in ar]
        n_gt += sum(1 for i in ign if not i)
        usado = set()
        for x in (b[np.argsort(-b[:, 4])] if len(b) else []):
            j = max(range(len(g)), key=lambda k: CH._iou(x[:4], g[k]), default=None)
            if j is not None and CH._iou(x[:4], g[j]) >= iou_thr and j not in usado:
                usado.add(j)
                if ign[j]:
                    continue
                dets.append((x[4], 1))
            else:
                a = (x[2] - x[0]) * (x[3] - x[1])
                if area is not None and not (area[0] <= a < area[1]):
                    continue
                dets.append((x[4], 0))
    dets.sort(key=lambda t: -t[0])
    ok = np.array([o for _, o in dets], float)
    ctp, cfp = np.cumsum(ok), np.cumsum(1 - ok)
    rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.r_[0, rec, 1], np.r_[1, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def bloco_deteccao(tag, iou_nms):
    r = {"AP50": ap_det(tag, "test", iou_nms, 0.5), "AP75": ap_det(tag, "test", iou_nms, 0.75),
         "AP50_95": float(np.mean([ap_det(tag, "test", iou_nms, t) for t in np.arange(0.5, 0.96, 0.05)])),
         "AP_pequenos": float(np.mean([ap_det(tag, "test", iou_nms, t, (0, 32 ** 2)) for t in np.arange(0.5, 0.96, 0.05)])),
         "AP_medios": float(np.mean([ap_det(tag, "test", iou_nms, t, (32 ** 2, 96 ** 2)) for t in np.arange(0.5, 0.96, 0.05)]))}   # como no COCO (IoU 0,50–0,95)
    # P/R/F1 no conf de melhor F1 escolhido na VALIDAÇÃO
    best = max(((c, CH.avaliar_deteccao(tag, "val", iou_nms, c)) for c in np.arange(0.05, 0.9, 0.05)),
               key=lambda t: 2 * t[1]["precisao"] * t[1]["recall"] / max(t[1]["precisao"] + t[1]["recall"], 1e-9))
    c = float(best[0]); m = CH.avaliar_deteccao(tag, "test", iou_nms, c)
    r.update(conf=c, precisao=m["recall"] and m["precisao"], recall=m["recall"],
             F1=2 * m["precisao"] * m["recall"] / max(m["precisao"] + m["recall"], 1e-9), FP_por_img=m["FP_por_img"])
    r["objetos_achados_teto"] = round(CH.avaliar_deteccao(tag, "test", iou_nms, 0.05)["recall"] * 109)
    return r


def ponta_a_ponta(cache, conf, aceita):
    pt = json.load(open(CH.RES / "pipeline" / f"{cache}_test.json"))
    pt = FC.filtrar(pt, conf)
    m = CH.avaliar_pipeline(pt, "test", aceita)
    G = CH.gt_split("test")
    img_mina = img_achou = img_sem = img_alarme = 0
    for stem, (p, ann, _) in G.items():
        g = [(a[0], a[1:]) for a in ann]
        milco_pred = [x for x in pt.get(stem, []) if int(np.argmax(x[5:])) == 0 and x[5] >= aceita]
        if any(c == 0 for c, _ in g):
            img_mina += 1
            img_achou += any(any(CH._iou(x[:4], b) >= 0.5 for x in milco_pred) for c, b in g if c == 0)
        else:
            img_sem += 1
            img_alarme += bool(milco_pred)
    return {k: m[k] for k in ("MILCO_tp", "MILCO_fp", "MILCO_fn", "n_MILCO", "MILCO_precisao", "MILCO_recall", "MILCO_f1", "confusao", "img_auc")} | \
        dict(imagens_com_mina=img_mina, imagens_com_mina_achadas=img_achou, imagens_sem_mina=img_sem, imagens_sem_mina_com_alarme=img_alarme)


def classif_gt():
    y = None; P = []
    for j in ("b1_convnextpico_224", "f_b1_convnextpico_224_s43", "f_b1_convnextpico_224_s44"):
        d = np.load(CH.RES / "runs" / f"{j}_probs.npz")
        if y is None:
            idx = d["test_idx"]; y = np.load(CH.CACHE / "crops_m0.15_224.npz")["y"][idx]
        assert (d["test_idx"] == idx).all()
        P.append(d["p_test"])
    pred = np.mean(P, 0).argmax(1)
    M = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    def prf(k):
        tp = M[k][k]; p = tp / max(M[0][k] + M[1][k], 1); r = tp / max(sum(M[k]), 1)
        return dict(precisao=p, recall=r, f1=2 * p * r / max(p + r, 1e-9))
    return dict(matriz=M, acuracia=(M[0][0] + M[1][1]) / len(y), MILCO=prf(0), NOMBO=prf(1))


def curvas_yolo(csvf):
    rows = list(csv.DictReader(open(csvf)))
    k = lambda r, n: float(r[[c for c in r if c.strip() == n][0]])
    ep = [int(k(r, "epoch")) for r in rows]
    P = [k(r, "metrics/precision(B)") for r in rows]; R = [k(r, "metrics/recall(B)") for r in rows]
    F = [2 * p * q / (p + q) if p + q > 0 else 0 for p, q in zip(P, R)]
    m50 = [k(r, "metrics/mAP50(B)") for r in rows]; m95 = [k(r, "metrics/mAP50-95(B)") for r in rows]
    fit = [0.1 * a + 0.9 * b for a, b in zip(m50, m95)]
    return dict(epoca=ep, precisao=P, recall=R, F1=F, mAP50=m50, melhor_epoca=ep[int(np.argmax(fit))], tempo_min=k(rows[-1], "time") / 60)


dados = {}
dados["deteccao"] = {"antigo": bloco_deteccao("D1_640", 0.7), "novo": bloco_deteccao("F_G5tta+Atta_wbf", 0.5)}
dados["classificacao_recortes_reais"] = {
    "antigo_apresentacao": dict(matriz=[[67, 8], [8, 26]], acuracia=0.85, MILCO=dict(precisao=0.89, recall=0.89, f1=0.89), NOMBO=dict(precisao=0.76, recall=0.76, f1=0.76)),
    "antigo_reavaliado": json.load(open(CH.RES / "linha_base" / "fusao_antiga_recortes_gt.json")),
    "novo": classif_gt()}
dados["ponta_a_ponta"] = {"antigo": ponta_a_ponta("D1_640_iou0.7__fusao_antiga", 0.5, 0.0),
                          "novo_recall": ponta_a_ponta("F_G5tta+Atta_wbf_iou0.5__ens-b1_convnextpico_224", 0.2, 0.6),
                          "novo_equilibrado": ponta_a_ponta("F_G5tta+Atta_wbf_iou0.5__ens-b1_convnextpico_224", 0.4, 0.6)}
MODELS = CH.HERE.parent
dados["curvas_yolo"] = {"antigo_A": curvas_yolo(MODELS / "TESTE_BOX_YOLO/runs/detect/work/runs/modelA_yolo_p2/results.csv"),
                        "novo_G5": curvas_yolo(CH.HERE / "yolo_gpu/saida/runs/G5/results.csv")}
dados["curvas_convnext"] = {j: json.load(open(CH.RES / "runs" / f"{j}.json"))["hist"] for j in
                            ("b1_convnextpico_224", "f_b1_convnextpico_224_s43", "f_b1_convnextpico_224_s44")}
json.dump(dados, open(OUT, "w"), indent=1)
print(json.dumps({k: dados[k] for k in ("deteccao", "classificacao_recortes_reais", "ponta_a_ponta")}, indent=1)[:4000])
print({k: (v["melhor_epoca"], len(v["epoca"]), round(v["tempo_min"])) for k, v in dados["curvas_yolo"].items()})
