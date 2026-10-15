"""Compara os fluxos sob a premissa 'encontrar o MAIOR número de minas' (recall).
(a) teto: conf mínima (0,05) e sem limiar de aceite; (b) ponto de operação escolhido na VALIDAÇÃO por F2 (recall pesa 4×)."""
import json, sys, time
import crop_harness as CH, fase_c as FC
t_end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
CH.DET_CONFIGS.update({"E_G5_1024": dict(imgsz=1024), "E_G5_1024_tta": dict(imgsz=1024, augment=True)})
ens = "ens:b1_convnextpico_224"
FL = [("P0 antigo (YOLO A, conf 0,5 fixo)", "D1_640", 0.7, "fusao_antiga"),
      ("P2 YOLO A+TTA + fusão antiga", "D4_640_tta", 0.5, "fusao_antiga"),
      ("P3 YOLO A + ConvNeXt", "D1_640", 0.7, ens),
      ("P4 YOLO A+TTA + ConvNeXt", "D4_640_tta", 0.5, ens),
      ("P5 mosaico 2x2 + ConvNeXt 3 classes", "D5_tiles2x2_640", 0.5, "ens:b5_fundo_3classes"),
      ("P6 YOLO G5 + ConvNeXt", "E_G5_1024", 0.5, ens),
      ("P6t YOLO G5+TTA + ConvNeXt", "E_G5_1024_tta", 0.5, ens)]
def f2(m):
    p, r = m["MILCO_precisao"], m["MILCO_recall"]; return 5 * p * r / max(4 * p + r, 1e-9)
out = []
for nome, det, iou, cls in FL:
    p = dict(nome=nome, det=det, iou=iou, cls=cls)
    pv, ok1 = FC.preds_cache(p, "val", t_end); pt, ok2 = FC.preds_cache(p, "test", t_end)
    if not (ok1 and ok2): print("parcial:", nome); json.dump(out, open(CH.RES / "pipeline" / "recall_max.json", "w"), indent=1); sys.exit()
    teto = CH.avaliar_pipeline(FC.filtrar(pt, 0.05), "test", 0.0)
    best = None
    for c in FC.CONF_GRID:
        for a in FC.ACEITA_GRID:
            mv = CH.avaliar_pipeline(FC.filtrar(pv, c), "val", a)
            k = (f2(mv), mv["MILCO_recall"])
            if best is None or k > best[0]: best = (k, c, a, mv)
    _, c, a, mv = best
    mt = CH.avaliar_pipeline(FC.filtrar(pt, c), "test", a)
    if nome.startswith("P0"):   # P0 é o fluxo antigo como era: conf 0,5 fixo
        mt = CH.avaliar_pipeline(FC.filtrar(pt, 0.5), "test", 0.0); mv = CH.avaliar_pipeline(FC.filtrar(pv, 0.5), "val", 0.0); c, a = 0.5, 0.0
    out.append(dict(fluxo=nome, teto_minas=teto["MILCO_tp"], teto_fp=teto["MILCO_fp"], conf=c, aceita=a,
                    val_recall=mv["MILCO_recall"], val_f2=f2(mv), minas=mt["MILCO_tp"], fp=mt["MILCO_fp"],
                    recall=mt["MILCO_recall"], precisao=mt["MILCO_precisao"], f2=f2(mt), img_auc=mt["img_auc"]))
    print(f"{nome:38s} teto {teto['MILCO_tp']}/75 (FP {teto['MILCO_fp']}) | val F2 {f2(mv):.3f} rec {mv['MILCO_recall']:.3f} (conf {c}, aceite {a}) | TESTE {mt['MILCO_tp']}/75 FP {mt['MILCO_fp']} P {mt['MILCO_precisao']:.2f} R {mt['MILCO_recall']:.3f}")
json.dump(out, open(CH.RES / "pipeline" / "recall_max.json", "w"), indent=1)
print("OK")
