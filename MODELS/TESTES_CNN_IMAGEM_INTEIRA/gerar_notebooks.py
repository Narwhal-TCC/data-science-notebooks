"""Gera os notebooks 01–04 (depois executados com nbconvert para gravar as saídas)."""
import json
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import json, sys
from pathlib import Path
import pandas as pd
pd.set_option("display.precision", 3); pd.set_option("display.width", 200)
sys.path.insert(0, str(Path.cwd()))
import sonar_harness as h
import analise as A
fila = json.load(open("fila_experimentos.json"))"""

TREINO = """# ------------------------------------------------------------------
# (Opcional) Re-treinar os experimentos deste notebook.
# Os resultados abaixo já foram gerados pela fila de execução (máquina local, CPU, treino fatiado).
# Em uma máquina com GPU basta trocar RODAR_TREINO para True: o harness treina de ponta a ponta
# (o mesmo código que roda fatiado também roda sem limite de tempo).
# ------------------------------------------------------------------
RODAR_TREINO = False
GRUPO = "{grupo}"
jobs = [j for j in fila if j.get("group") == GRUPO]
for j in jobs:
    print(f"{{j['id']:45s}} {{j.get('desc','')}}")
if RODAR_TREINO:
    import math
    for j in jobs:
        h.train_job(j, deadline=math.inf)"""


def nb(nome, celulas):
    n = nbf.v4.new_notebook()
    n["cells"] = celulas
    n["metadata"]["kernelspec"] = {"name": "sonar", "display_name": "Python (sonar venv)", "language": "python"}
    nbf.write(n, HERE / nome)


PROTOCOLO = """### Protocolo comum (todos os notebooks)
- **Dados:** 1170 imagens side-scan sonar (Marinha Portuguesa, 2010–2021). Rótulo 1 = contém ≥1 **MILCO** (mine-like contact), 255 imagens (21,8%); 0 = só NOMBO ou fundo.
- **Split de triagem (holdout):** estratificado fixo 70/15/15 (seed 42) → treino 819 · validação 175 · teste 176 (38 minas). Mesmo split para todos os experimentos.
- **Seleção do checkpoint:** melhor AUC na **validação** (early stopping). O **threshold** de decisão também é escolhido na validação — o teste só é usado uma vez.
- **Métricas no teste:** ROC-AUC (principal), PR-AUC, F1/precisão/recall com threshold da validação, F1@0.5 e *F1 oráculo* (threshold escolhido no próprio teste, só para comparar com a metodologia do notebook antigo).
- **Atenção:** com 38 positivos no teste, diferenças < ~0,02 de AUC estão dentro do ruído — por isso os finalistas passam por **validação cruzada 5-fold** (notebook 04)."""

# ---------------------------------------------------------------- 01
nb("01_resolucao_e_backbones.ipynb", [
    md("""# 01 — Resolução de entrada + backbones modernos (fine-tuning completo)

**Pergunta:** o que limita o campeão atual (MobileNetV2 ImageNet congelado, 96×96, Adam)?

**Hipótese principal:** resolução. Os objetos ocupam ~5% da largura da imagem (mediana 33 px em imagens de 416–1024 px); em 96 px viram 2–6 px. Testamos separadamente o efeito de (a) resolução, (b) fine-tuning completo e (c) backbones mais modernos (EfficientNetV2-B0, ConvNeXt-Pico), em 224 e 320 px.

Configuração do fine-tuning: AdamW (lr 2e-4 no backbone, 10× na cabeça, weight decay 0,05), cosine com warmup de 5%, 25 épocas, early stopping (paciência 8), class weight suavizado (0,6) como no notebook original, augmentation padrão (flip horizontal, rotação ±10°, zoom/translação leves, brilho/contraste)."""),
    md(PROTOCOLO),
    code(SETUP),
    code(TREINO.format(grupo="NB01")),
    md("## Resultados no teste do holdout"),
    code("""runs = A.load_runs("nb01_")
df = A.tabela(runs)
df"""),
    code("""A.barras(df, titulo="NB01 — resolução e backbone (cinza = baseline reproduzido)", arquivo="nb01_barras.png");"""),
    md("### Com TTA (média com a imagem espelhada horizontalmente)"),
    code("""A.tabela(runs, tta=True)[["experimento","AUC","PR-AUC","F1 (thr val)","Precisão (thr val)","Recall (thr val)"]]"""),
    md("## Curvas de aprendizado"),
    code("""ids = ["nb01_baseline_mnv2_96_frozen_adam", "nb01_mnv2_224_full_adamw", "nb01_effv2b0_224_full_adamw", "nb01_convnextpico_320_full_adamw"]
A.curvas_treino(runs, ids, arquivo="nb01_curvas.png");"""),
    md("## ROC e Precisão × Recall"),
    code("""A.roc_pr(["nb01_baseline_mnv2_96_frozen_adam", "nb01_effv2b0_224_full_adamw", "nb01_effv2b0_320_full_adamw", "nb01_convnextpico_320_full_adamw"],
         titulo="NB01 — teste do holdout", arquivo="nb01_roc_pr.png");"""),
    md("""## Leitura
- **Baseline reproduzido** no harness: AUC ≈ 0,91 (o notebook original reportou 0,898 em outro split) — confirma que o harness é comparável.
- **Só aumentar a resolução** (96→224) com a base congelada quase não muda nada; **só fazer fine-tuning** em 96 px piora (overfitting com objetos minúsculos).
- O salto vem da **combinação backbone moderno + fine-tuning completo + resolução ≥ 224**: EfficientNetV2-B0 e ConvNeXt-Pico sobem para AUC ≈ 0,95–0,96 e PR-AUC ≈ 0,85–0,88.
- 320 px ajuda o ConvNeXt-Pico (melhor F1 do grupo), mas custa ~2× o tempo de treino."""),
])

# ---------------------------------------------------------------- 02
nb("02_otimizadores_e_loss.ipynb", [
    md("""# 02 — Otimizadores e função de perda

Backbone fixo: **EfficientNetV2-B0, 224 px, fine-tuning completo** (vencedor de AUC no NB01 e o mais rápido dos modernos). Comparamos, sob o mesmo protocolo:

| Experimento | Otimizador | Scheduler | Loss |
|---|---|---|---|
| `nb01_effv2b0_224_full_adamw` (referência) | AdamW (wd 0,05) | cosine + warmup | BCE ponderada |
| `nb02_effv2b0_adam_plateau` | **Adam** (o do campeão atual) | ReduceLROnPlateau | BCE ponderada |
| `nb02_effv2b0_sgd_nesterov` | SGD Nesterov (lr 0,01, m 0,9) | cosine + warmup | BCE ponderada |
| `nb02_effv2b0_sam_adamw` | **SAM** (ρ 0,05) sobre AdamW | cosine + warmup | BCE ponderada |
| `nb02_effv2b0_adamw_focal` | AdamW | cosine + warmup | **Focal** (γ 2, α 0,6) |

SAM (*Sharpness-Aware Minimization*) busca mínimos "planos", o que tende a generalizar melhor em datasets pequenos — ao custo de 2 passadas por batch. SGD foi incluído porque a literatura recente de detecção em sonar (Liu et al., JMSE 2026) o encontrou geralmente melhor que AdamW."""),
    md(PROTOCOLO),
    code(SETUP),
    code(TREINO.format(grupo="NB02")),
    md("## Resultados no teste do holdout"),
    code("""runs = {**A.load_runs("nb02_"), **{k: v for k, v in A.load_runs("nb01_").items() if k in ("nb01_effv2b0_224_full_adamw", "nb01_baseline_mnv2_96_frozen_adam")}}
df = A.tabela(runs)
df"""),
    code("""A.barras(df, titulo="NB02 — otimizadores e loss (EfficientNetV2-B0 224 px)", arquivo="nb02_barras.png");"""),
    code("""A.tabela(runs, tta=True)[["experimento","AUC","PR-AUC","F1 (thr val)","Precisão (thr val)","Recall (thr val)"]]"""),
    code("""A.curvas_treino(runs, ["nb01_effv2b0_224_full_adamw","nb02_effv2b0_adam_plateau","nb02_effv2b0_sgd_nesterov","nb02_effv2b0_sam_adamw","nb02_effv2b0_adamw_focal"], arquivo="nb02_curvas.png");"""),
    code("""A.roc_pr(["nb01_baseline_mnv2_96_frozen_adam","nb01_effv2b0_224_full_adamw","nb02_effv2b0_adam_plateau","nb02_effv2b0_sam_adamw"],
         titulo="NB02 — teste do holdout", arquivo="nb02_roc_pr.png");"""),
    md("""## Leitura
- **SAM + AdamW** foi o melhor do grupo em AUC e, principalmente, em **PR-AUC (≈0,92)** — a métrica mais informativa com classes desbalanceadas.
- **Adam com ReduceLROnPlateau** (o otimizador do campeão atual) também funciona bem com o novo backbone: o ganho grande veio do backbone/resolução/fine-tuning, não da troca de otimizador.
- SGD Nesterov ficou no mesmo patamar do AdamW; Focal Loss não trouxe ganho sobre a BCE ponderada.
- **Na validação cruzada (notebook 04)** a vantagem do SAM encolhe: AUC 0,941 vs 0,934 do Adam e PR-AUC igual (0,878) — ganho pequeno, ao custo de 2× o tempo de treino."""),
])

# ---------------------------------------------------------------- 03
nb("03_aug_sonar_vit_ensemble.ipynb", [
    md("""# 03 — Augmentação específica de sonar, Vision Transformer, TTA e ensemble

1. **Augmentação física de sonar** (inspirada em BenthicDINO, 2026): ruído *speckle* multiplicativo (Gamma, L ∈ [4,16]), ganho dependente do alcance (gradiente ao longo do eixo across-track) e *random erasing* pequeno — além da augmentação padrão. Só flip **horizontal** (a sombra acústica tem direção física).
2. **MixUp** (α 0,4, p 0,5) sobre a augmentação de sonar.
3. **DeiT-Tiny** (ViT, 5,7 M parâmetros): a literatura (arXiv 2409.12026) reporta ViTs acima de CNNs em sonar.
4. **TTA e ensemble** dos melhores modelos no mesmo holdout (média de probabilidades)."""),
    md(PROTOCOLO),
    code(SETUP),
    code(TREINO.format(grupo="NB03")),
    md("## Resultados no teste do holdout"),
    code("""runs = {**A.load_runs("nb03_"), **{k: v for k, v in A.load_runs("nb0").items() if k in ("nb01_effv2b0_224_full_adamw", "nb01_baseline_mnv2_96_frozen_adam")}}
df = A.tabela(runs)
df"""),
    code("""A.barras(df, titulo="NB03 — augmentação de sonar, MixUp e ViT", arquivo="nb03_barras.png");"""),
    code("""A.curvas_treino(runs, ["nb01_effv2b0_224_full_adamw","nb03_effv2b0_sonaraug","nb03_effv2b0_sonaraug_mixup","nb03_deittiny_224"], arquivo="nb03_curvas.png");"""),
    md("## TTA e ensemble de modelos (holdout)"),
    code("""todos = A.load_runs("nb0")
cands = {
  "SAM EffV2-B0 224": ["nb02_effv2b0_sam_adamw"],
  "ConvNeXt-Pico 320": ["nb01_convnextpico_320_full_adamw"],
  "Ens: SAM EffV2 + ConvNeXt320": ["nb02_effv2b0_sam_adamw","nb01_convnextpico_320_full_adamw"],
  "Ens: SAM EffV2 + ConvNeXt320 + Adam EffV2": ["nb02_effv2b0_sam_adamw","nb01_convnextpico_320_full_adamw","nb02_effv2b0_adam_plateau"],
  "Ens: top-5 por AUC val": sorted(todos, key=lambda k: -todos[k]["best_val_auc"])[:5],
}
rows = []
for nome, ids in cands.items():
    for tta in (False, True):
        m = A.ensemble_holdout(ids, tta=tta)
        rows.append({"modelo": nome, "TTA": tta, "AUC": m["auc"], "PR-AUC": m["pr_auc"], "F1 (thr val)": m["tval_f1"],
                     "Precisão (thr val)": m["tval_precision"], "Recall (thr val)": m["tval_recall"]})
pd.DataFrame(rows)"""),
    md("""## Leitura
- A **augmentação física de sonar** (com ou sem MixUp) **piorou** o teste (AUC ≈ 0,93 vs 0,957), apesar de AUC alta na validação: com objetos de poucos pixels, speckle forte e oclusão apagam justamente a assinatura do alvo. Não recomendado sem calibrar a intensidade.
- **DeiT-Tiny** ficou abaixo das CNNs (AUC ≈ 0,94) — com 819 imagens de treino e sem pré-treino em sonar, a vantagem reportada para ViTs não apareceu.
- **TTA** tem efeito pequeno e inconsistente; **ensembles** de modelos diferentes tendem a estabilizar as métricas — confirmado em 5-fold no notebook 04."""),
])

# ---------------------------------------------------------------- 04
nb("04_validacao_cruzada_e_campeao.ipynb", [
    md("""# 04 — Validação cruzada 5-fold dos finalistas e escolha do campeão

O holdout tem só 38 positivos no teste, então diferenças pequenas são ruído. Os finalistas foram re-treinados em **StratifiedKFold(5)** sobre as 1170 imagens (em cada fold: ~795 treino · ~141 validação para early stopping/threshold · 234 teste). Cada imagem é prevista exatamente uma vez como teste (*out-of-fold*).

Finalistas: baseline reproduzido · EfficientNetV2-B0 224 + Adam/plateau · EfficientNetV2-B0 224 + SAM · ConvNeXt-Pico 320 + AdamW (20 épocas, paciência 6), mais o **ensemble** dos modelos novos."""),
    code(SETUP),
    code(TREINO.format(grupo="CV5")),
    md("## Métricas por fold e média ± desvio"),
    code("""por_fold, resumo, oof = A.cv_resumo("cv_")
resumo.drop(columns="AUC_mean")"""),
    code("""pd.DataFrame({m: A.oof_metricas(oof[m]) for m in oof}).T"""),
    md("## Ensemble dos finalistas (média de probabilidades, fold a fold)"),
    code("""novos = [m for m in oof if not m.startswith("baseline") and len(oof[m]) == 5]
ens = A.oof_ensemble(oof, novos); ens["modelo"] = "ensemble(" + "+".join(novos) + ")"
display(ens.drop(columns="modelo").agg(["mean","std"]).T.round(3))
ens_plot = ens.assign(modelo="ensemble finalistas")
A.cv_boxplot(por_fold, extra=ens_plot, arquivo="cv_folds.png");"""),
    md("## Com TTA (flip horizontal)"),
    code("""pf_tta, resumo_tta, oof_tta = A.cv_resumo("cv_", tta=True)
resumo_tta.drop(columns="AUC_mean")"""),
    md("## Comparação pareada com o baseline (mesmos folds)"),
    code("""import numpy as np
piv = por_fold.pivot(index="fold", columns="modelo", values="AUC")
dif = piv.drop(columns="baseline_mnv2_96_frozen_adam").sub(piv["baseline_mnv2_96_frozen_adam"], axis=0)
print("Ganho de AUC sobre o baseline, por fold:"); display(dif.round(3))
print("Folds em que o modelo vence o baseline:", (dif > 0).sum().to_dict())
pivp = por_fold.pivot(index="fold", columns="modelo", values="PR-AUC")
difp = pivp.drop(columns="baseline_mnv2_96_frozen_adam").sub(pivp["baseline_mnv2_96_frozen_adam"], axis=0)
print("Ganho de PR-AUC sobre o baseline, por fold:"); display(difp.round(3))"""),
    md("## Campeão e threshold operacional"),
    code("""tab = por_fold.groupby("modelo")[["AUC","PR-AUC","F1 (thr val)"]].mean()
tab.loc["ensemble finalistas"] = ens[["AUC","PR-AUC","F1 (thr val)"]].mean()
tab = tab.sort_values("PR-AUC", ascending=False)
print(tab.round(3))"""),
    code("""cfg_campeao = json.load(open("modelos/campeao_config.json"))
print(json.dumps({k: v for k, v in cfg_campeao.items() if k != "cfg"}, indent=1, ensure_ascii=False))"""),
    md("""## Conclusão
- **Campeão: ConvNeXt-Pico (ImageNet), 320 px, fine-tuning completo com AdamW + cosine, TTA com flip horizontal.** Em 5-fold: AUC ≈ 0,965 ± 0,010, PR-AUC ≈ 0,91, F1 ≈ 0,85 — contra AUC 0,90 / PR-AUC 0,77 / F1 0,67 do baseline (MobileNetV2 96 px congelado + Adam), vencendo o baseline em todos os folds.
- O ensemble ConvNeXt-Pico 320 + EfficientNetV2-B0 SAM empata com o ConvNeXt sozinho, custando o dobro: não compensa.
- Para produção, o arquivo `modelos/` guarda os 5 modelos dos folds (média = *bagging*) e os thresholds calibrados nas predições out-of-fold; `inferencia.py` aplica tudo em novas imagens."""),
])
print("notebooks gerados")
