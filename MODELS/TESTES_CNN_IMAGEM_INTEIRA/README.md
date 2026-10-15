# CNN na imagem inteira — mina × não-mina

Classificação binária por imagem de sonar de varredura lateral. A imagem é positiva quando contém pelo menos um MILCO.

## Dados e protocolo

- **Dataset:** SIMD (Marinha Portuguesa, figshare 24574879). São 1.170 imagens de 2010 a 2021, das quais 255 são positivas.
- **Triagem:** holdout estratificado 70/15/15 (seed 42). O checkpoint e o threshold são escolhidos na validação.
- **Finalistas:** validação cruzada estratificada 5-fold, com predições *out-of-fold*.

## Notebooks

| # | Notebook | O que testa |
|---|---|---|
| 01 | `01_resolucao_e_backbones.ipynb` | Resolução (96–320 px), fine-tuning, MobileNetV2 / EfficientNetV2-B0 / ConvNeXt-Pico |
| 02 | `02_otimizadores_e_loss.ipynb` | Adam, AdamW, SGD Nesterov, SAM, Focal Loss |
| 03 | `03_aug_sonar_vit_ensemble.ipynb` | Augmentação física de sonar, MixUp, ViT (DeiT-Tiny), ensemble |
| 04 | `04_validacao_cruzada_e_campeao.ipynb` | 5-fold dos finalistas e escolha do campeão |

## Resultado (5-fold, com TTA)

| Modelo | AUC | PR-AUC | F1 |
|---|---|---|---|
| MobileNetV2 96 px congelada (anterior) | 0,903 ± 0,014 | 0,770 ± 0,036 | 0,675 ± 0,019 |
| EfficientNetV2-B0 224 + SAM | 0,939 ± 0,032 | 0,881 ± 0,050 | 0,810 ± 0,084 |
| **ConvNeXt-Pico 320 + AdamW (campeão)** | **0,965 ± 0,010** | **0,912 ± 0,029** | **0,846 ± 0,031** |

**Conclusões:**
- O ganho vem de três fatores juntos: resolução, backbone moderno e fine-tuning completo.
- O otimizador é secundário.
- A augmentação de sonar, a Focal Loss, o ViT e o ensemble não ajudaram.

## Generalização (fase 2)

- Teste em datasets SSS externos (KLSG, SCTD, SCTD2), usados só como negativos: `avaliar_externo.py`.
- **Imagem em tons de cinza:** o falso positivo externo cai de 14,2% (campeão) para 3,9%, com AUC de holdout de 0,966.
- **Pendente na fila (`fila_experimentos.json`):** validação deixando um ano de fora, 416 px e SAM.

## Arquivos

| Arquivo | Função |
|---|---|
| `sonar_harness.py` | Dados, modelos, otimizadores, métricas e treino retomável por fila |
| `fila_experimentos.json` | Configuração e status de cada experimento |
| `analise.py` / `gerar_notebooks.py` | Tabelas e gráficos / geração dos notebooks |
| `avaliar_externo.py` | Avaliação nos datasets externos |
| `exportar_campeao.py` / `inferencia.py` | Exportação e inferência do campeão (5 folds + TTA, threshold 0,632) |
| `resultados/` | Métricas por execução (`runs/`), tabelas, figuras e avaliação externa |

## Como rodar

```bash
export SONAR_RAW_DIR=/caminho/dataset   # pastas 2010..2021 extraídas do zip
python sonar_harness.py 999999          # executa a fila de experimentos
python inferencia.py pasta_de_imagens/  # campeão (pesos em modelos/, fora do git)
```
