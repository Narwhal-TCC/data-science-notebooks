# Fluxo em duas etapas — YOLO recorta, ConvNeXt classifica

O YOLO localiza os objetos na imagem de sonar. Cada caixa vira um recorte, e o recorte é classificado como **MILCO** (mina) ou **NOMBO** (não-mina).

## Protocolo

- **Split:** o mesmo do notebook YOLO original. São 819 / 175 / 176 imagens; o teste tem 109 objetos (75 MILCO e 34 NOMBO).
- **Escolha na validação:** modelo, checkpoint, confiança e limiar são escolhidos na validação. O teste é usado uma única vez.

## Steps

| # | Notebook / script | O que faz | Resultado principal |
|---|---|---|---|
| 01 | `01_deteccao_recorte.ipynb` | YOLO A sem retreino: 640 / 960 / 1280 px, TTA, mosaico 2×2; margem do recorte | TTA é o melhor (AP50 val 0,668 → 0,722) |
| 02 | `02_classificador_recortes.ipynb` | Classificador de recortes: backbones, otimizadores, fusões, 3 classes; finalistas com 3 sementes | ConvNeXt-Pico 224: acurácia 0,89 (fusão antiga: 0,81) |
| 03 | `03_pipeline_ponta_a_ponta.ipynb` (`fase_c.py`) | Fluxo completo: detector → recorte → classificador | Minas encontradas: 36 → 54 de 75 |
| 04 | `04_yolo_gpu_fusao_e_comparacao.ipynb` | Retreino do YOLO em GPU, fusão de detectores, comparação final | Minas encontradas: 58 de 75, com 25 falsos alarmes |

Scripts do step 04:
- `yolo_gpu/treinar_yolo_gpu.py`: variantes G1–G6.
- `fase_e_detector_gpu.py`: integra o YOLO retreinado ao fluxo.
- `fusao_detectores.py`: combina detectores (WBF / NMS).
- `recall_max.py` e `analise_erros_recall.py`: ponto de recall máximo e análise das minas perdidas.
- `dados_comparacao_final.py` e `graficos_comparacao_final.py`: comparação com a versão anterior.

## Fluxo final

```
imagem ─► YOLO G5 (v8s-P2, 1024 px, TTA) ─┐
                                          ├─ WBF ─► recorte (margem 15%) ─► ConvNeXt-Pico 224 (média de 3 sementes) ─► MILCO / NOMBO
imagem ─► YOLO A  (v8n-P2,  640 px, TTA) ─┘
```

| Teste (75 minas) | Versão anterior | Atual, modo recall | Atual, modo equilibrado |
|---|---|---|---|
| Minas encontradas | 36 | **58** | 53 |
| Minas não detectadas | 34 | **15** | 20 |
| Falsos alarmes | 7 | 25 | 11 |
| F1 MILCO | 0,61 | 0,73 | **0,76** |
| AUC por imagem | 0,90 | **0,97** | 0,94 |

Figuras: `resultados/figuras/comparacao_final/`.

## Uso

```python
from pipeline_inferencia import Pipeline
pipe = Pipeline("modelos/pipeline_config_fusao_recall.json")  # ou ..._fusao_equilibrado.json
deteccoes = pipe.analisar("imagem.jpg")                       # [{caixa, conf_detector, prob_MILCO, classe}]
```

- **Pesos já no repositório (não precisa treinar):**
  - classificadores: `modelos/classificador_recortes_b1_convnextpico_224_s{42,43,44}.pt`;
  - YOLOs retreinados: `yolo_gpu/saida/pesos/G1_best.pt` … `G6_best.pt` (o fluxo final usa o G5);
  - YOLO A: `../TESTE_BOX_YOLO/runs/detect/work/runs/modelA_yolo_p2/weights/best.pt`.
- **Tempo de inferência:** ~3 s por imagem em CPU.

## Arquivos

| Arquivo | Função |
|---|---|
| `crop_harness.py` | Recortes, treino retomável do classificador, cache de detecções, avaliação ponta a ponta |
| `fila_recortes.json` | Configuração e status dos treinos do classificador |
| `analise_recortes.py` / `gerar_notebooks.py` | Tabelas dos notebooks / geração dos notebooks |
| `exportar_pipeline.py` / `pipeline_inferencia.py` | Exportação dos pesos / inferência (compatível com o Streamlit) |
| `figuras_pipeline.py` | Exemplos anotados (antigo × novo) |
| `yolo_gpu/` | Retreino do YOLO em GPU (`COMO_RODAR.md`) |
| `resultados/` | `runs/` (classificadores), `deteccao/` (caches de detecção), `pipeline/` (fluxos), `figuras/` |

Variáveis de ambiente: `SONAR_RAW_DIR` (imagens), `RECORTES_CACHE` e `RECORTES_WORK`.
