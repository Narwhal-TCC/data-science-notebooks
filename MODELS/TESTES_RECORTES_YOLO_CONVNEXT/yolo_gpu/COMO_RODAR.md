# Retreino do YOLO em GPU

Executado no Kaggle (2× Tesla T4), com uma variante por GPU em paralelo. Levou ~7 h.

## Comandos

```bash
pip install -r requirements_gpu.txt
python treinar_yolo_gpu.py --zip PASTA_OU_ZIP_DO_DATASET --so G2 --device 0 --horas 2 --cache
```

| Opção | Efeito |
|---|---|
| `--zip` | Zip do figshare ou pasta com os zips por ano |
| `--so` | Variantes a treinar (G1–G6) |
| `--horas` | Teto de tempo de treino por variante |
| `--smoke` | Teste rápido (1 época) |

## Variantes

| Variante | Modelo | px | AP50 val | AP50 teste |
|---|---|---|---|---|
| G1 | YOLOv8n-P2 | 640 | 0,670 | 0,758 |
| G2 | YOLOv8n-P2 | 1024 | 0,666 | 0,724 |
| G3 | YOLOv8s-P2 | 1024 | 0,697 | 0,692 |
| G4 | YOLO11s | 1024 | 0,670 | 0,685 |
| **G5** | YOLOv8s-P2 + augmentação para objetos pequenos | 1024 | 0,693 | 0,722 |
| G6 | YOLOv8s-P2 + AdamW | 1024 | 0,680 | 0,691 |

- **Escolha pela validação:** AP50 + recall de objetos pequenos → **G5**.
- **Saída:** `saida/resultados_yolo_gpu.csv`, `saida/runs/` (curvas) e `saida/pesos/` (pesos `G*_best.pt`).
