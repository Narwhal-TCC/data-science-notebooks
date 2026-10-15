"""Gera os notebooks 01–03 do fluxo de recortes (depois executados com nbconvert para gravar as saídas)."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import json, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.precision", 3); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
sys.path.insert(0, str(Path.cwd()))
import analise_recortes as A"""

PROTOCOLO = """### Protocolo comum
- **Dados:** SIMD (Marinha Portuguesa, 1170 imagens side-scan, 2010–2021). **MILCO** = objeto com aparência de mina; **NOMBO** = objeto que não é mina.
- **Split:** idêntico ao notebook `sonar_deteccao_comparacao_modelos_V3_AA.ipynb` (arquivos ordenados por nome, `random.Random(42).shuffle`, 70/15/15) → 819 / 175 / 176 imagens. No teste há **109 objetos** (75 MILCO, 34 NOMBO).
- **Seleção sempre na validação** (checkpoint, conf do detector, limiar de aceite, finalista). O teste é usado uma vez, para reportar.
- **Hardware:** máquina local com 2 núcleos de CPU e 4 GB de RAM, **sem GPU**; treinos fatiados e retomáveis (`crop_harness.py` + `fila_recortes.json`).
- **Pesos antigos** (YOLO Modelo A e fusão GhostNetV2 + MobileNetV2) foram apenas lidos, nunca alterados."""

TREINO = """# (Opcional) Re-treinar os experimentos deste notebook. Os resultados abaixo já vêm de resultados/runs/.
# Em máquina com GPU, troque RODAR_TREINO para True: o mesmo harness treina sem limite de tempo.
RODAR_TREINO = False
GRUPOS = {grupos}
import crop_harness as CH
fila = json.load(open("fila_recortes.json"))
jobs = [j for j in fila if j.get("group") in GRUPOS]
for j in jobs:
    print(f"{{j['id']:32s}} {{j.get('desc','')}}")
if RODAR_TREINO:
    import math
    for j in jobs:
        CH.train_job(j, deadline=math.inf)"""


def nb(nome, celulas):
    n = nbf.v4.new_notebook()
    n["cells"] = celulas
    n["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, HERE / nome)


nb("01_deteccao_recorte.ipynb", [
    md("""# 01 — Etapa de detecção e recorte

**Pergunta:** dá para melhorar o *recorte* (o YOLO que localiza os objetos) **sem retreinar**, só mudando a inferência? E qual recorte (margem/janela) ajuda mais o classificador?

Sem GPU, o retreino do YOLO ficou como kit pronto em `yolo_gpu/` (variantes G1–G6). Aqui testamos o **Modelo A** já treinado (YOLOv8n-P2, 640 px) com 5 formas de inferência:

| Config | O que muda |
|---|---|
| D1_640 | Referência (como no notebook) |
| D2_960 / D3_1280 | Imagem ampliada na inferência |
| D4_640_tta | *Test-time augmentation* (flip + escalas) |
| D5_tiles2x2_640 | Imagem cortada em 4 blocos com sobreposição + imagem inteira |"""),
    md(PROTOCOLO),
    code(SETUP),
    md("## Varredura de detecção (conf 0,25; AP50 com IoU 0,5)"),
    code("A.resumo_deteccao(conf=0.25)"),
    md("""**Leitura:**
- **TTA (D4) é a melhor opção:** AP50 de validação 0,722 (contra 0,668 da referência) e recall de teste de 0,67 → 0,73, com poucos falsos positivos extras.
- **Ampliar a imagem piora muito** (D2, D3): o modelo foi treinado em 640 px e não generaliza para objetos "maiores".
- **Blocos (D5)** têm o maior recall de validação, mas dobram os falsos positivos. Só valem a pena com um classificador que rejeite fundo (testado no notebook 03).
- Escolha (na validação): **D4_640_tta com NMS 0,5**."""),
    md("## Forma do recorte para o classificador (grupo B3, ConvNeXt-Pico 224)"),
    code("""t = A.tabela_classificadores(["B1", "B3"])
t[t.job.isin(["b1_convnextpico_224", "b3_margem030", "b3_margem050", "b3_quadrado2x"])]"""),
    md("""**Leitura:** a margem de 15% (a do notebook antigo) tem a melhor AUC de validação. Margens maiores dão resultados de teste instáveis: 30% cai para 0,79 de acurácia e 50% sobe para 0,92 com uma semente só. Ou seja, com 109 objetos no teste, essas diferenças estão dentro do ruído. Ficamos com **m0.15**."""),
    code(TREINO.format(grupos='["B3"]')),
])

nb("02_classificador_recortes.ipynb", [
    md("""# 02 — Classificador de recortes: MILCO × NOMBO

**Pergunta:** o ConvNeXt, que venceu na imagem inteira (fase 1), também vence no recorte? Contra que referência?

**Referência antiga:** a fusão GhostNetV2 + MobileNetV2 do `TESTE_JUNCAO`, reavaliada aqui nos mesmos 109 recortes verdadeiros do teste: **acurácia 0,807**. O notebook antigo reportava 0,853; a diferença provavelmente vem de outro checkpoint ou outra execução.

**Grupos testados** (todos com recortes verdadeiros do teste, margem de 15%, 224 px, AdamW + cosine, salvo indicação):

| Grupo | O que varia |
|---|---|
| B1 | Backbone e resolução |
| B2 | Otimizador e loss |
| B4 | Fusão de 2 backbones (como a fusão antiga) |
| B5 | 3 classes: MILCO, NOMBO e **fundo** (recortes aleatórios sem objeto) |
| F | Os 5 melhores com 2 sementes extras (43, 44) |"""),
    md(PROTOCOLO),
    code(SETUP),
    code("""print(open("resultados/linha_base/fusao_antiga_recortes_gt.json").read())"""),
    md("## B1 — backbones e resolução"),
    code("""A.tabela_classificadores(["B1"]).drop(columns=["grupo"])"""),
    md("""**Leitura:** o ConvNeXt-Pico é o melhor backbone também nos recortes (acurácia de 0,90 a 0,91, contra 0,78 do MobileNetV2 nas mesmas condições). Em **128 px** ele é tão bom quanto em 224 px e custa **3× menos** (7,5 contra 24 ms por recorte em CPU)."""),
    md("## B2 — otimizadores e loss"),
    code("""A.tabela_classificadores(["B2"]).drop(columns=["grupo"])"""),
    md("""**Leitura:**
- **AdamW + cosine continua o melhor.**
- **SAM e SGD colapsaram** neste problema pequeno: passaram a prever só MILCO (acurácia 0,69 = proporção de MILCO). É diferente da imagem inteira, onde o SAM empatou com o Adam.
- **Focal loss e Adam com LR constante** ficam próximos, sem ganho.
- **Label smoothing** tem boa AUC de validação, mas resultado de teste pior."""),
    md("## B4 / B5 — fusões e o classificador de 3 classes"),
    code("""A.tabela_classificadores(["B4", "B5"]).drop(columns=["grupo"])"""),
    md("""**Leitura:**
- **Fusões não compensam:** dobram o custo e ficam abaixo do ConvNeXt sozinho, inclusive a fusão com GhostNet, análoga à antiga.
- **O classificador de 3 classes** teve a melhor acurácia nos objetos (0,927): aprender o que é "fundo" ajuda a separar MILCO de NOMBO."""),
    md("## Finalistas com 3 sementes (média ± desvio-padrão)"),
    code("A.tabela_sementes()"),
    md("""**Leitura:** com 3 sementes, os melhores (ConvNeXt-Pico 224, 128 e o de 3 classes) ficam entre **0,89 e 0,92 de acurácia**, contra **0,81 da fusão antiga**.
- **O 128 px é o mais estável** (desvio de 0,004).
- **Escolha do fluxo final:** não foi feita só pela AUC do recorte. Cada finalista foi colocado no fluxo completo (detector → recorte → classificador) e o escolhido foi o de **maior F1 de MILCO na validação**. Os detalhes estão no notebook 03."""),
    code(TREINO.format(grupos='["B1", "B2", "B4", "B5", "F"]')),
])

nb("03_pipeline_ponta_a_ponta.ipynb", [
    md("""# 03 — Fluxo ponta a ponta: detector → recorte → classificador

**Pergunta:** quanto o fluxo completo melhora em relação ao **fluxo antigo**? O fluxo antigo é YOLO Modelo A com conf 0,5, recorte de 15% e a fusão GhostNetV2 + MobileNetV2.

**Como medimos:**
- **Por objeto:** uma detecção é acerto de MILCO quando cobre um MILCO verdadeiro (IoU ≥ 0,5) e o classificador diz MILCO. Daí saem precisão, recall, F1 e AP50 da classe MILCO.
- **Por imagem:** a AUC de "a imagem tem mina?", usando score = conf × p(MILCO).
- **Ajuste:** em todos os fluxos, exceto o P0, a conf do detector e o limiar de aceite de MILCO foram escolhidos na **validação**."""),
    md(PROTOCOLO),
    code(SETUP),
    code("""d, escolhido = A.tabela_pipelines()
print("Classificador escolhido na validação:", escolhido)
d.drop(columns=["descrição"])"""),
    code("""A.grafico_pipelines(); plt.title("MILCO no teste (por objeto)"); plt.tight_layout(); plt.show()"""),
    md("""**Leitura** (teste, 75 MILCO):
- **Fluxo antigo (P0):** encontra só **36 de 75 minas** (recall 0,48, F1 0,61). O gargalo é o detector com conf 0,5, que perde 34 minas antes mesmo de classificar.
- **Detector com TTA (P2):** sozinho, já sobe o recall para 0,60.
- **ConvNeXt de recortes no detector antigo (P3):** também melhora (F1 0,685).
- **Fluxo novo (P4: TTA + ConvNeXt-Pico 224, 3 sementes):** **54 de 75 minas** (recall 0,72), **F1 0,69** e AUC por imagem de 0,96 (era 0,90).
- **Uma semente só (P4b):** quase igual (F1 0,68) e 3× mais barata.
- **Blocos + 3 classes (P5):** gera falsos positivos demais (F1 0,58).

**O erro que sobra agora é do detector:** das 21 minas não encontradas no P4, 18 o YOLO nem localizou; o classificador errou só 3. Por isso o próximo ganho grande é **retreinar o YOLO em GPU** (kit em `yolo_gpu/`)."""),
    code("""fc = json.load(open("resultados/pipeline/fase_c.json"))
pd.DataFrame({k: fc["pipelines"][k]["test"]["confusao"] for k in ["P0_antigo", "P4_novo"]})"""),
    md("## Exemplos do teste: verdade | fluxo antigo | fluxo novo\nVermelho = MILCO, azul = NOMBO. As 5 primeiras linhas são onde o fluxo novo mais ganha; a última é um caso em que ele não ganha."),
    code("""from IPython.display import Image as Img
Img("resultados/figuras/antigo_vs_novo_teste.png", width=1000)"""),
    md("## Usando o fluxo exportado (`pipeline_inferencia.py`, compatível com o Streamlit)"),
    code("""import crop_harness as CH, time
from pipeline_inferencia import Pipeline
pipe = Pipeline()                                   # lê modelos/pipeline_config.json
img = CH.split_imagens()["test"][0]
t = time.time(); det = pipe.analisar(img); print(f"{time.time()-t:.2f} s em CPU")
pd.DataFrame(det)"""),
    code("""pipe.desenhar(img, det).resize((400, 400))"""),
    md("""**Conferência:** nas 40 primeiras imagens de teste, o `pipeline_inferencia.py` com pesos em fp16 reproduz exatamente as decisões do `fase_c.py` (diferença máxima de p(MILCO) = 0,0004). O tempo é de ~0,4 s por imagem em CPU."""),
])

nb("04_yolo_gpu_fusao_e_comparacao.ipynb", [
    md("""# 04 — YOLO retreinado em GPU, fusão de detectores e comparação final

Objetivo: reduzir as minas perdidas na etapa de detecção, que é o gargalo apontado no notebook 03.

| Passo | Script | Saída |
|---|---|---|
| 1. Retreino do YOLO G1–G6 (Kaggle, 2× T4) | `yolo_gpu/treinar_yolo_gpu.py` | `yolo_gpu/saida/` |
| 2. Integração do melhor YOLO ao fluxo | `fase_e_detector_gpu.py` | `resultados/pipeline/fase_e.json` |
| 3. Fusão de detectores (WBF / NMS) | `fusao_detectores.py` | `resultados/pipeline/fusao_detectores.json` |
| 4. Ponto de operação para recall máximo | `recall_max.py` | `resultados/pipeline/recall_max.json` |
| 5. Comparação com a versão anterior | `dados_comparacao_final.py`, `graficos_comparacao_final.py` | `resultados/figuras/comparacao_final/` |"""),
    code(SETUP + "\nfrom IPython.display import Image as Img"),
    md("## 1. Retreino do YOLO (mesmo split; escolha pela validação: AP50 + recall de objetos pequenos)"),
    code("""y = pd.read_csv("yolo_gpu/saida/resultados_yolo_gpu.csv")
y.pivot_table(index=["variante", "modelo", "imgsz"], columns="split", values=["AP50", "recall@0.25", "recall_pequenos@0.25"]).round(3)"""),
    code('Img("resultados/figuras/comparacao_final/05_treino_yolo_epocas.png", width=1000)'),
    md("**G5** (YOLOv8s-P2, 1024 px, augmentação para objetos pequenos) teve o melhor critério na validação."),
    md("## 2. Integração do G5 ao fluxo (detector → recorte → ConvNeXt)"),
    code("""fe = json.load(open("resultados/pipeline/fase_e.json"))
pd.DataFrame({k: {m: v["test"][m] for m in ["MILCO_tp", "MILCO_fp", "MILCO_precisao", "MILCO_recall", "MILCO_f1"]}
              for k, v in {"P4 (YOLO A + TTA)": fe["P4_referencia"], "P6 (YOLO G5)": fe["pipelines"]["P6_yolo_gpu"]}.items()}).T.round(3)"""),
    md("## 3. Fusão de detectores"),
    code("""fz = json.load(open("resultados/pipeline/fusao_detectores.json"))
pd.DataFrame({k: {"varredura teste (minas)": v["varredura_test"]["MILCO_tp"], "varredura FP": v["varredura_test"]["MILCO_fp"],
                  "F1 val": v["F1"]["val_score"], "F1 teste": v["F1"]["test_score"], "minas (F1)": v["F1"]["test"]["MILCO_tp"],
                  "FP (F1)": v["F1"]["test"]["MILCO_fp"], "F2 val": v["F2"]["val_score"], "minas (F2)": v["F2"]["test"]["MILCO_tp"],
                  "FP (F2)": v["F2"]["test"]["MILCO_fp"]} for k, v in fz.items()}).T.round(3)"""),
    md("Escolhida pela validação: **G5+TTA ⊕ YOLO A+TTA (WBF)**. Configs: `modelos/pipeline_config_fusao_recall.json` e `modelos/pipeline_config_fusao_equilibrado.json`."),
    md("## 4. Fluxos sob a premissa de recall máximo"),
    code("""pd.DataFrame(json.load(open("resultados/pipeline/recall_max.json"))).set_index("fluxo")[["teto_minas", "teto_fp", "minas", "fp", "recall", "precisao", "val_f2"]].round(3)"""),
    md("## 5. Versão anterior × fluxo atual"),
    code('Img("resultados/figuras/comparacao_final/01_minas_encontradas.png", width=1000)'),
    code('Img("resultados/figuras/comparacao_final/02_destino_dos_objetos.png", width=1000)'),
    code('Img("resultados/figuras/comparacao_final/03_classificacao_recortes.png", width=1000)'),
    code('Img("resultados/figuras/comparacao_final/04_detector.png", width=1000)'),
    code('Img("resultados/figuras/comparacao_final/06_treino_convnext_epocas.png", width=1000)'),
])
print("ok")
