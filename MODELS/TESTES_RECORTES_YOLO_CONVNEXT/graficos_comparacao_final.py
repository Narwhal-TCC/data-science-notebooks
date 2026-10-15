import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

D = json.load(open("resultados/figuras/comparacao_final/dados.json"))
OUT = "resultados/figuras/comparacao_final/"
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 15, "axes.titlesize": 18, "axes.titleweight": "semibold",
                     "axes.edgecolor": "#c9c7c0", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff"})
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e8e7e2"
ANT, NOV, EQU = "#eb6834", "#2a78d6", "#1baf7a"   # slots validados: laranja = antigo, azul = novo (modo recall), aqua = novo (equilibrado)
from matplotlib.ticker import FuncFormatter
VIR = FuncFormatter(lambda v, p: (f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}").replace(".", ","))
c2 = lambda v: f"{v:.2f}".replace(".", ",")
BLUES = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#1d4f91"])


def titulo(fig, t, s):
    fig.text(0.04, 0.955, t, fontsize=26, fontweight="bold", color=TXT, va="top")
    fig.text(0.04, 0.895, s, fontsize=15, color=TXT2, va="top")


def nota(fig, t):
    fig.text(0.04, 0.025, t, fontsize=12, color=TXT2)


def barras_h(ax, labels, vals, cores, fmt, maxv, sufixo=""):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cores, height=0.62, edgecolor="#ffffff", linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + maxv * 0.015, yi, fmt(v) + sufixo, va="center", fontsize=16, fontweight="semibold", color=TXT)
    ax.set_yticks(y, labels); ax.set_xlim(0, maxv * 1.18)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- 01 resumo: minas encontradas
P = D["ponta_a_ponta"]; a, r, e = P["antigo"], P["novo_recall"], P["novo_equilibrado"]
labs = ["Antigo (modelos da apresentação)\nYOLO A + fusão Ghost/MobileNet", "Novo — modo recall\n2 YOLOs combinados + ConvNeXt", "Novo — modo equilibrado\n2 YOLOs combinados + ConvNeXt"]
cores = [ANT, NOV, EQU]
fig, axs = plt.subplots(1, 3, figsize=(16, 9), dpi=120, gridspec_kw=dict(wspace=0.55))
fig.subplots_adjust(left=0.235, right=0.97, top=0.76, bottom=0.12)
titulo(fig, "Fluxo completo: o novo encontra muito mais minas", "Teste: 176 imagens · 75 minas (MILCO) · conf e limiar escolhidos na validação")
barras_h(axs[0], labs, [a["MILCO_tp"], r["MILCO_tp"], e["MILCO_tp"]], cores, lambda v: f"{v}", 75, " / 75")
axs[0].set_title("Minas encontradas e\nclassificadas como mina", loc="left", color=TXT)
barras_h(axs[1], ["", "", ""], [a["imagens_com_mina_achadas"], r["imagens_com_mina_achadas"], e["imagens_com_mina_achadas"]], cores, lambda v: f"{v}", 41, " / 41")
axs[1].set_title("Imagens com mina em que\nao menos 1 mina foi achada", loc="left", color=TXT)
barras_h(axs[2], ["", "", ""], [a["MILCO_fp"], r["MILCO_fp"], e["MILCO_fp"]], cores, lambda v: f"{v}", 30)
axs[2].set_title("Falsos alarmes de mina\n(no teste inteiro)", loc="left", color=TXT)
nota(fig, "Recall de minas: antigo 48% → novo 77% (recall) / 71% (equilibrado) · AUC por imagem (\"tem mina?\"): 0,90 → 0,97 / 0,94")
fig.savefig(OUT + "01_minas_encontradas.png"); plt.close(fig)

# ---------------------------------------------------------------- 02 destino de cada objeto (ponta a ponta)
fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=120, gridspec_kw=dict(wspace=0.35))
fig.subplots_adjust(left=0.1, right=0.93, top=0.74, bottom=0.14)
titulo(fig, "O que aconteceu com cada objeto do teste", "Linhas = o que o objeto é · colunas = o que o fluxo respondeu (109 objetos: 75 MILCO, 34 NOMBO)")
for ax, (nome, m) in zip(axs, [("Antigo (modelos da apresentação)", a), ("Novo — modo recall", r)]):
    c = m["confusao"]
    M = np.array([[c["MILCO→MILCO"], c["MILCO→NOMBO"], c["MILCO→perdido"]], [c["NOMBO→MILCO"], c["NOMBO→NOMBO"], c["NOMBO→perdido"]]])
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=75, aspect="auto")
    for i in range(2):
        for j in range(3):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=26, fontweight="bold", color="#ffffff" if M[i, j] > 35 else TXT)
    ax.set_xticks(range(3), ["Mina\n(MILCO)", "Não mina\n(NOMBO)", "Não\ndetectado"]); ax.set_yticks(range(2), ["MILCO", "NOMBO"])
    ax.set_title(nome, color=ANT if "Antigo" in nome else NOV, fontsize=20, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
nota(fig, "Antigo: o YOLO com conf 0,5 deixava 34 minas sem recorte. Novo: só 15 minas não detectadas e 2 confundidas com NOMBO.")
fig.savefig(OUT + "02_destino_dos_objetos.png"); plt.close(fig)

# ---------------------------------------------------------------- 03 classificador nos recortes reais (mesmo formato da apresentação)
C = D["classificacao_recortes_reais"]; ca, cn = C["antigo_apresentacao"], C["novo"]
fig = plt.figure(figsize=(16, 9), dpi=120)
titulo(fig, "Etapa 2 — classificar o recorte: ConvNeXt × fusão antiga", "Os mesmos 109 recortes reais do teste (75 MILCO, 34 NOMBO) · acurácia 0,85 → 0,89 · mina reconhecida: 67 → 72 de 75")
for k, (nome, m, cor) in enumerate([("Fusão Ghost+Mobile (apresentação)", ca, ANT), ("ConvNeXt-Pico (3 sementes)", cn, NOV)]):
    ax = fig.add_axes([0.08 + k * 0.28, 0.14, 0.23, 0.6])
    M = np.array(m["matriz"])
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=75)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=28, fontweight="bold", color="#ffffff" if M[i, j] > 35 else TXT)
    ax.set_xticks([0, 1], ["MILCO", "NOMBO"]); ax.set_yticks([0, 1], ["MILCO", "NOMBO"])
    ax.set_xlabel("Predito"); ax.set_ylabel("Real") if k == 0 else None
    ax.set_title(nome, color=cor, fontsize=15, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
ax = fig.add_axes([0.665, 0.14, 0.31, 0.6]); ax.yaxis.set_major_formatter(VIR)
mets = [("MILCO", "precisao"), ("MILCO", "recall"), ("MILCO", "f1"), ("NOMBO", "precisao"), ("NOMBO", "recall"), ("NOMBO", "f1")]
x = np.arange(len(mets)); w = 0.38
va = [ca[c][m] for c, m in mets]; vn = [cn[c][m] for c, m in mets]
ax.bar(x - w / 2, va, w, color=ANT, edgecolor="#ffffff", linewidth=2, label="Antigo")
ax.bar(x + w / 2, vn, w, color=NOV, edgecolor="#ffffff", linewidth=2, label="Novo")
for xi, v1, v2 in zip(x, va, vn):
    ax.text(xi - w / 2, v1 - 0.03, c2(v1), ha="center", va="top", fontsize=11, fontweight="semibold", color="#ffffff", rotation=90)
    ax.text(xi + w / 2, v2 - 0.03, c2(v2), ha="center", va="top", fontsize=11, fontweight="semibold", color="#ffffff", rotation=90)
ax.set_xticks(x, ["Precisão\nMILCO", "Recall\nMILCO", "F1\nMILCO", "Precisão\nNOMBO", "Recall\nNOMBO", "F1\nNOMBO"], fontsize=11)
ax.set_ylim(0, 1.08); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=12, bbox_to_anchor=(1, 1.1))
nota(fig, "Antigo = números da apresentação de 12/09. Reavaliando o mesmo checkpoint aqui, a fusão deu acurácia 0,81 (63/12/9/25).")
fig.savefig(OUT + "03_classificacao_recortes.png"); plt.close(fig)

# ---------------------------------------------------------------- 04 detector (mesmos painéis da apresentação)
da, dn = D["deteccao"]["antigo"], D["deteccao"]["novo"]
fig, axs = plt.subplots(1, 3, figsize=(16, 9), dpi=120, gridspec_kw=dict(wspace=0.28))
fig.subplots_adjust(left=0.06, right=0.98, top=0.74, bottom=0.17)
titulo(fig, "Etapa 1 — achar o objeto: detector novo × YOLO A", "Teste: 176 imagens, 109 objetos · mesmo avaliador para os dois")
grupos = [("Acerto da caixa (AP)", [("AP50", "AP50"), ("AP50-95", "AP50_95"), ("AP75", "AP75")]),
          ("Precisão, recall e F1", [("Precisão", "precisao"), ("Recall", "recall"), ("F1", "F1")]),
          ("AP por tamanho do objeto", [("Pequenos\n(<32 px)", "AP_pequenos"), ("Médios", "AP_medios")])]
for ax, (t, ms) in zip(axs, grupos):
    x = np.arange(len(ms)); w = 0.38
    v1 = [da[k] for _, k in ms]; v2 = [dn[k] for _, k in ms]
    ax.bar(x - w / 2, v1, w, color=ANT, edgecolor="#ffffff", linewidth=2, label="Antigo: YOLO A (640)")
    ax.bar(x + w / 2, v2, w, color=NOV, edgecolor="#ffffff", linewidth=2, label="Novo: YOLO G5 (GPU) + YOLO A, combinados")
    for xi, a1, a2 in zip(x, v1, v2):
        ax.text(xi - w / 2, a1 + 0.015, c2(a1), ha="center", fontsize=13, color=TXT)
        ax.text(xi + w / 2, a2 + 0.015, c2(a2), ha="center", fontsize=13, color=TXT)
    ax.set_xticks(x, [n for n, _ in ms]); ax.set_ylim(0, 1); ax.set_title(t, loc="left", color=TXT)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(VIR)
axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.2), ncols=2, fontsize=13)
nota(fig, f"Precisão/recall no conf de melhor F1 da validação (antigo {c2(da['conf'])}, novo {c2(dn['conf'])}) · Apresentação (avaliador do YOLO): AP50 0,704 · P 0,831 · R 0,630 · F1 0,717")
fig.savefig(OUT + "04_detector.png"); plt.close(fig)

# ---------------------------------------------------------------- 05 treino do YOLO por época
cA, cG = D["curvas_yolo"]["antigo_A"], D["curvas_yolo"]["novo_G5"]
fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=120, sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.16))
fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.14)
titulo(fig, "Treino do detector, época a época (validação)", f"Antigo: YOLO A · 640 px · {len(cA['epoca'])} épocas · ~{cA['tempo_min']:.0f} min   |   Novo: YOLO G5 · 1024 px · {len(cG['epoca'])} épocas · ~{cG['tempo_min']:.0f} min em GPU T4 (Kaggle)")
for ax, (t, k) in zip(axs.flat, [("Precisão", "precisao"), ("Recall", "recall"), ("F1", "F1"), ("mAP50", "mAP50")]):
    for c, cor, nome in ((cA, ANT, "Antigo"), (cG, NOV, "Novo (G5)")):
        y = np.array(c[k]); ys = np.array([y[max(0, i - 4):i + 1].mean() for i in range(len(y))])
        ax.plot(c["epoca"], y, color=cor, lw=1, alpha=0.25)
        ax.plot(c["epoca"], ys, color=cor, lw=2.2, label=nome)
        be = c["melhor_epoca"]; ax.plot([be], [y[be - 1]], "o", ms=9, color=cor, mec="#ffffff", mew=2)
    ax.set_title(t, loc="left", color=TXT); ax.set_ylim(0, 1); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(VIR)
axs[0, 0].legend(frameon=False, loc="lower right", fontsize=13)
for ax in axs[1]: ax.set_xlabel("Época")
nota(fig, f"Linha forte = média das últimas 5 épocas · ponto = melhor época, o checkpoint salvo (antigo {cA['melhor_epoca']}, novo {cG['melhor_epoca']}) · F1 = 2·P·R/(P+R)")
fig.savefig(OUT + "05_treino_yolo_epocas.png"); plt.close(fig)

# ---------------------------------------------------------------- 06 treino do ConvNeXt por época
H = D["curvas_convnext"]
cores_s = ["#1d4f91", "#2a78d6", "#7fb2ec"]
fig, axs = plt.subplots(1, 3, figsize=(16, 9), dpi=120, gridspec_kw=dict(wspace=0.22))
fig.subplots_adjust(left=0.06, right=0.98, top=0.74, bottom=0.14)
titulo(fig, "Treino do ConvNeXt nos recortes, época a época (validação)", "ConvNeXt-Pico · 224 px · AdamW + cosine · 3 sementes (o fluxo final usa a média das 3) · parada antecipada")
for ax, (t, k) in zip(axs, [("Acurácia", "val_acc"), ("F1 (média MILCO/NOMBO)", "val_f1_macro"), ("Perda no treino", "train_loss")]):
    for (j, h), cor, nome in zip(H.items(), cores_s, ["Semente 42", "Semente 43", "Semente 44"]):
        ep = [x["epoch"] for x in h]; y = [x[k] for x in h]
        ax.plot(ep, y, color=cor, lw=2.2, label=nome)
        sc = [x["val_score"] for x in h]; be = int(np.argmax(sc))
        ax.plot([ep[be]], [y[be]], "o", ms=9, color=cor, mec="#ffffff", mew=2)
    ax.set_title(t, loc="left", color=TXT); ax.set_xlabel("Época"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if k != "train_loss": ax.set_ylim(0.4, 1)
    ax.yaxis.set_major_formatter(VIR)
axs[0].legend(frameon=False, loc="lower right", fontsize=13)
nota(fig, "Ponto = época salva (melhor AUC de validação) · treino em CPU: ~16 min por semente · a fusão antiga não guardou histórico por época, por isso não aparece aqui")
fig.savefig(OUT + "06_treino_convnext_epocas.png"); plt.close(fig)
print("ok")
