"""
PÁGINA WEB ONDE SUBO AS IMAGENS E VEJO OS RESULTADOS DO YOLO + CNN
ISSO NÃO SERÁ USADO NO FINAL POIS TEREMOS NOSSO FRONT, ISSO SERÁ USADO APENAS PARA APRENDIZADO E ESTUDO

Como rodar:
    pip install streamlit torch torchvision timm ultralytics pillow numpy pandas
    streamlit run app_sonar_streamlit.py

Isso abre uma aba no seu navegador (geralmente http://localhost:8501) com:
- Upload de várias imagens de uma vez (selecione todos os arquivos da pasta no seletor)
- Sliders para ajustar o limiar de confiança da detecção e a margem de contexto do recorte
- Visualização de cada imagem com as caixas + rótulo + confiança desenhados
- Botão para baixar todas as imagens anotadas (.zip) e um resumo em .csv
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import models, transforms as T

import timm
from ultralytics import YOLO

# ============================================================
# Configuração
# ============================================================
CLASS_NAMES = ["MILCO", "NOMBO"]
FUSION_IMG_SIZE = 224
CAMINHO_YOLO_PADRAO = r"D:\testte\SEPARACAO_MODELS\TESTE_BOX_YOLO\runs\detect\work\runs\modelA_yolo_p2\weights\best.pt"
CAMINHO_FUSAO_PADRAO = r"D:\testte\SEPARACAO_MODELS\TESTE_JUNCAO\checkpoints\fusion_melhor.pt"
CORES = {"MILCO": (0, 255, 255), "NOMBO": (255, 255, 0)}  # ciano, amarelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Modelo de fusão — mesma arquitetura usada no treinamento
# ============================================================
class FusionGhostNetV2MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, img_size=FUSION_IMG_SIZE):
        super().__init__()
        self.ghostnet = timm.create_model("ghostnetv2_100", pretrained=False, num_classes=0)
        self.mobilenet = models.mobilenet_v2(weights=None)
        self.mobilenet.classifier = nn.Identity()

        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            ghost_dim = self.ghostnet(dummy).shape[1]
            mobile_dim = self.mobilenet(dummy).shape[1]

        fused_dim = ghost_dim + mobile_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f_ghost = self.ghostnet(x)
        f_mobile = self.mobilenet(x)
        fused = torch.cat([f_ghost, f_mobile], dim=1)
        return self.classifier(fused)


transform_classificacao = T.Compose([
    T.Resize((FUSION_IMG_SIZE, FUSION_IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner="Carregando modelos...")
def carregar_modelos(caminho_yolo: str, caminho_fusao: str):
    detector = YOLO(caminho_yolo)
    fusion_model = FusionGhostNetV2MobileNetV2(num_classes=len(CLASS_NAMES)).to(device)
    fusion_model.load_state_dict(torch.load(caminho_fusao, map_location=device))
    fusion_model.eval()
    return detector, fusion_model


@torch.no_grad()
def detectar_e_classificar(img: Image.Image, detector, fusion_model, score_thresh: float, margin: float):
    w, h = img.size
    saida_yolo = detector.predict(source=img, conf=score_thresh, verbose=False)[0]

    resultados = []
    for box in saida_yolo.boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        score = float(box.conf[0])
        bw, bh = xmax - xmin, ymax - ymin

        xmin_m = max(0, xmin - bw * margin)
        ymin_m = max(0, ymin - bh * margin)
        xmax_m = min(w, xmax + bw * margin)
        ymax_m = min(h, ymax + bh * margin)

        crop = img.crop((xmin_m, ymin_m, xmax_m, ymax_m))
        crop_tensor = transform_classificacao(crop).unsqueeze(0).to(device)
        probs = torch.softmax(fusion_model(crop_tensor), dim=1)[0].cpu().numpy()
        classe_idx = int(np.argmax(probs))

        resultados.append({
            "box": (xmin, ymin, xmax, ymax),
            "score_deteccao": score,
            "classe": CLASS_NAMES[classe_idx],
            "confianca_classificacao": float(probs[classe_idx]),
        })

    return resultados


def desenhar_resultado(img: Image.Image, resultados) -> Image.Image:
    img_anotada = img.copy()
    draw = ImageDraw.Draw(img_anotada)
    try:
        fonte = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        fonte = ImageFont.load_default()

    for r in resultados:
        xmin, ymin, xmax, ymax = r["box"]
        cor = CORES.get(r["classe"], (255, 0, 0))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=cor, width=3)
        rotulo = f"{r['classe']} ({r['confianca_classificacao']*100:.0f}%)"
        texto_y = max(ymin - 18, 0)
        draw.rectangle([xmin, texto_y, xmin + 8 * len(rotulo), texto_y + 16], fill=(0, 0, 0))
        draw.text((xmin + 2, texto_y), rotulo, fill=cor, font=fonte)

    return img_anotada


# ============================================================
# Interface
# ============================================================
st.set_page_config(page_title="Detecção e Classificação MILCO/NOMBO", layout="wide")
st.title("🔍 Detecção e Classificação de Minas em Sonar (MILCO/NOMBO)")
st.caption("YOLO (detecção) + fusão GhostNetV2 + MobileNetV2 (classificação)")

with st.sidebar:
    st.header("Modelos")
    caminho_yolo = st.text_input("Pesos do YOLO (.pt)", value=CAMINHO_YOLO_PADRAO)
    caminho_fusao = st.text_input("Checkpoint da fusão (.pt)", value=CAMINHO_FUSAO_PADRAO)

    st.header("Parâmetros")
    score_thresh = st.slider("Limiar de confiança da detecção", 0.05, 0.95, 0.50, 0.05)
    margin = st.slider("Margem de contexto no recorte", 0.0, 0.5, 0.15, 0.05)

    st.caption(f"Dispositivo: **{device}**")

arquivos = st.file_uploader(
    "Envie as imagens de sonar — selecione TODOS os arquivos da pasta de uma vez (Ctrl+A)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

processar = st.button("▶️ Processar", type="primary", disabled=not arquivos)

if processar:
    if not Path(caminho_yolo).exists():
        st.error(f"Não encontrei os pesos do YOLO em: {caminho_yolo}")
    elif not Path(caminho_fusao).exists():
        st.error(f"Não encontrei o checkpoint da fusão em: {caminho_fusao}")
    else:
        detector, fusion_model = carregar_modelos(caminho_yolo, caminho_fusao)

        linhas_resumo = []
        imagens_para_zip = {}
        barra = st.progress(0.0, text="Processando imagens...")

        for i, arquivo in enumerate(arquivos):
            img = Image.open(arquivo).convert("RGB")
            resultados = detectar_e_classificar(img, detector, fusion_model, score_thresh, margin)
            img_anotada = desenhar_resultado(img, resultados)

            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.image(img_anotada, caption=arquivo.name, use_container_width=True)
            with col_info:
                st.markdown(f"**{arquivo.name}**")
                if not resultados:
                    st.write("Nenhum objeto detectado acima do limiar.")
                for r in resultados:
                    st.write(
                        f"- **{r['classe']}** — confiança: {r['confianca_classificacao']*100:.1f}% "
                        f"| score detecção: {r['score_deteccao']*100:.1f}%"
                    )
                    linhas_resumo.append({
                        "arquivo": arquivo.name,
                        "classe": r["classe"],
                        "confianca_classificacao": round(r["confianca_classificacao"], 4),
                        "score_deteccao": round(r["score_deteccao"], 4),
                        "xmin": round(r["box"][0], 1),
                        "ymin": round(r["box"][1], 1),
                        "xmax": round(r["box"][2], 1),
                        "ymax": round(r["box"][3], 1),
                    })

            buf = io.BytesIO()
            img_anotada.save(buf, format="PNG")
            imagens_para_zip[f"{Path(arquivo.name).stem}_resultado.png"] = buf.getvalue()

            barra.progress((i + 1) / len(arquivos), text=f"Processando imagens... ({i + 1}/{len(arquivos)})")

        st.divider()
        st.subheader("Downloads")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for nome, dados in imagens_para_zip.items():
                zf.writestr(nome, dados)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Baixar todas as imagens anotadas (.zip)",
                data=zip_buf.getvalue(),
                file_name="resultados_sonar.zip",
                mime="application/zip",
            )
        with col_b:
            if linhas_resumo:
                df = pd.DataFrame(linhas_resumo)
                st.download_button(
                    "⬇️ Baixar resumo das detecções (.csv)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="resumo_deteccoes.csv",
                    mime="text/csv",
                )
                st.dataframe(df, use_container_width=True)
