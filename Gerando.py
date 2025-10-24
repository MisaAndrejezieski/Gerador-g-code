import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# === CONFIGURAÇÕES DE PASTAS ===
PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(PASTA_BASE, "Imagens")
os.makedirs(PASTA_IMAGENS, exist_ok=True)

# === ESCALAS DO PROJETO ===
ALTURA_MAX_MM = 3000  # 3 metros
LARGURA_MAX_MM = 2000  # 2 metros
PROFUNDIDADE_MAX_MM = 30  # profundidade do entalhe em mm

# === SELECIONA A IMAGEM ===
def selecionar_imagem():
    root = tk.Tk()
    root.withdraw()
    caminho = filedialog.askopenfilename(
        title="Selecione a imagem para gerar o relevo",
        filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
    )
    return caminho

# === GERA RELEVO 3D ===
def gerar_relevo(caminho_imagem):
    if not caminho_imagem or not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Imagem '{caminho_imagem}' não encontrada.")

    print("📸 Carregando imagem:", caminho_imagem)
    imagem = Image.open(caminho_imagem).convert("L")

    # Redimensiona a imagem para 2000x3000mm (proporcional)
    largura, altura = imagem.size
    proporcao = min(LARGURA_MAX_MM / largura, ALTURA_MAX_MM / altura)
    nova_largura = int(largura * proporcao)
    nova_altura = int(altura * proporcao)
    imagem = imagem.resize((nova_largura, nova_altura))

    matriz = np.array(imagem)
    matriz = np.flipud(matriz)

    # Normaliza altura (Z)
    z = (matriz / 255.0) * PROFUNDIDADE_MAX_MM

    # Cria coordenadas X e Y
    x = np.linspace(0, LARGURA_MAX_MM, nova_largura)
    y = np.linspace(0, ALTURA_MAX_MM, nova_altura)
    X, Y = np.meshgrid(x, y)

    # === VISUALIZAÇÃO 3D ===
    fig = go.Figure(data=[go.Surface(
        z=z,
        x=X,
        y=Y,
        colorscale="gray",
        showscale=True
    )])

    fig.update_layout(
        title="Pré-visualização 3D — Relevo em Madeira",
        scene=dict(
            xaxis_title="Largura (mm)",
            yaxis_title="Altura (mm)",
            zaxis_title="Profundidade (mm)",
            aspectratio=dict(x=2, y=3, z=0.3),
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        template="plotly_dark"
    )

    caminho_preview = os.path.join(PASTA_IMAGENS, "preview_3d.png")
    fig.write_image(caminho_preview)
    fig.show()

    print(f"🖼️ Preview 3D salvo em: {caminho_preview}")
    return X, Y, z


# === GERA O G-CODE ===
def gerar_gcode(X, Y, Z):
    caminho_gcode = os.path.join(PASTA_IMAGENS, "relevo.gcode")
    caminho_nc = os.path.join(PASTA_IMAGENS, "relevo_3d.nc")

    print("⚙️ Gerando G-code...")

    with open(caminho_gcode, "w") as f:
        f.write("G21 ; Define unidades em milímetros\n")
        f.write("G90 ; Modo de posicionamento absoluto\n")
        f.write("G1 F800 ; Define velocidade de avanço\n")

        for i in range(len(Y)):
            if i % 2 == 0:
                x_seq = range(len(X[0]))
            else:
                x_seq = reversed(range(len(X[0])))

            for j in x_seq:
                f.write(f"G1 X{X[i][j]:.2f} Y{Y[i][j]:.2f} Z{-Z[i][j]:.2f}\n")

        f.write("G0 Z5 ; Retorna para posição segura\n")
        f.write("M30 ; Fim do programa\n")

    # copia para .nc
    with open(caminho_gcode, "r") as src, open(caminho_nc, "w") as dst:
        dst.write(src.read())

    print(f"✅ G-code salvo em: {caminho_gcode}")
    print(f"✅ Arquivo CNC (.nc) salvo em: {caminho_nc}")


# === EXECUÇÃO PRINCIPAL ===
def main():
    caminho = selecionar_imagem()
    if not caminho:
        messagebox.showerror("Erro", "Nenhuma imagem foi selecionada.")
        return

    try:
        X, Y, Z = gerar_relevo(caminho)
        gerar_gcode(X, Y, Z)
        messagebox.showinfo("Concluído", "Relevo 3D e G-code gerados com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", str(e))
        print("❌ Erro:", e)


if __name__ == "__main__":
    main()
