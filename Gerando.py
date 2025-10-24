import os
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
from tkinter import Tk, filedialog, messagebox

# =============================================================
# CONFIGURAÇÕES GERAIS
# =============================================================
PASTA_IMAGENS = os.path.join(os.getcwd(), "Imagens")
os.makedirs(PASTA_IMAGENS, exist_ok=True)

GCODE_PATH = os.path.join(PASTA_IMAGENS, "relevo.gcode")
NC_PATH = os.path.join(PASTA_IMAGENS, "relevo_3d.nc")
PREVIEW_3D = os.path.join(PASTA_IMAGENS, "preview_3d.png")

# Escalas reais em milímetros
LARGURA_MM = 2000   # 2 metros
ALTURA_MM = 3000    # 3 metros
PROFUNDIDADE_MM = 30  # profundidade máxima (Z)

# =============================================================
# FUNÇÃO: SELECIONAR IMAGEM
# =============================================================
def selecionar_imagem():
    Tk().withdraw()
    caminho = filedialog.askopenfilename(
        title="Selecione a imagem para gerar o relevo",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    if not caminho:
        messagebox.showwarning("Aviso", "Nenhuma imagem selecionada.")
        exit()
    return caminho

# =============================================================
# FUNÇÃO: GERAR RELEVO 3D
# =============================================================
def gerar_relevo(imagem_path):
    if not os.path.exists(imagem_path):
        raise FileNotFoundError(f"Imagem '{imagem_path}' não encontrada.")

    print("\n📂 Abrindo imagem...")
    img = Image.open(imagem_path).convert("L")

    # Reduz automaticamente se for muito grande
    max_size = (1000, 1500)
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        print("⚙️  Redimensionando imagem para processamento...")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    print("🔄 Invertendo tons e preparando matriz de relevo...")
    img = ImageOps.invert(img)
    img = img.rotate(-90, expand=True)

    img_resized = img.resize((740, 1900), Image.Resampling.LANCZOS)
    heightmap_array = np.array(img_resized, dtype=np.float32) / 255.0 * PROFUNDIDADE_MM

    print("💾 Salvando heightmap normalizado...")
    heightmap_img = Image.fromarray(np.uint8(heightmap_array / PROFUNDIDADE_MM * 255))
    heightmap_img.save(os.path.join(PASTA_IMAGENS, "heightmap.png"))

    return heightmap_array

# =============================================================
# FUNÇÃO: GERAR VISUALIZAÇÃO 3D
# =============================================================
def gerar_preview_3d(heightmap):
    print("🖼️  Gerando visualização 3D...")

    y, x = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]
    fig = go.Figure(data=[go.Surface(z=heightmap, x=x, y=y, colorscale="gray")])
    fig.update_layout(
        title="Pré-visualização 3D — Relevo em Madeira",
        scene=dict(
            xaxis_title="Largura (mm)",
            yaxis_title="Altura (mm)",
            zaxis_title="Profundidade (mm)",
            aspectratio=dict(x=2, y=3, z=0.2),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_image(PREVIEW_3D, width=1200, height=900, scale=2)
    print(f"✅ Pré-visualização salva em: {PREVIEW_3D}")

# =============================================================
# FUNÇÃO: GERAR G-CODE E ARQUIVO NC
# =============================================================
def gerar_gcode(heightmap, gcode_path, nc_path):
    print("🛠️  Gerando G-code...")
    h, w = heightmap.shape

    step_x = LARGURA_MM / w
    step_y = ALTURA_MM / h

    with open(gcode_path, "w") as g:
        g.write("(G-code gerado automaticamente)\nG21 ; Unidades em mm\nG90 ; Posição absoluta\n")
        g.write("G0 Z5.000 ; Levantar ferramenta\n")

        for j in range(h):
            linha = heightmap[j]
            if j % 2 == 0:
                xs = range(w)
            else:
                xs = range(w - 1, -1, -1)

            for i in xs:
                x = i * step_x
                y = j * step_y
                z = -linha[i]
                g.write(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F800\n")

            g.write("G0 Z5.000\n")

        g.write("G0 Z10.000\nM30 ; Fim do programa\n")

    os.replace(gcode_path, nc_path)
    print(f"✅ G-code salvo em: {gcode_path}")
    print(f"✅ Arquivo NC salvo em: {nc_path}")

# =============================================================
# FUNÇÃO PRINCIPAL
# =============================================================
def main():
    print("\n=== GERADOR DE RELEVO 3D E G-CODE ===\n")
    imagem_path = selecionar_imagem()

    try:
        heightmap = gerar_relevo(imagem_path)
        gerar_preview_3d(heightmap)
        gerar_gcode(heightmap, GCODE_PATH, NC_PATH)
        messagebox.showinfo("Concluído", "✅ Relevo 3D e G-code gerados com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        raise

if __name__ == "__main__":
    main()
