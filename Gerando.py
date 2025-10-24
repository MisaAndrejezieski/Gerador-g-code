import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# ================================
# CONFIGURAÇÕES PRINCIPAIS
# ================================
PASTA_IMAGENS = os.path.join(os.getcwd(), "Imagens")
os.makedirs(PASTA_IMAGENS, exist_ok=True)

profundidade_max_mm = 3.0  # profundidade máxima (em mm)
resolucao_saida = (2000, 2000)  # resolução máxima da imagem
passo_mm = 1.0  # distância entre linhas do G-code
feedrate = 800  # velocidade de corte
safe_z = 2.0  # altura segura entre deslocamentos

# ================================
# FUNÇÕES DE PROCESSAMENTO
# ================================
def preprocessar_imagem(path):
    """Trata automaticamente a imagem para gerar relevo ideal."""
    img = Image.open(path).convert("L")

    # Autoajuste de contraste
    img = ImageOps.autocontrast(img, cutoff=2)

    # Desfoque suave
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Detecta inversão (maioria escura = imagem invertida)
    img_array = np.array(img)
    if np.mean(img_array) < 128:
        img = ImageOps.invert(img)
        print("➡️ Imagem invertida automaticamente para manter relevo correto.")

    # Redimensiona mantendo proporção
    img.thumbnail(resolucao_saida, Image.LANCZOS)

    # Normaliza a altura
    img_array = np.array(img)
    heightmap = (img_array / 255.0) * profundidade_max_mm

    return img, heightmap


def gerar_gcode(heightmap, caminho_saida):
    """Gera o G-code a partir do heightmap processado."""
    altura, largura = heightmap.shape
    with open(caminho_saida, "w") as f:
        f.write("; --- G-code gerado automaticamente ---\n")
        f.write("G21 ; Unidades em milímetros\n")
        f.write("G90 ; Coordenadas absolutas\n")
        f.write(f"G0 Z{safe_z}\n")

        for y in range(altura):
            linha = heightmap[y]
            if y % 2 == 0:
                xs = range(largura)
            else:
                xs = range(largura - 1, -1, -1)
            for x in xs:
                z = -linha[x]  # negativo = corte para baixo
                f.write(f"G1 X{x * passo_mm:.2f} Y{y * passo_mm:.2f} Z{z:.3f} F{feedrate}\n")
            f.write(f"G0 Z{safe_z}\n")

        f.write("G0 Z{safe_z}\nM30\n")

    print(f"G-code salvo em: {caminho_saida}")


def gerar_preview_3d(heightmap):
    """Gera preview 3D simples (sem abrir janela principal)."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    y = np.linspace(0, heightmap.shape[0], heightmap.shape[0])
    x = np.linspace(0, heightmap.shape[1], heightmap.shape[1])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, -heightmap, cmap="gray", linewidth=0, antialiased=False)
    ax.set_title("Prévia 3D do Relevo")
    ax.set_xlabel("Largura (mm)")
    ax.set_ylabel("Altura (mm)")
    ax.set_zlabel("Profundidade (mm)")

    preview_path = os.path.join(PASTA_IMAGENS, "preview_3d.png")
    plt.savefig(preview_path)
    plt.close(fig)
    print(f"Prévia 3D salva em: {preview_path}")


# ================================
# INTERFACE GRÁFICA
# ================================
def selecionar_imagem():
    caminho = filedialog.askopenfilename(
        title="Selecione a imagem",
        filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
    )
    if not caminho:
        return

    try:
        img, heightmap = preprocessar_imagem(caminho)
        base = os.path.splitext(os.path.basename(caminho))[0]

        # Salvar heightmap
        heightmap_path = os.path.join(PASTA_IMAGENS, f"{base}_heightmap.png")
        Image.fromarray(np.uint8((heightmap / profundidade_max_mm) * 255)).save(heightmap_path)

        # Gerar G-code
        gcode_path = os.path.join(PASTA_IMAGENS, f"{base}.gcode")
        gerar_gcode(heightmap, gcode_path)

        # Gerar NC
        nc_path = os.path.join(PASTA_IMAGENS, f"{base}_3d.nc")
        gerar_gcode(heightmap, nc_path)

        # Gerar prévia
        gerar_preview_3d(heightmap)

        messagebox.showinfo("Concluído", f"Arquivos gerados:\n\n- {base}.gcode\n- {base}_3d.nc\n- preview_3d.png")

    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao processar imagem:\n{e}")


# ================================
# EXECUÇÃO PRINCIPAL
# ================================
def main():
    root = tk.Tk()
    root.title("Gerador de G-code Automático para Relevo 3D")
    root.geometry("450x200")

    tk.Label(root, text="Gerador Automático de G-code CNC", font=("Segoe UI", 14, "bold")).pack(pady=20)
    tk.Button(root, text="Selecionar Imagem", command=selecionar_imagem, font=("Segoe UI", 12), width=25).pack(pady=10)
    tk.Label(root, text="Cria automaticamente: heightmap, .gcode, .nc e preview 3D", font=("Segoe UI", 9)).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
