import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


# =====================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =====================================================

def gerar_gcode(
    entrada_img_path,
    largura_mm=2000,
    altura_mm=3000,
    profundidade_max_mm=6.0,
    feed_rate=800,
    safe_z=5.0,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Pasta Imagens
    imagens_dir = os.path.join(base_dir, "Imagens")
    os.makedirs(imagens_dir, exist_ok=True)

    # Saídas
    saida_heightmap = os.path.join(imagens_dir, "Heightmap_Porta.png")
    saida_preview = os.path.join(imagens_dir, "Preview_3D.png")
    saida_gcode = os.path.join(imagens_dir, "3d.nc")

    print("🔄 Processando imagem...")

    # Abrir imagem
    img = Image.open(entrada_img_path).convert("L")
    img = img.rotate(-90, expand=True)
    img = ImageOps.invert(img)

    largura_px, altura_px = img.size

    # Redimensionar para proporção física desejada (2m x 3m)
    img_resized = img.resize((int(largura_mm / 2.7), int(altura_mm / 2.7)), Image.LANCZOS)

    # Converter imagem em matriz de profundidades
    heightmap_array = np.array(img_resized) / 255.0 * profundidade_max_mm

    # Salvar heightmap
    heightmap_img = Image.fromarray(np.uint8(heightmap_array / profundidade_max_mm * 255))
    heightmap_img.save(saida_heightmap)
    print(f"✅ Heightmap salvo em: {saida_heightmap}")

    # =====================================================
    # VISUALIZAÇÃO 3D
    # =====================================================
    print("🖼️ Gerando visualização 3D...")

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(0, 1, heightmap_array.shape[1])
    y = np.linspace(0, 1, heightmap_array.shape[0])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, heightmap_array, cmap="gray", linewidth=0, antialiased=False)
    ax.set_title("Pré-visualização 3D do Relevo")
    ax.set_xlabel("Largura (m)")
    ax.set_ylabel("Altura (m)")
    ax.set_zlabel("Profundidade (mm)")

    plt.savefig(saida_preview)
    plt.close(fig)
    print(f"✅ Preview 3D salvo em: {saida_preview}")

    # =====================================================
    # GERAÇÃO DO G-CODE
    # =====================================================
    print("⚙️ Gerando G-code...")

    escala_x = largura_mm / heightmap_array.shape[1]
    escala_y = altura_mm / heightmap_array.shape[0]

    with open(saida_gcode, "w") as f:
        f.write("(G-code gerado automaticamente por Gerando.py)\n")
        f.write("G21 ; unidades em mm\n")
        f.write("G90 ; coordenadas absolutas\n")
        f.write(f"G0 Z{safe_z:.2f}\n\n")

        for j, linha in enumerate(heightmap_array):
            if j % 2 == 0:
                linha_iter = enumerate(linha)
            else:
                linha_iter = reversed(list(enumerate(linha)))

            y_mm = j * escala_y

            for i, altura in linha_iter:
                x_mm = i * escala_x
                z_mm = -altura
                f.write(f"G1 X{x_mm:.3f} Y{y_mm:.3f} Z{z_mm:.3f} F{feed_rate}\n")

            f.write(f"G0 Z{safe_z:.2f}\n")

        f.write("G0 Z5.000\nM30\n")

    print(f"✅ G-code salvo em: {saida_gcode}")

    messagebox.showinfo("Concluído", "Processo finalizado com sucesso!\nArquivos salvos em 'Imagens'.")


# =====================================================
# INTERFACE GRÁFICA (Tkinter)
# =====================================================

def abrir_interface():
    def selecionar_imagem():
        caminho = filedialog.askopenfilename(
            title="Selecione a imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if caminho:
            entrada_path.set(caminho)

    def iniciar_processamento():
        try:
            if not entrada_path.get():
                messagebox.showwarning("Atenção", "Selecione uma imagem de entrada!")
                return

            largura = float(largura_var.get())
            altura = float(altura_var.get())
            profundidade = float(profundidade_var.get())

            gerar_gcode(
                entrada_img_path=entrada_path.get(),
                largura_mm=largura,
                altura_mm=altura,
                profundidade_max_mm=profundidade
            )
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}")

    # Criar janela
    root = tk.Tk()
    root.title("Gerador de G-code 3D — Entalhe em Madeira")
    root.geometry("480x380")
    root.resizable(False, False)

    tk.Label(root, text="🪵 GERADOR DE G-CODE 3D", font=("Segoe UI", 14, "bold")).pack(pady=10)

    entrada_path = tk.StringVar()
    largura_var = tk.StringVar(value="2000")
    altura_var = tk.StringVar(value="3000")
    profundidade_var = tk.StringVar(value="6.0")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    tk.Label(frame, text="Imagem de Entrada:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(frame, textvariable=entrada_path, width=35).grid(row=0, column=1)
    tk.Button(frame, text="📂 Selecionar", command=selecionar_imagem).grid(row=0, column=2, padx=5)

    tk.Label(frame, text="Largura (mm):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(frame, textvariable=largura_var, width=10).grid(row=1, column=1, sticky="w")

    tk.Label(frame, text="Altura (mm):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(frame, textvariable=altura_var, width=10).grid(row=2, column=1, sticky="w")

    tk.Label(frame, text="Profundidade máx (mm):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(frame, textvariable=profundidade_var, width=10).grid(row=3, column=1, sticky="w")

    tk.Button(root, text="🚀 Gerar Heightmap e G-code", font=("Segoe UI", 11, "bold"), command=iniciar_processamento).pack(pady=15)

    tk.Label(root, text="Arquivos serão salvos na pasta: /Imagens", font=("Segoe UI", 9)).pack(pady=10)
    tk.Label(root, text="© 2025 — Projeto Misael Andrejezieski", font=("Segoe UI", 8, "italic")).pack(side="bottom", pady=5)

    root.mainloop()


# =====================================================
# EXECUÇÃO
# =====================================================
if __name__ == "__main__":
    abrir_interface()
