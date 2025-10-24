import os
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkhtmlview import HTMLLabel


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
    visualizar_callback=None
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    imagens_dir = os.path.join(base_dir, "Imagens")
    os.makedirs(imagens_dir, exist_ok=True)

    saida_heightmap = os.path.join(imagens_dir, "Heightmap_Porta.png")
    saida_preview_html = os.path.join(imagens_dir, "Preview_3D.html")
    saida_gcode = os.path.join(imagens_dir, "3d.nc")

    # =====================================================
    # PROCESSAMENTO DA IMAGEM
    # =====================================================
    print("🔄 Processando imagem...")

    img = Image.open(entrada_img_path).convert("L")
    img = img.rotate(-90, expand=True)
    img = ImageOps.invert(img)

    img_resized = img.resize((int(largura_mm / 2.7), int(altura_mm / 2.7)), Image.LANCZOS)
    heightmap_array = np.array(img_resized) / 255.0 * profundidade_max_mm

    heightmap_img = Image.fromarray(np.uint8(heightmap_array / profundidade_max_mm * 255))
    heightmap_img.save(saida_heightmap)

    print(f"✅ Heightmap salvo em: {saida_heightmap}")

    # =====================================================
    # VISUALIZAÇÃO 3D (PLOTLY)
    # =====================================================
    print("🖼️ Gerando visualização 3D...")

    x = np.linspace(0, largura_mm, heightmap_array.shape[1])
    y = np.linspace(0, altura_mm, heightmap_array.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=heightmap_array,
                colorscale="gray",
                lighting=dict(ambient=0.5, diffuse=0.9, specular=0.5, roughness=0.5),
                lightposition=dict(x=100, y=200, z=1000),
            )
        ]
    )

    fig.update_layout(
        title="Pré-visualização 3D — Relevo em Madeira",
        scene=dict(
            xaxis_title="Largura (mm)",
            yaxis_title="Altura (mm)",
            zaxis_title="Profundidade (mm)",
            aspectratio=dict(x=1, y=1.5, z=0.1),
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.write_html(saida_preview_html)
    print(f"✅ Visualização salva em: {saida_preview_html}")

    if visualizar_callback:
        visualizar_callback(saida_preview_html)

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

    messagebox.showinfo(
        "Concluído",
        "Processo finalizado com sucesso!\n"
        "Arquivos salvos em: pasta 'Imagens'\n\n"
        "✔ Heightmap_Porta.png\n✔ 3d.nc\n✔ Preview_3D.html"
    )


# =====================================================
# INTERFACE GRÁFICA
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
                messagebox.showwarning("Atenção", "Selecione uma imagem!")
                return

            largura = float(largura_var.get())
            altura = float(altura_var.get())
            profundidade = float(profundidade_var.get())

            gerar_gcode(
                entrada_img_path=entrada_path.get(),
                largura_mm=largura,
                altura_mm=altura,
                profundidade_max_mm=profundidade,
                visualizar_callback=mostrar_visualizacao
            )
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}")

    def mostrar_visualizacao(html_path):
        with open(html_path, "r", encoding="utf-8") as file:
            conteudo = file.read()
        preview_html.set_html(conteudo)

    root = tk.Tk()
    root.title("Gerador de G-code 3D — Entalhe em Madeira")
    root.geometry("900x700")
    root.configure(bg="#2b2b2b")

    tk.Label(root, text="🪵 GERADOR DE G-CODE 3D", font=("Segoe UI", 18, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)

    frame = tk.Frame(root, bg="#2b2b2b")
    frame.pack(pady=10)

    entrada_path = tk.StringVar()
    largura_var = tk.StringVar(value="2000")
    altura_var = tk.StringVar(value="3000")
    profundidade_var = tk.StringVar(value="6.0")

    tk.Label(frame, text="Imagem:", bg="#2b2b2b", fg="white").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(frame, textvariable=entrada_path, width=50).grid(row=0, column=1)
    tk.Button(frame, text="📂 Selecionar", command=selecionar_imagem).grid(row=0, column=2, padx=5)

    tk.Label(frame, text="Largura (mm):", bg="#2b2b2b", fg="white").grid(row=1, column=0, sticky="e", padx=5)
    tk.Entry(frame, textvariable=largura_var, width=10).grid(row=1, column=1, sticky="w")

    tk.Label(frame, text="Altura (mm):", bg="#2b2b2b", fg="white").grid(row=2, column=0, sticky="e", padx=5)
    tk.Entry(frame, textvariable=altura_var, width=10).grid(row=2, column=1, sticky="w")

    tk.Label(frame, text="Profundidade máx (mm):", bg="#2b2b2b", fg="white").grid(row=3, column=0, sticky="e", padx=5)
    tk.Entry(frame, textvariable=profundidade_var, width=10).grid(row=3, column=1, sticky="w")

    tk.Button(
        root,
        text="🚀 Gerar Heightmap + G-code + Visualização",
        font=("Segoe UI", 11, "bold"),
        command=iniciar_processamento,
        bg="#4CAF50",
        fg="white",
        relief="raised",
        padx=10,
        pady=5
    ).pack(pady=15)

    ttk.Separator(root, orient="horizontal").pack(fill="x", pady=10)

    preview_html = HTMLLabel(root, html="<h3 style='color:white;'>Pré-visualização 3D aparecerá aqui...</h3>")
    preview_html.pack(fill="both", expand=True, padx=10, pady=10)

    tk.Label(root, text="© 2025 — Projeto Misael Andrejezieski", bg="#2b2b2b", fg="#aaaaaa", font=("Segoe UI", 9, "italic")).pack(pady=5)

    root.mainloop()


# =====================================================
# EXECUÇÃO
# =====================================================

if __name__ == "__main__":
    abrir_interface()
