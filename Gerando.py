import os
import numpy as np
from PIL import Image, ImageOps
from tkinter import Tk, filedialog, messagebox, Label, Button, Entry, StringVar, Frame
from tkinter.ttk import Progressbar
import threading
import plotly.graph_objects as go

# =============================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================
PASTA_IMAGENS = os.path.join(os.getcwd(), "Imagens")
os.makedirs(PASTA_IMAGENS, exist_ok=True)

# =============================================================
# FUNÇÕES PRINCIPAIS
# =============================================================

def gerar_relevo(imagem_path, profundidade_mm):
    """Converte imagem em heightmap normalizado."""
    img = Image.open(imagem_path).convert("L")

    # Reduz se for muito grande (para desempenho)
    max_size = (1200, 1200)
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Inverte para que áreas claras sejam altas
    img = ImageOps.invert(img)

    # Cria heightmap
    heightmap = np.array(img, dtype=np.float32)
    heightmap = (heightmap / 255.0) * profundidade_mm

    # Salva heightmap visual
    heightmap_img = Image.fromarray(np.uint8(heightmap / profundidade_mm * 255))
    heightmap_img.save(os.path.join(PASTA_IMAGENS, "heightmap.png"))

    return heightmap


def gerar_preview_3d(heightmap):
    """Gera o preview 3D do relevo sem abrir visualizador."""
    print("🔧 Gerando preview 3D...")
    y, x = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]

    fig = go.Figure(data=[go.Surface(z=heightmap, x=x, y=y, colorscale="gray", showscale=False)])
    fig.update_layout(
        title="Preview 3D — Relevo CNC",
        scene=dict(
            aspectmode="data",
            xaxis_title="Largura (px)",
            yaxis_title="Altura (px)",
            zaxis_title="Profundidade (mm)",
            camera=dict(eye=dict(x=1.3, y=1.2, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    preview_path = os.path.join(PASTA_IMAGENS, "preview_3d.png")
    fig.write_image(preview_path, width=1000, height=800, scale=2)
    print(f"✅ Preview 3D salvo em: {preview_path}")


def gerar_gcode(heightmap, largura_mm, altura_mm, profundidade_mm):
    """Gera o G-code e arquivo NC do relevo."""
    print("🧩 Gerando G-code...")
    h, w = heightmap.shape
    step_x = largura_mm / w
    step_y = altura_mm / h

    gcode_path = os.path.join(PASTA_IMAGENS, "relevo.gcode")
    nc_path = os.path.join(PASTA_IMAGENS, "relevo_3d.nc")

    with open(gcode_path, "w") as g:
        g.write("(G-code gerado automaticamente)\nG21 ; Unidades em mm\nG90 ; Posição absoluta\n")
        g.write("G0 Z5.000 ; Levantar ferramenta\n")

        for j in range(h):
            linha = heightmap[j]
            xs = range(w) if j % 2 == 0 else range(w - 1, -1, -1)

            for i in xs:
                x = i * step_x
                y = j * step_y
                z = -linha[i]  # Z negativo = corte
                g.write(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F800\n")

            g.write("G0 Z5.000\n")

        g.write("G0 Z10.000\nM30 ; Fim do programa\n")

    # Gera cópia NC
    with open(nc_path, "w") as nc:
        nc.write(open(gcode_path).read())

    print(f"✅ G-code salvo em: {gcode_path}")
    print(f"✅ Arquivo NC salvo em: {nc_path}")

# =============================================================
# INTERFACE (Tkinter)
# =============================================================
class App:
    def __init__(self, master):
        self.master = master
        master.title("Gerador de Relevo 3D e G-code CNC")
        master.geometry("500x400")
        master.configure(bg="#1c1c1c")

        self.imagem_path = None
        self.largura_var = StringVar(value="2000")
        self.altura_var = StringVar(value="3000")
        self.profundidade_var = StringVar(value="30")

        Label(master, text="GERADOR DE RELEVO 3D E G-CODE", fg="#00ccff", bg="#1c1c1c", font=("Arial", 14, "bold")).pack(pady=10)
        Frame(master, height=2, bg="#444").pack(fill="x", pady=10)

        Button(master, text="Selecionar Imagem", command=self.selecionar_imagem, bg="#00ccff", fg="white", font=("Arial", 12), width=20).pack(pady=10)

        self.label_status = Label(master, text="Nenhuma imagem selecionada", fg="white", bg="#1c1c1c", font=("Arial", 10))
        self.label_status.pack(pady=5)

        # Parâmetros personalizados
        frame_param = Frame(master, bg="#1c1c1c")
        frame_param.pack(pady=10)

        Label(frame_param, text="Largura (mm):", bg="#1c1c1c", fg="white").grid(row=0, column=0, sticky="e", padx=5)
        Entry(frame_param, textvariable=self.largura_var, width=10).grid(row=0, column=1)

        Label(frame_param, text="Altura (mm):", bg="#1c1c1c", fg="white").grid(row=1, column=0, sticky="e", padx=5)
        Entry(frame_param, textvariable=self.altura_var, width=10).grid(row=1, column=1)

        Label(frame_param, text="Profundidade (mm):", bg="#1c1c1c", fg="white").grid(row=2, column=0, sticky="e", padx=5)
        Entry(frame_param, textvariable=self.profundidade_var, width=10).grid(row=2, column=1)

        Button(master, text="Gerar Arquivos", command=self.iniciar_geracao, bg="#00cc66", fg="white", font=("Arial", 12), width=20).pack(pady=10)

        self.progress = Progressbar(master, orient="horizontal", length=350, mode="determinate")
        self.progress.pack(pady=10)

    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(title="Selecione a imagem", filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if caminho:
            self.imagem_path = caminho
            self.label_status.config(text=f"Imagem selecionada: {os.path.basename(caminho)}", fg="#00ff99")
        else:
            self.label_status.config(text="Nenhuma imagem selecionada", fg="red")

    def iniciar_geracao(self):
        if not self.imagem_path:
            messagebox.showwarning("Aviso", "Selecione uma imagem primeiro!")
            return

        self.progress["value"] = 0
        self.label_status.config(text="Processando...", fg="#ffff00")
        threading.Thread(target=self.processar).start()

    def processar(self):
        try:
            largura = float(self.largura_var.get())
            altura = float(self.altura_var.get())
            profundidade = float(self.profundidade_var.get())

            self.progress["value"] = 10
            heightmap = gerar_relevo(self.imagem_path, profundidade)

            self.progress["value"] = 40
            gerar_preview_3d(heightmap)

            self.progress["value"] = 80
            gerar_gcode(heightmap, largura, altura, profundidade)

            self.progress["value"] = 100
            self.label_status.config(text="✅ Arquivos gerados com sucesso!", fg="#00ff99")
            messagebox.showinfo("Sucesso", "Arquivos salvos na pasta 'Imagens'.\nVisualize o preview 3D se desejar.")
        except Exception as e:
            self.label_status.config(text=f"Erro: {e}", fg="red")
            messagebox.showerror("Erro", str(e))

# =============================================================
# EXECUÇÃO
# =============================================================
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
