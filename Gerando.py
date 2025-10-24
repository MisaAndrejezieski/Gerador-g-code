import os
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
from tkinter import Tk, filedialog, messagebox, Label, Button, PhotoImage, Canvas, Frame
from tkinter.ttk import Progressbar
import threading

# =============================================================
# CONFIGURAÇÕES
# =============================================================
PASTA_IMAGENS = os.path.join(os.getcwd(), "Imagens")
os.makedirs(PASTA_IMAGENS, exist_ok=True)

GCODE_PATH = os.path.join(PASTA_IMAGENS, "relevo.gcode")
NC_PATH = os.path.join(PASTA_IMAGENS, "relevo_3d.nc")
PREVIEW_3D = os.path.join(PASTA_IMAGENS, "preview_3d.png")

LARGURA_MM = 2000
ALTURA_MM = 3000
PROFUNDIDADE_MM = 30

# =============================================================
# FUNÇÕES PRINCIPAIS
# =============================================================
def gerar_relevo(imagem_path):
    img = Image.open(imagem_path).convert("L")

    # Reduz automaticamente se for muito grande
    max_size = (1200, 1200)
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    img = ImageOps.invert(img)

    heightmap_array = np.array(img, dtype=np.float32) / 255.0 * PROFUNDIDADE_MM
    heightmap_img = Image.fromarray(np.uint8(heightmap_array / PROFUNDIDADE_MM * 255))
    heightmap_img.save(os.path.join(PASTA_IMAGENS, "heightmap.png"))

    return heightmap_array


def gerar_preview_3d(heightmap):
    y, x = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]

    fig = go.Figure(data=[go.Surface(z=heightmap, x=x, y=y, colorscale="gray", showscale=False)])
    fig.update_layout(
        title="Pré-visualização 3D — Relevo em Madeira",
        scene=dict(
            xaxis_title="Largura (px)",
            yaxis_title="Altura (px)",
            zaxis_title="Profundidade (mm)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.write_image(PREVIEW_3D, width=1000, height=800, scale=2)


def gerar_gcode(heightmap):
    h, w = heightmap.shape
    step_x = LARGURA_MM / w
    step_y = ALTURA_MM / h

    with open(GCODE_PATH, "w") as g:
        g.write("(G-code gerado automaticamente)\nG21 ; Unidades em mm\nG90 ; Posição absoluta\n")
        g.write("G0 Z5.000\n")

        for j in range(h):
            linha = heightmap[j]
            xs = range(w) if j % 2 == 0 else range(w - 1, -1, -1)
            for i in xs:
                x = i * step_x
                y = j * step_y
                z = -linha[i]
                g.write(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F800\n")
            g.write("G0 Z5.000\n")

        g.write("G0 Z10.000\nM30\n")

    with open(NC_PATH, "w") as f:
        f.write(open(GCODE_PATH).read())

# =============================================================
# INTERFACE GRÁFICA (TKINTER)
# =============================================================
class App:
    def __init__(self, master):
        self.master = master
        master.title("Gerador de Relevo 3D e G-Code")
        master.geometry("600x500")
        master.configure(bg="#1c1c1c")

        self.imagem_path = None

        Label(master, text="GERADOR DE RELEVO 3D E G-CODE", fg="#00ccff", bg="#1c1c1c", font=("Arial", 14, "bold")).pack(pady=10)
        Label(master, text="Criação de relevo CNC a partir de imagem", fg="#aaaaaa", bg="#1c1c1c", font=("Arial", 10)).pack()

        Frame(master, height=2, bg="#444").pack(fill="x", pady=10)

        self.label_status = Label(master, text="Selecione uma imagem para começar", fg="#ffffff", bg="#1c1c1c", font=("Arial", 11))
        self.label_status.pack(pady=10)

        Button(master, text="Selecionar Imagem", command=self.selecionar_imagem, bg="#00ccff", fg="white", font=("Arial", 12), width=20).pack(pady=10)
        Button(master, text="Gerar G-code", command=self.iniciar_geracao, bg="#00cc66", fg="white", font=("Arial", 12), width=20).pack(pady=5)

        self.progress = Progressbar(master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=15)

        self.canvas_preview = Canvas(master, width=400, height=300, bg="#222")
        self.canvas_preview.pack(pady=10)

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
        self.label_status.config(text="Gerando relevo 3D...", fg="#ffff00")
        threading.Thread(target=self.processar).start()

    def processar(self):
        try:
            self.progress["value"] = 10
            heightmap = gerar_relevo(self.imagem_path)
            self.progress["value"] = 40
            gerar_preview_3d(heightmap)
            self.progress["value"] = 70
            gerar_gcode(heightmap)
            self.progress["value"] = 100

            img = PhotoImage(file=PREVIEW_3D)
            self.canvas_preview.create_image(200, 150, image=img)
            self.canvas_preview.image = img

            self.label_status.config(text="✅ Relevo 3D e G-code gerados com sucesso!", fg="#00ff99")
            messagebox.showinfo("Sucesso", "Arquivos gerados na pasta 'Imagens'.")
        except Exception as e:
            self.label_status.config(text=f"Erro: {e}", fg="red")
            messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

# =============================================================
# EXECUÇÃO
# =============================================================
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
