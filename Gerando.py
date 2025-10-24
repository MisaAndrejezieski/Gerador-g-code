# Gerando.py
import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def pil_from_cv2_gray(img_cv):
    """Converte imagem OpenCV (grayscale) para PIL Image (L)."""
    return Image.fromarray(np.uint8(img_cv))

# ---------------------------
# Processamento do heightmap
# ---------------------------
def preprocess_image_cv2(path,
                         rotate_angle=0,
                         invert=False,
                         bg_blur_kernel=101,
                         bilateral_d=9,
                         bilateral_sigmaColor=75,
                         bilateral_sigmaSpace=75,
                         gauss_smooth=3,
                         target_size_px=(740, 1900)):
    """
    Fluxo recomendado:
    1) abre em grayscale
    2) rotaciona se pedido
    3) equaliza iluminação por subtração de fundo (Gaussian blur grande)
    4) normaliza e aplica bilateral (preserva bordas)
    5) suaviza com Gaussian
    6) redimensiona para target_size_px
    Retorna numpy array (float32) 0..255
    """
    # 1) abrir
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {path}")

    # 2) rotacionar se preciso
    if rotate_angle != 0:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 3) estimar iluminação de fundo e remover
    # bg_blur_kernel precisa ser ímpar e grande (ex: 101) para captar iluminação suave
    if bg_blur_kernel % 2 == 0:
        bg_blur_kernel += 1
    background = cv2.GaussianBlur(img, (bg_blur_kernel, bg_blur_kernel), 0)
    # Subtrair fundo e reacentuar (evitar negativos)
    img_sub = cv2.subtract(img, background)
    # Alternativa: img_div = cv2.divide(img, background, scale=255)  # but can amplify noise
    # 4) normalizar para 0..255
    img_norm = cv2.normalize(img_sub, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # 5) bilateral filter para preservar arestas ornamentais
    img_bilat = cv2.bilateralFilter(img_norm, d=bilateral_d, sigmaColor=bilateral_sigmaColor, sigmaSpace=bilateral_sigmaSpace)
    # 6) suavização leve (remover texturas finas)
    if gauss_smooth > 0:
        if gauss_smooth % 2 == 0:
            gauss_smooth += 1
        img_smooth = cv2.GaussianBlur(img_bilat, (gauss_smooth, gauss_smooth), 0)
    else:
        img_smooth = img_bilat

    # 7) opcional inverter tons
    if invert:
        img_smooth = cv2.bitwise_not(img_smooth)

    # 8) redimensionar mantendo proporção para target (largura, altura)
    target_w, target_h = target_size_px
    img_resized = cv2.resize(img_smooth, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return img_resized.astype(np.uint8)


# ---------------------------
# Converter heightmap para profundidade (mm)
# ---------------------------
def heightmap_to_depth_array(img_gray_uint8, profundidade_max_mm=6.0, smooth_mm_sigma=1.0, escala_xy_mm=1.0):
    """
    img_gray_uint8: 0..255
    retorna array float32 com alturas em mm (0 = base, profundidade_max_mm = máximo)
    """
    normalized = img_gray_uint8.astype(np.float32) / 255.0  # 0..1
    depth = normalized * profundidade_max_mm  # mm

    # Aplicar suavização espacial para evitar picos (em pixels)
    # converte sigma_mm para sigma_px via escala_xy_mm (mm/pixel)
    if smooth_mm_sigma > 0:
        sigma_px = max(1.0, smooth_mm_sigma / escala_xy_mm)
        # OpenCV GaussianBlur sigma em pixels via kernel size:
        k = int(6 * sigma_px + 1)
        if k % 2 == 0:
            k += 1
        depth = cv2.GaussianBlur(depth, (k, k), sigmaX=sigma_px, borderType=cv2.BORDER_REPLICATE)

    return depth.astype(np.float32)


# ---------------------------
# G-code generator (zig-zag)
# ---------------------------
def generate_gcode_from_depth(depth_array, out_gcode_path,
                              largura_mm=2000.0, altura_mm=3000.0,
                              feed_rate=800, safe_z=5.0, passo_mm=None):
    """
    depth_array: H x W float32 (mm), 0..profundidade_max
    passo_mm: se None, usa escala (1 pixel -> largura_mm/width mm)
    Gera gcode em out_gcode_path
    """
    h, w = depth_array.shape
    escala_x = largura_mm / float(w)
    escala_y = altura_mm / float(h)
    if passo_mm is None:
        passo_mm = min(escala_x, escala_y)

    with open(out_gcode_path, "w", encoding="utf-8") as f:
        f.write("(G-code gerado automaticamente por Gerando.py)\n")
        f.write("G21 ; mm\nG90 ; absoluta\n")
        f.write(f"G0 Z{safe_z:.3f}\n")
        # Zig-zag por linhas (y)
        for j in range(h):
            if j % 2 == 0:
                xrange = range(0, w)
            else:
                xrange = range(w - 1, -1, -1)
            y_mm = j * escala_y
            for i in xrange:
                x_mm = i * escala_x
                z_mm = -float(depth_array[j, i])  # negativo para descer
                f.write(f"G1 X{x_mm:.3f} Y{y_mm:.3f} Z{z_mm:.3f} F{feed_rate}\n")
            # levantar entre linhas
            f.write(f"G0 Z{safe_z:.3f}\n")
        f.write("G0 X0 Y0 Z5.000\nM30\n")


# ---------------------------
# Plotly preview generator
# ---------------------------
def save_plotly_preview(depth_array, out_html, largura_mm=2000.0, altura_mm=3000.0, out_png=None):
    h, w = depth_array.shape
    xs = np.linspace(0, largura_mm, w)
    ys = np.linspace(0, altura_mm, h)
    X, Y = np.meshgrid(xs, ys)

    surf = go.Surface(x=X, y=Y, z=depth_array, colorscale="gray",
                      lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.6),
                      lightposition=dict(x=1000, y=1000, z=2000),
                      showscale=True)
    fig = go.Figure(data=[surf])
    fig.update_layout(scene=dict(
        xaxis_title="Largura (mm)", yaxis_title="Altura (mm)", zaxis_title="Profundidade (mm)",
        aspectratio=dict(x=1.0, y=1.5, z=0.08)),
        margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
    fig.write_html(out_html)
    if out_png:
        # exige kaleido
        fig.write_image(out_png, scale=2)


# ---------------------------
# GUI
# ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Gerador de Heightmap & G-code (pré-processamento melhorado)")
        root.geometry("980x720")

        base = os.path.dirname(os.path.abspath(__file__))
        self.imagens_dir = os.path.join(base, "Imagens")
        ensure_dir(self.imagens_dir)

        # Vars
        self.input_path = tk.StringVar()
        self.rotate_angle = tk.IntVar(value=0)
        self.invert = tk.BooleanVar(value=False)
        self.bg_blur = tk.IntVar(value=101)
        self.bilat_d = tk.IntVar(value=9)
        self.bilat_sigmaColor = tk.IntVar(value=75)
        self.gauss_smooth = tk.IntVar(value=3)
        self.target_w = tk.IntVar(value=740)
        self.target_h = tk.IntVar(value=1900)
        self.prof_mm = tk.DoubleVar(value=6.0)
        self.smooth_mm = tk.DoubleVar(value=1.5)
        self.larg_mm = tk.DoubleVar(value=2000.0)
        self.alt_mm = tk.DoubleVar(value=3000.0)
        self.feedrate = tk.IntVar(value=800)

        # Left frame: controls
        left = tk.Frame(root, width=360, padx=10, pady=10)
        left.pack(side="left", fill="y")

        tk.Label(left, text="Arquivo de entrada:").pack(anchor="w")
        tk.Entry(left, textvariable=self.input_path, width=45).pack(anchor="w")
        tk.Button(left, text="Selecionar imagem", command=self.select_file).pack(pady=6, anchor="w")

        # Options
        opts = tk.LabelFrame(left, text="Pré-processamento", padx=6, pady=6)
        opts.pack(fill="x", pady=6)
        tk.Checkbutton(opts, text="Inverter (claro->alto)", variable=self.invert).grid(row=0, column=0, sticky="w")
        tk.Label(opts, text="Rotacionar (graus):").grid(row=1, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.rotate_angle, width=6).grid(row=1, column=1, sticky="w")
        tk.Label(opts, text="Blur de fundo (kernel ímpar):").grid(row=2, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.bg_blur, width=6).grid(row=2, column=1, sticky="w")
        tk.Label(opts, text="Bilateral d:").grid(row=3, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.bilat_d, width=6).grid(row=3, column=1, sticky="w")
        tk.Label(opts, text="Gauss smooth (px):").grid(row=4, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.gauss_smooth, width=6).grid(row=4, column=1, sticky="w")

        sizef = tk.LabelFrame(left, text="Tamanho & física", padx=6, pady=6)
        sizef.pack(fill="x", pady=6)
        tk.Label(sizef, text="Largura px:").grid(row=0, column=0, sticky="w")
        tk.Entry(sizef, textvariable=self.target_w, width=8).grid(row=0, column=1)
        tk.Label(sizef, text="Altura px:").grid(row=1, column=0, sticky="w")
        tk.Entry(sizef, textvariable=self.target_h, width=8).grid(row=1, column=1)
        tk.Label(sizef, text="Profundidade (mm):").grid(row=2, column=0, sticky="w")
        tk.Entry(sizef, textvariable=self.prof_mm, width=8).grid(row=2, column=1)
        tk.Label(sizef, text="Smooth (mm):").grid(row=3, column=0, sticky="w")
        tk.Entry(sizef, textvariable=self.smooth_mm, width=8).grid(row=3, column=1)
        tk.Label(sizef, text="Largura peça (mm):").grid(row=4, column=0, sticky="w")
        tk.Entry(sizef, textvariable=self.larg_mm, width=8).grid(row=4, column=1)
        tk.Label(sizef, text="Altura peça (mm):").grid(row=5, column=0, sticky="w")
        tk.Entry(sizef, textvariable=self.alt_mm, width=8).grid(row=5, column=1)

        tk.Label(left, text="Feedrate (mm/min):").pack(anchor="w", pady=(6,0))
        tk.Entry(left, textvariable=self.feedrate, width=10).pack(anchor="w")

        tk.Button(left, text="Gerar heightmap → preview → gcode", bg="#4CAF50", fg="white",
                  command=self.run_all).pack(fill="x", pady=10)

        # Right frame: previews
        right = tk.Frame(root)
        right.pack(side="right", expand=True, fill="both")

        # image preview (input)
        self.label_in = tk.Label(right, text="Entrada", bd=1, relief="sunken")
        self.label_in.pack(padx=6, pady=6)
        self.canvas_in = tk.Label(right)
        self.canvas_in.pack()

        # image preview (heightmap)
        tk.Label(right, text="Heightmap (prévia)").pack()
        self.canvas_out = tk.Label(right)
        self.canvas_out.pack()

        # status
        self.status = tk.StringVar(value="Pronto")
        tk.Label(right, textvariable=self.status).pack(pady=6)

    def select_file(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")])
        if p:
            self.input_path.set(p)
            # show small preview
            img = Image.open(p)
            img.thumbnail((420, 300))
            imgtk = ImageTk.PhotoImage(img.convert("RGB"))
            self.canvas_in.image = imgtk
            self.canvas_in.config(image=imgtk)

    def run_all(self):
        try:
            if not self.input_path.get():
                messagebox.showwarning("Atenção", "Selecione a imagem de entrada.")
                return
            self.status.set("Processando...")
            # preprocess
            pre = preprocess_image_cv2(
                self.input_path.get(),
                rotate_angle=self.rotate_angle.get(),
                invert=self.invert.get(),
                bg_blur_kernel=self.bg_blur.get(),
                bilateral_d=self.bilat_d.get(),
                bilateral_sigmaColor=self.bilat_sigmaColor.get(),
                bilateral_sigmaSpace=self.bilat_sigmaColor.get(),
                gauss_smooth=self.gauss_smooth.get(),
                target_size_px=(self.target_w.get(), self.target_h.get())
            )

            # show heightmap preview in UI
            pil_hm = pil_from_cv2_gray(pre)
            pil_hm_thumb = pil_hm.copy()
            pil_hm_thumb.thumbnail((420, 300))
            imgtk2 = ImageTk.PhotoImage(pil_hm_thumb.convert("RGB"))
            self.canvas_out.image = imgtk2
            self.canvas_out.config(image=imgtk2)

            # convert to depth (mm)
            depth = heightmap_to_depth_array(pre, profundidade_max_mm=self.prof_mm.get(),
                                             smooth_mm_sigma=self.smooth_mm.get(),
                                             escala_xy_mm=self.larg_mm.get() / float(self.target_w.get()))

            # save heightmap file & preview & gcode
            base = os.path.dirname(os.path.abspath(__file__))
            imagens_dir = os.path.join(base, "Imagens")
            ensure_dir(imagens_dir)
            hm_path = os.path.join(imagens_dir, "Heightmap_Porta.png")
            pil_hm.save(hm_path)
            self.status.set(f"Heightmap salvo: {hm_path}")

            html_preview = os.path.join(imagens_dir, "Preview_3D.html")
            png_preview = os.path.join(imagens_dir, "Preview_3D.png")
            save_plotly_preview(depth, html_preview, largura_mm=self.larg_mm.get(), altura_mm=self.alt_mm.get(),
                                out_png=png_preview)
            self.status.set(f"Preview salvo: {html_preview}")

            gcode_path = os.path.join(imagens_dir, "3d.nc")
            generate_gcode_from_depth(depth, gcode_path, largura_mm=self.larg_mm.get(), altura_mm=self.alt_mm.get(),
                                      feed_rate=self.feedrate.get(), safe_z=5.0)
            self.status.set(f"G-code salvo: {gcode_path}")

            messagebox.showinfo("Concluído", f"Arquivos gerados na pasta: {imagens_dir}")
            self.status.set("Pronto")
        except Exception as e:
            self.status.set("Erro")
            messagebox.showerror("Erro", str(e))

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
