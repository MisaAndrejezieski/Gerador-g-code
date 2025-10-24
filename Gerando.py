import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# ==============================================
# Funções principais
# ==============================================

def processar_imagem(img_path, largura_mm, altura_mm, profundidade_max, passo, feedrate, safe_z):
    try:
        # Criar diretório de saída
        output_dir = os.path.join(os.getcwd(), "Imagens")
        os.makedirs(output_dir, exist_ok=True)

        # Abrir e tratar imagem
        img = Image.open(img_path).convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = ImageOps.invert(img)

        # Redimensionar conforme proporção original
        img_resized = img.resize((int(largura_mm / passo), int(altura_mm / passo)), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0

        # Normalizar profundidade
        z_map = -img_array * profundidade_max  # negativo = cortar para baixo

        # Salvar imagem tratada
        heightmap_img = Image.fromarray(np.uint8((img_array) * 255))
        heightmap_path = os.path.join(output_dir, "Heightmap.png")
        heightmap_img.save(heightmap_path)

        # Gerar G-code
        gcode_path = os.path.join(output_dir, "3d.nc")
        with open(gcode_path, "w") as f:
            f.write(f"(G-code gerado automaticamente)\n")
            f.write("G21 ; Unidades em mm\n")
            f.write("G90 ; Modo absoluto\n")
            f.write(f"G0 Z{safe_z:.3f}\n")

            linhas, colunas = z_map.shape
            for y in range(linhas):
                if y % 2 == 0:
                    x_range = range(colunas)
                else:
                    x_range = range(colunas - 1, -1, -1)

                for x in x_range:
                    z = z_map[y, x]
                    pos_x = x * passo
                    pos_y = y * passo
                    f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z:.3f} F{feedrate}\n")
                f.write(f"G0 Z{safe_z:.3f}\n")

            f.write("G0 X0 Y0\nM30 ; Fim do programa\n")

        return heightmap_path, gcode_path

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao processar: {e}")
        return None, None


# ==============================================
# Interface Gráfica
# ==============================================

def selecionar_imagem():
    caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
    if caminho:
        entry_imagem.delete(0, tk.END)
        entry_imagem.insert(0, caminho)

def gerar():
    try:
        img_path = entry_imagem.get()
        if not os.path.exists(img_path):
            messagebox.showwarning("Aviso", "Selecione uma imagem válida.")
            return

        largura_mm = float(entry_largura.get())
        altura_mm = float(entry_altura.get())
        profundidade_max = float(entry_profundidade.get())
        passo = float(entry_passo.get())
        feedrate = float(entry_feed.get())
        safe_z = float(entry_safez.get())

        heightmap, gcode = processar_imagem(
            img_path, largura_mm, altura_mm, profundidade_max, passo, feedrate, safe_z
        )

        if gcode:
            messagebox.showinfo("Sucesso", f"G-code gerado com sucesso!\n\nArquivos salvos em:\n{os.path.dirname(gcode)}")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao gerar: {e}")


# ==============================================
# Janela Principal
# ==============================================

root = tk.Tk()
root.title("Gerador de G-code - Relevo 3D para CNC")
root.geometry("600x500")
root.resizable(False, False)

style = ttk.Style(root)
style.configure("TLabel", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10, "bold"))
style.configure("TEntry", font=("Segoe UI", 10))

frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)

# Campos
ttk.Label(frame, text="Imagem de entrada:").grid(row=0, column=0, sticky="w")
entry_imagem = ttk.Entry(frame, width=50)
entry_imagem.grid(row=1, column=0, padx=5, pady=5)
ttk.Button(frame, text="Selecionar", command=selecionar_imagem).grid(row=1, column=1, padx=5)

ttk.Label(frame, text="Largura (mm):").grid(row=2, column=0, sticky="w", pady=(10,0))
entry_largura = ttk.Entry(frame)
entry_largura.insert(0, "2000")
entry_largura.grid(row=3, column=0, pady=5)

ttk.Label(frame, text="Altura (mm):").grid(row=4, column=0, sticky="w")
entry_altura = ttk.Entry(frame)
entry_altura.insert(0, "3000")
entry_altura.grid(row=5, column=0, pady=5)

ttk.Label(frame, text="Profundidade máxima (mm):").grid(row=6, column=0, sticky="w")
entry_profundidade = ttk.Entry(frame)
entry_profundidade.insert(0, "6")
entry_profundidade.grid(row=7, column=0, pady=5)

ttk.Label(frame, text="Passo entre linhas (mm):").grid(row=8, column=0, sticky="w")
entry_passo = ttk.Entry(frame)
entry_passo.insert(0, "2")
entry_passo.grid(row=9, column=0, pady=5)

ttk.Label(frame, text="Velocidade de avanço (mm/min):").grid(row=10, column=0, sticky="w")
entry_feed = ttk.Entry(frame)
entry_feed.insert(0, "800")
entry_feed.grid(row=11, column=0, pady=5)

ttk.Label(frame, text="Safe Z (mm):").grid(row=12, column=0, sticky="w")
entry_safez = ttk.Entry(frame)
entry_safez.insert(0, "5")
entry_safez.grid(row=13, column=0, pady=5)

ttk.Button(frame, text="GERAR G-CODE", command=gerar).grid(row=14, column=0, columnspan=2, pady=20)

ttk.Label(frame, text="© 2025 - Gerador CNC by Misael & GPT-5", font=("Segoe UI", 8)).grid(row=15, column=0, columnspan=2, pady=10)

root.mainloop()
