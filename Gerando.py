import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2

# ==============================================
# FUNÇÕES OTIMIZADAS
# ==============================================

def processar_imagem(img_path, largura_mm, altura_mm, profundidade_max, passo, feedrate, safe_z, tipo_relevo="baixo"):
    try:
        # Criar diretório de saída
        output_dir = os.path.join(os.getcwd(), "Imagens_Processadas")
        os.makedirs(output_dir, exist_ok=True)

        # Abrir e tratar imagem
        img = Image.open(img_path).convert("L")
        
        # PRÉ-PROCESSAMENTO MELHORADO
        img = ImageEnhance.Contrast(img).enhance(1.5)  # Aumenta contraste
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = img.filter(ImageFilter.SHARPEN)  # Realça detalhes
        
        # CORREÇÃO: Respeitar proporção original da imagem
        img_ratio = img.width / img.height
        target_ratio = largura_mm / altura_mm
        
        if img_ratio > target_ratio:
            # Imagem mais larga
            new_width = int(largura_mm / passo)
            new_height = int(new_width / img_ratio)
        else:
            # Imagem mais alta
            new_height = int(altura_mm / passo)
            new_width = int(new_height * img_ratio)
        
        # Redimensionar mantendo proporção
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0

        # CONTROLE DE TIPO DE RELEVO
        if tipo_relevo == "baixo":
            z_map = (1 - img_array) * profundidade_max  # Áreas claras = mais profundas
        else:  # alto relevo
            z_map = img_array * profundidade_max  # Áreas escuras = mais profundas

        # Suavizar transições (evita movimentos bruscos)
        kernel = np.ones((3,3), np.float32)/9
        z_map_smooth = cv2.filter2D(z_map, -1, kernel)

        # Salvar imagem tratada
        heightmap_img = Image.fromarray(np.uint8((z_map_smooth / profundidade_max) * 255))
        heightmap_path = os.path.join(output_dir, "Heightmap_Processado.png")
        heightmap_img.save(heightmap_path)

        # GERAR G-CODE OTIMIZADO
        gcode_path = os.path.join(output_dir, "relevo_3d_otimizado.nc")
        gerar_gcode_otimizado(z_map_smooth, passo, feedrate, safe_z, gcode_path, largura_mm, new_width, new_height)

        return heightmap_path, gcode_path

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao processar: {str(e)}")
        return None, None

def gerar_gcode_otimizado(z_map, passo, feedrate, safe_z, gcode_path, largura_mm, img_width, img_height):
    with open(gcode_path, "w") as f:
        # CABEÇALHO MELHORADO
        f.write("(G-code para Relevo 3D - Gerado Automaticamente)\n")
        f.write("G21 ; Unidades em mm\n")
        f.write("G90 ; Posicionamento absoluto\n")
        f.write("G17 ; Plano XY\n")
        f.write("G94 ; Avanço em mm/min\n")
        f.write("G49 ; Cancela compensação de comprimento\n\n")
        
        # POSICIONAMENTO INICIAL
        f.write(f"; === INICIO DO CORTE ===\n")
        f.write(f"G0 Z{safe_z:.3f} ; Eleva para Safe Z\n")
        f.write(f"G0 X0 Y0 ; Posiciona na origem\n\n")
        
        linhas, colunas = z_map.shape
        f.write(f"; Dimensões: {colunas}x{linhas} pontos, Passo: {passo}mm\n")
        f.write(f"F{feedrate} ; Define velocidade de avanço\n\n")
        
        # ESTRATÉGIA DE CORTE MELHORADA
        for y in range(linhas):
            # Determina direção (zig-zag)
            if y % 2 == 0:
                x_range = range(colunas)
            else:
                x_range = range(colunas - 1, -1, -1)
            
            primeiro_ponto = True
            
            for x in x_range:
                z = z_map[y, x]
                pos_x = (x * passo) - (img_width * passo / 2)  # Centraliza no eixo X
                pos_y = (y * passo) - (img_height * passo / 2)  # Centraliza no eixo Y
                
                if primeiro_ponto:
                    # Move para primeiro ponto da linha com Safe Z
                    f.write(f"G0 X{pos_x:.3f} Y{pos_y:.3f} ; Posiciona na linha {y}\n")
                    f.write(f"G1 Z{z:.3f} ; Desce para cortar\n")
                    primeiro_ponto = False
                else:
                    # Movimento de corte contínuo
                    f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z:.3f} ; Corte\n")
            
            # CORREÇÃO: Só sobe para Safe Z se necessário (mudança de área)
            if y < linhas - 1:
                # Verifica se próxima linha está muito longe
                next_y = y + 1
                if abs(z_map[y, x] - z_map[next_y, x]) > 2:  # Se diferença > 2mm
                    f.write(f"G0 Z{safe_z:.3f} ; Eleva para transição segura\n")
        
        # FINALIZAÇÃO
        f.write(f"\n; === FINALIZACAO ===\n")
        f.write(f"G0 Z{safe_z:.3f} ; Retorna para Safe Z\n")
        f.write("G0 X0 Y0 ; Volta para origem\n")
        f.write("M5 ; Desliga spindle\n")
        f.write("M30 ; Fim do programa\n")
        
        # ESTATÍSTICAS
        f.write(f"\n; === ESTATISTICAS ===\n")
        f.write(f"; Pontos totais: {linhas * colunas}\n")
        f.write(f"; Dimensão real: {img_width * passo:.1f} x {img_height * passo:.1f} mm\n")
        f.write(f"; Tempo estimado: {(linhas * colunas * passo) / (feedrate / 60):.1f} minutos\n")

# ==============================================
# INTERFACE GRÁFICA MELHORADA
# ==============================================

class GeradorCNC:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("Gerador de G-code - CNC Router 3D")
        self.root.geometry("650x600")
        self.root.resizable(True, True)
        
        # Configurar estilo
        self.setup_styles()
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Título
        title = ttk.Label(main_frame, text="GERADOR DE RELEVO 3D PARA CNC", 
                         font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Campos da interface
        self.create_widgets(main_frame)
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        
    def create_widgets(self, parent):
        row = 1
        
        # Imagem de entrada
        ttk.Label(parent, text="Imagem de Entrada:", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(10,5))
        row += 1
        
        self.entry_imagem = ttk.Entry(parent, width=50)
        self.entry_imagem.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(parent, text="Selecionar", command=self.selecionar_imagem).grid(row=row, column=1, padx=5)
        row += 1
        
        # Tipo de relevo
        ttk.Label(parent, text="Tipo de Relevo:", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(10,5))
        row += 1
        
        self.tipo_relevo = tk.StringVar(value="baixo")
        ttk.Radiobutton(parent, text="Baixo Relevo", variable=self.tipo_relevo, value="baixo").grid(row=row, column=0, sticky="w")
        ttk.Radiobutton(parent, text="Alto Relevo", variable=self.tipo_relevo, value="alto").grid(row=row, column=1, sticky="w")
        row += 1
        
        # Parâmetros de usinagem
        params = [
            ("Largura (mm):", "200", "entry_largura"),
            ("Altura (mm):", "150", "entry_altura"),
            ("Profundidade máxima (mm):", "3", "entry_profundidade"),
            ("Passo entre pontos (mm):", "1", "entry_passo"),
            ("Velocidade de avanço (mm/min):", "1000", "entry_feed"),
            ("Safe Z (mm):", "5", "entry_safez")
        ]
        
        for label, default, attr_name in params:
            ttk.Label(parent, text=label, style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(10,5))
            row += 1
            
            entry = ttk.Entry(parent)
            entry.insert(0, default)
            entry.grid(row=row, column=0, pady=5, sticky="ew")
            setattr(self, attr_name, entry)
            row += 1
        
        # Botão gerar
        ttk.Button(parent, text="🎯 GERAR G-CODE OTIMIZADO", 
                  command=self.gerar, style="TButton").grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        # Rodapé
        ttk.Label(parent, text="© 2025 - CNC Router 3D - Versão Otimizada", 
                 font=("Segoe UI", 8), foreground="gray").grid(row=row, column=0, columnspan=2, pady=10)
        
        # Configurar grid
        parent.columnconfigure(0, weight=1)
    
    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
    
    def gerar(self):
        try:
            img_path = self.entry_imagem.get()
            if not os.path.exists(img_path):
                messagebox.showwarning("Aviso", "Selecione uma imagem válida.")
                return

            # Coletar parâmetros
            params = {
                'largura_mm': float(self.entry_largura.get()),
                'altura_mm': float(self.entry_altura.get()),
                'profundidade_max': float(self.entry_profundidade.get()),
                'passo': float(self.entry_passo.get()),
                'feedrate': float(self.entry_feed.get()),
                'safe_z': float(self.entry_safez.get()),
                'tipo_relevo': self.tipo_relevo.get()
            }

            # Validar parâmetros
            if params['passo'] < 0.1:
                messagebox.showerror("Erro", "Passo muito pequeno! Use no mínimo 0.1mm")
                return
                
            if params['profundidade_max'] > 10:
                if not messagebox.askyesno("Confirmação", "Profundidade maior que 10mm. Tem certeza?"):
                    return

            heightmap, gcode = processar_imagem(img_path, **params)

            if gcode:
                messagebox.showinfo("Sucesso", 
                    f"✅ G-code gerado com sucesso!\n\n"
                    f"📁 Pasta: {os.path.dirname(gcode)}\n"
                    f"📊 Arquivos: Heightmap + G-code\n"
                    f"⚡ Estratégia: Corte contínuo otimizado")

        except ValueError as e:
            messagebox.showerror("Erro", f"Valor inválido: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar: {str(e)}")

# ==============================================
# EXECUÇÃO
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GeradorCNC(root)
    root.mainloop()