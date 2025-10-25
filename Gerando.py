import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==============================================
# CLASSE DE IA SIMPLIFICADA (SEM scikit-learn)
# ==============================================

class AICNCProcessor:
    def __init__(self):
        pass
        
    def predict_relevo_map(self, image_array):
        """
        Prediz mapa de relevos usando OpenCV (sem scikit-learn)
        """
        try:
            # Usar threshold adaptativo para segmentação
            img_uint8 = (image_array * 255).astype(np.uint8)
            
            # Aplicar threshold adaptativo
            binary = cv2.adaptiveThreshold(
                img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Criar máscara baseada em contornos
            relevo_map = np.zeros_like(image_array)
            
            # Áreas externas (contornos) = Alto relevo
            contour_mask = np.zeros_like(image_array)
            cv2.drawContours(contour_mask, contours, -1, 1, thickness=cv2.FILLED)
            
            # Áreas internas claras = Baixo relevo, escuras = Plano
            mean_intensity = np.mean(image_array)
            
            relevo_map[contour_mask == 1] = 2  # Alto relevo nas bordas
            relevo_map[(contour_mask == 0) & (image_array > mean_intensity)] = 0  # Baixo relevo
            relevo_map[(contour_mask == 0) & (image_array <= mean_intensity)] = 1  # Plano
            
            return relevo_map
        except Exception as e:
            print(f"Erro na predição IA: {e}")
            # Fallback: usar método simples baseado em intensidade
            relevo_map = np.zeros_like(image_array)
            relevo_map[image_array > 0.7] = 0  # Baixo relevo (áreas claras)
            relevo_map[image_array < 0.3] = 2  # Alto relevo (áreas escuras)
            relevo_map[(image_array >= 0.3) & (image_array <= 0.7)] = 1  # Plano
            return relevo_map

    def generate_adaptive_height_map(self, image_array, relevo_map, profundidade_max):
        """
        Gera mapa de alturas adaptativo
        """
        try:
            height_map = np.zeros_like(image_array, dtype=np.float32)
            
            # Parâmetros de profundidade por tipo de relevo
            depths = {
                0: {'min': 0.3, 'max': 1.5},   # Baixo relevo
                1: {'min': 0.1, 'max': 0.8},   # Plano
                2: {'min': 1.5, 'max': 3.0}    # Alto relevo
            }
            
            scale_factor = profundidade_max / 3.0
            
            for relevo_type in [0, 1, 2]:
                mask = relevo_map == relevo_type
                if np.any(mask):
                    region_intensity = image_array[mask]
                    
                    depth_range = depths[relevo_type]
                    if region_intensity.max() > region_intensity.min():
                        normalized_intensity = (region_intensity - region_intensity.min()) / \
                                             (region_intensity.max() - region_intensity.min())
                    else:
                        normalized_intensity = np.ones_like(region_intensity) * 0.5
                    
                    region_depth = depth_range['min'] + normalized_intensity * \
                                 (depth_range['max'] - depth_range['min'])
                    
                    region_depth = region_depth * scale_factor
                    height_map[mask] = region_depth
            
            # Suavizar
            height_map = cv2.GaussianBlur(height_map, (3, 3), 0.3)
            
            return height_map
        except Exception as e:
            print(f"Erro na geração do height map: {e}")
            # Fallback: método tradicional
            return (1 - image_array) * profundidade_max

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================

def salvar_analise_ia(img_array, relevo_map, height_map, output_dir):
    """
    Salva visualizações da análise da IA
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        axes[0,0].imshow(img_array, cmap='gray')
        axes[0,0].set_title('Imagem Original Processada')
        
        # Mapa de Relevos
        relevo_rgb = np.zeros((*relevo_map.shape, 3))
        colors = [[1,1,0], [0.5,0.5,1], [1,0,0]]  # amarelo, azul, vermelho
        for i, color in enumerate(colors):
            mask = relevo_map == i
            relevo_rgb[mask] = color
        
        axes[0,1].imshow(relevo_rgb)
        axes[0,1].set_title('Mapa de Relevos (IA)\nAmarelo=Baixo, Azul=Plano, Vermelho=Alto')
        
        # Mapa de Alturas
        im3 = axes[1,0].imshow(height_map, cmap='viridis')
        axes[1,0].set_title('Mapa de Alturas Final')
        plt.colorbar(im3, ax=axes[1,0])
        
        # Histograma de distribuição
        axes[1,1].hist(height_map.flatten(), bins=50, alpha=0.7, color='green')
        axes[1,1].set_title('Distribuição de Profundidades')
        axes[1,1].set_xlabel('Profundidade (mm)')
        axes[1,1].set_ylabel('Frequência')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analise_ia_detalhada.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Análise da IA salva com sucesso!")
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível salvar análise da IA: {e}")

def salvar_imagens_processo(img_array, z_map, output_dir, uso_ia):
    """
    Salva imagens do processo para verificação
    """
    try:
        # Heightmap para visualização
        if z_map.max() > 0:
            heightmap_vis = (z_map / z_map.max() * 255).astype(np.uint8)
        else:
            heightmap_vis = np.zeros_like(z_map, dtype=np.uint8)
            
        heightmap_img = Image.fromarray(heightmap_vis)
        heightmap_img.save(os.path.join(output_dir, "heightmap_final.png"))
        
        # Imagem original processada
        original_vis = (img_array * 255).astype(np.uint8)
        original_img = Image.fromarray(original_vis)
        original_img.save(os.path.join(output_dir, "original_processada.png"))
        
        print("✅ Imagens de processo salvas!")
    except Exception as e:
        print(f"⚠️ Aviso: Erro ao salvar imagens: {e}")

def gerar_gcode_otimizado(z_map, passo, feedrate, safe_z, gcode_path, img_width, img_height):
    """
    Gera G-code otimizado com movimento contínuo
    """
    try:
        with open(gcode_path, "w") as f:
            # CABEÇALHO
            f.write("(G-code para Relevo 3D - Gerado com IA)\n")
            f.write("G21 G90 G17 G94 G49\n")
            f.write(f"F{feedrate}\n\n")
            
            # POSICIONAMENTO INICIAL
            f.write(f"G0 Z{safe_z:.3f}\n")
            f.write("G0 X0 Y0\n\n")
            
            linhas, colunas = z_map.shape
            f.write(f"; Dimensões: {colunas}x{linhas} pontos\n")
            f.write(f"; Área usinagem: {img_width * passo:.1f}x{img_height * passo:.1f}mm\n\n")
            
            # ESTRATÉGIA DE CORTE INTELIGENTE
            for y in range(linhas):
                # Direção zig-zag
                if y % 2 == 0:
                    x_range = range(colunas)
                else:
                    x_range = range(colunas - 1, -1, -1)
                
                primeiro_ponto = True
                
                for x in x_range:
                    z = z_map[y, x]
                    pos_x = (x * passo) - (img_width * passo / 2)  # Centralizado
                    pos_y = (y * passo) - (img_height * passo / 2) # Centralizado
                    
                    if primeiro_ponto:
                        # Move para primeiro ponto com Safe Z
                        f.write(f"G0 X{pos_x:.3f} Y{pos_y:.3f}\n")
                        f.write(f"G1 Z{z:.3f}\n")
                        primeiro_ponto = False
                    else:
                        # Movimento de corte contínuo
                        f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z:.3f}\n")
            
            # FINALIZAÇÃO
            f.write(f"\nG0 Z{safe_z:.3f}\n")
            f.write("G0 X0 Y0\n")
            f.write("M30\n")
        
        print("✅ G-code gerado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao gerar G-code: {e}")
        return False

# ==============================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# ==============================================

def processar_imagem_ia(img_path, largura_mm, altura_mm, profundidade_max, passo, feedrate, safe_z, uso_ia=True, tipo_relevo="baixo"):
    """
    Função principal de processamento com IA
    """
    try:
        # Criar diretório de saída
        output_dir = os.path.join(os.getcwd(), "Imagens_Processadas_IA")
        os.makedirs(output_dir, exist_ok=True)

        # Abrir e tratar imagem
        print("📁 Carregando imagem...")
        img = Image.open(img_path).convert("L")
        
        # PRÉ-PROCESSAMENTO MELHORADO
        print("🔄 Processando imagem...")
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = img.filter(ImageFilter.SHARPEN)
        
        # Converter para array numpy
        img_array_original = np.array(img) / 255.0
        
        # CORREÇÃO: Respeitar proporção original da imagem
        img_ratio = img.width / img.height
        target_ratio = largura_mm / altura_mm
        
        if img_ratio > target_ratio:
            new_width = int(largura_mm / passo)
            new_height = int(new_width / img_ratio)
        else:
            new_height = int(altura_mm / passo)
            new_width = int(new_height * img_ratio)
        
        # Redimensionar mantendo proporção
        print(f"📐 Redimensionando para {new_width}x{new_height}...")
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0

        # PROCESSAMENTO COM IA OU TRADICIONAL
        if uso_ia:
            print("🤖 Processando com IA...")
            # USAR IA PARA MAPEAMENTO INTELIGENTE
            ai_processor = AICNCProcessor()
            relevo_map = ai_processor.predict_relevo_map(img_array)
            z_map = ai_processor.generate_adaptive_height_map(img_array, relevo_map, profundidade_max)
            
            # Salvar análise da IA
            salvar_analise_ia(img_array, relevo_map, z_map, output_dir)
        else:
            print("🔧 Processando método tradicional...")
            # PROCESSAMENTO TRADICIONAL (fallback)
            if tipo_relevo == "baixo":
                z_map = (1 - img_array) * profundidade_max
            else:  # alto relevo
                z_map = img_array * profundidade_max
            
            # Suavizar
            kernel = np.ones((2,2), np.float32)/4
            z_map = cv2.filter2D(z_map, -1, kernel)

        # Garantir que não ultrapasse a profundidade máxima
        z_map = np.clip(z_map, 0, profundidade_max)

        # Salvar imagens de processo
        salvar_imagens_processo(img_array, z_map, output_dir, uso_ia)

        # Gerar G-code otimizado
        print("⚡ Gerando G-code...")
        gcode_path = os.path.join(output_dir, "relevo_3d_ia.nc")
        success = gerar_gcode_otimizado(z_map, passo, feedrate, safe_z, gcode_path, new_width, new_height)

        if success:
            print("🎉 Processamento concluído com sucesso!")
            return gcode_path, output_dir
        else:
            return None, None

    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")
        messagebox.showerror("Erro", f"Erro no processamento: {str(e)}")
        return None, None

# ==============================================
# INTERFACE GRÁFICA CORRIGIDA
# ==============================================

class GeradorCNCIA:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("Gerador de G-code CNC com IA")
        self.root.geometry("700x700")  # Aumentei a altura para caber tudo
        self.root.resizable(True, True)
        
        # Configurar estilo
        self.setup_styles()
        
        # Frame principal com scroll
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Canvas e Scrollbar para conteúdo rolável
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Título
        title = ttk.Label(scrollable_frame, text="🛠 CNC ROUTER COM INTELIGÊNCIA ARTIFICIAL", 
                         font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Criar widgets
        self.create_widgets(scrollable_frame)
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Checkbox.TCheckbutton", font=("Segoe UI", 10))
        style.configure("Generate.TButton", font=("Segoe UI", 12, "bold"), background="#4CAF50")
        
    def create_widgets(self, parent):
        row = 1
        
        # Imagem de entrada
        ttk.Label(parent, text="📁 Imagem de Entrada:", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(10,5))
        row += 1
        
        frame_imagem = ttk.Frame(parent)
        frame_imagem.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.entry_imagem = ttk.Entry(frame_imagem, width=60)
        self.entry_imagem.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(frame_imagem, text="Procurar", command=self.selecionar_imagem).pack(side="right")
        row += 1
        
        # Configurações de IA
        ttk.Label(parent, text="🤖 Configurações de IA:", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(20,10))
        row += 1
        
        frame_ia = ttk.Frame(parent)
        frame_ia.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.uso_ia = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_ia, text="Usar Inteligência Artificial para análise de relevos", 
                       variable=self.uso_ia, style="Checkbox.TCheckbutton").pack(side="left")
        
        ttk.Label(frame_ia, text="(Recomendado)", 
                 font=("Segoe UI", 8), foreground="gray").pack(side="left", padx=(10,0))
        row += 1
        
        # Tipo de relevo
        ttk.Label(parent, text="🎨 Tipo de Relevo (se IA desativada):", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(10,5))
        row += 1
        
        frame_relevo = ttk.Frame(parent)
        frame_relevo.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        
        self.tipo_relevo = tk.StringVar(value="baixo")
        ttk.Radiobutton(frame_relevo, text="Baixo Relevo", variable=self.tipo_relevo, value="baixo").pack(side="left", padx=(0,20))
        ttk.Radiobutton(frame_relevo, text="Alto Relevo", variable=self.tipo_relevo, value="alto").pack(side="left")
        row += 1
        
        # Parâmetros de usinagem
        params = [
            ("📏 Largura (mm):", "200", "entry_largura"),
            ("📐 Altura (mm):", "150", "entry_altura"),
            ("⏬ Profundidade máxima (mm):", "4", "entry_profundidade"),
            ("🔍 Passo entre pontos (mm):", "1.0", "entry_passo"),
            ("⚡ Velocidade de avanço (mm/min):", "1200", "entry_feed"),
            ("🛡️ Safe Z (mm):", "5", "entry_safez")
        ]
        
        for label, default, attr_name in params:
            ttk.Label(parent, text=label, style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(15,5))
            row += 1
            
            entry = ttk.Entry(parent, width=25, font=("Segoe UI", 10))
            entry.insert(0, default)
            entry.grid(row=row, column=0, sticky="w", pady=2, padx=(20,0))
            setattr(self, attr_name, entry)
            row += 1
        
        # BOTÃO GERAR - AGORA VISÍVEL
        ttk.Label(parent, text="", style="Title.TLabel").grid(row=row, column=0, pady=(20,0))
        row += 1
        
        btn_gerar = ttk.Button(parent, text="🚀 GERAR G-CODE", 
                              command=self.gerar, 
                              style="Generate.TButton",
                              width=20)
        btn_gerar.grid(row=row, column=0, columnspan=2, pady=30)
        row += 1
        
        # Rodapé
        rodape = ttk.Label(parent, 
                          text="© 2025 - CNC Router IA - Análise inteligente de relevos | Versão 2.0", 
                          font=("Segoe UI", 8), foreground="gray")
        rodape.grid(row=row, column=0, columnspan=2, pady=20)
        
        # Configurar grid
        parent.columnconfigure(0, weight=1)
    
    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para processamento",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            messagebox.showinfo("Imagem Selecionada", f"Imagem carregada:\n{os.path.basename(caminho)}")
            
    def gerar(self):
        try:
            img_path = self.entry_imagem.get()
            if not img_path or not os.path.exists(img_path):
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
                'uso_ia': self.uso_ia.get(),
                'tipo_relevo': self.tipo_relevo.get()
            }

            # Validações
            if params['profundidade_max'] <= 0:
                messagebox.showerror("Erro", "Profundidade deve ser maior que zero!")
                return
                
            if params['passo'] < 0.1:
                if not messagebox.askyesno("Confirmação", "Passo muito pequeno pode gerar arquivos enormes. Continuar?"):
                    return

            # Mostrar mensagem de processamento
            messagebox.showinfo("Processando", "Iniciando processamento da imagem...\nIsso pode levar alguns minutos.")
            
            # Processar imagem
            gcode_path, output_dir = processar_imagem_ia(img_path, **params)

            if gcode_path and output_dir:
                messagebox.showinfo("Sucesso!", 
                    f"✅ Processamento concluído!\n\n"
                    f"🤖 IA: {'ATIVADA' if params['uso_ia'] else 'Desativada'}\n"
                    f"📁 Pasta: {output_dir}\n"
                    f"📊 Análise: analise_ia_detalhada.png\n"
                    f"⚡ G-code: {os.path.basename(gcode_path)}\n\n"
                    f"Verifique a análise visual gerada antes de usinar!")

        except ValueError as e:
            messagebox.showerror("Erro", f"Valor inválido nos parâmetros: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha no processamento: {str(e)}")

# ==============================================
# EXECUÇÃO
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GeradorCNCIA(root)
    root.mainloop()
    