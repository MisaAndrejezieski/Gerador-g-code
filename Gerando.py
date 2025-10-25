import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==============================================
# CLASSE DE IA MELHORADA - PRESERVA DETALHES ORIGINAIS
# ==============================================

class AICNCProcessor:
    def __init__(self):
        pass
        
    def processar_imagem_inteligente(self, image_array, profundidade_max, tipo_processamento="preservar_detalhes"):
        """
        Processamento inteligente que preserva os detalhes visuais da imagem original
        """
        try:
            if tipo_processamento == "preservar_detalhes":
                return self._processar_preservando_detalhes(image_array, profundidade_max)
            elif tipo_processamento == "relevo_natural":
                return self._processar_relevo_natural(image_array, profundidade_max)
            else:
                return self._processar_tradicional(image_array, profundidade_max)
                
        except Exception as e:
            print(f"Erro no processamento IA: {e}")
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_preservando_detalhes(self, image_array, profundidade_max):
        """
        Técnica avançada que preserva MAXIMAMENTE os detalhes da imagem original
        """
        img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 0.8)
        
        kernel_aguçamento = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        img_detalhada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
        
        z_map = (1 - img_detalhada) * profundidade_max
        gamma = 0.8
        z_map = np.power(z_map / profundidade_max, gamma) * profundidade_max
        
        bordas = cv2.Canny((image_array * 255).astype(np.uint8), 50, 150) / 255.0
        z_map_suavizado = cv2.bilateralFilter(z_map.astype(np.float32), 5, 50, 50)
        
        mascara_bordas = bordas > 0.1
        z_map_suavizado[mascara_bordas] = z_map[mascara_bordas]
        
        return z_map_suavizado

    def _processar_relevo_natural(self, image_array, profundidade_max):
        img_contraste = self._ajustar_contraste_perceptivo(image_array)
        z_map = self._mapeamento_perceptivo(img_contraste, profundidade_max)
        z_map = cv2.bilateralFilter(z_map.astype(np.float32), 7, 30, 30)
        return z_map

    def _processar_tradicional(self, image_array, profundidade_max):
        z_map = (1 - image_array) * profundidade_max
        z_map = cv2.GaussianBlur(z_map, (2, 2), 0.5)
        return z_map

    def _ajustar_contraste_perceptivo(self, image_array):
        return np.power(image_array, 0.7)

    def _mapeamento_perceptivo(self, image_array, profundidade_max):
        media = np.mean(image_array)
        desvio = np.std(image_array)
        
        z_map = np.zeros_like(image_array)
        mascara_escuro = image_array < (media - desvio/2)
        mascara_claro = image_array > (media + desvio/2)
        mascara_medio = ~(mascara_escuro | mascara_claro)
        
        z_map[mascara_escuro] = profundidade_max * 0.9
        z_map[mascara_claro] = profundidade_max * 0.1
        
        if np.any(mascara_medio):
            intensidades_medio = image_array[mascara_medio]
            z_map[mascara_medio] = (1 - intensidades_medio) * profundidade_max * 0.8 + profundidade_max * 0.1
        
        return z_map

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================

def salvar_comparacao_visual(img_original, z_map_final, output_dir):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title('IMAGEM ORIGINAL')
        axes[0].axis('off')
        
        im2 = axes[1].imshow(z_map_final, cmap='viridis')
        axes[1].set_title('MAPA DE PROFUNDIDADE 3D')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        img_rgb = np.stack([img_original] * 3, axis=-1)
        depth_normalized = z_map_final / z_map_final.max() if z_map_final.max() > 0 else z_map_final
        heatmap = plt.cm.viridis(depth_normalized)
        alpha = 0.6
        sobreposicao = img_rgb * (1 - alpha) + heatmap[:, :, :3] * alpha
        
        axes[2].imshow(sobreposicao)
        axes[2].set_title('SOBREPOSIÇÃO')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparacao_visual_detalhada.png'), 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"Erro ao salvar comparação: {e}")

def gerar_gcode_otimizado(z_map, passo, feedrate, safe_z, gcode_path, img_width, img_height):
    try:
        with open(gcode_path, "w") as f:
            f.write("(G-code para Relevo 3D - Gerado com IA)\n")
            f.write("G21 G90 G17 G94 G49\n")
            f.write(f"F{feedrate}\n\n")
            f.write(f"G0 Z{safe_z:.3f}\n")
            f.write("G0 X0 Y0\n\n")
            
            linhas, colunas = z_map.shape
            
            for y in range(linhas):
                if y % 2 == 0:
                    x_range = range(colunas)
                else:
                    x_range = range(colunas - 1, -1, -1)
                
                primeiro_ponto = True
                
                for x in x_range:
                    z = z_map[y, x]
                    pos_x = (x * passo) - (img_width * passo / 2)
                    pos_y = (y * passo) - (img_height * passo / 2)
                    
                    if primeiro_ponto:
                        f.write(f"G0 X{pos_x:.3f} Y{pos_y:.3f}\n")
                        f.write(f"G1 Z{z:.3f}\n")
                        primeiro_ponto = False
                    else:
                        f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z:.3f}\n")
            
            f.write(f"\nG0 Z{safe_z:.3f}\n")
            f.write("G0 X0 Y0\n")
            f.write("M30\n")
        
        return True
    except Exception as e:
        print(f"Erro ao gerar G-code: {e}")
        return False

# ==============================================
# INTERFACE MODERNA E BONITA
# ==============================================

class ModernCNCApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("🎨 CNC Studio Pro - Gerador de Relevo 3D")
        self.root.geometry("900x750")
        self.root.configure(bg='#2C3E50')
        self.root.resizable(True, True)
        
        # Configurar estilo moderno
        self.setup_styles()
        
        # Frame principal
        main_container = tk.Frame(self.root, bg='#2C3E50', padx=20, pady=20)
        main_container.pack(fill='both', expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Conteúdo principal com abas
        self.create_notebook(main_container)
        
        # Footer
        self.create_footer(main_container)
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar cores modernas
        style.configure('Modern.TFrame', background='#34495E')
        style.configure('Header.TLabel', background='#2C3E50', foreground='white', 
                       font=('Segoe UI', 20, 'bold'))
        style.configure('Section.TLabel', background='#34495E', foreground='#ECF0F1',
                       font=('Segoe UI', 12, 'bold'))
        style.configure('Modern.TButton', background='#3498DB', foreground='white',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        style.map('Modern.TButton', background=[('active', '#2980B9')])
        style.configure('Accent.TButton', background='#27AE60', foreground='white',
                       font=('Segoe UI', 12, 'bold'), borderwidth=0)
        style.map('Accent.TButton', background=[('active', '#229954')])
        style.configure('Modern.TCheckbutton', background='#34495E', foreground='#ECF0F1')
        style.configure('Modern.TRadiobutton', background='#34495E', foreground='#ECF0F1')
        style.configure('Modern.TEntry', fieldbackground='#ECF0F1', borderwidth=1)
        
    def create_header(self, parent):
        header_frame = ttk.Frame(parent, style='Modern.TFrame')
        header_frame.pack(fill='x', pady=(0, 20))
        
        title = ttk.Label(header_frame, 
                         text="🎨 CNC STUDIO PRO", 
                         style='Header.TLabel')
        title.pack(pady=10)
        
        subtitle = ttk.Label(header_frame,
                           text="Conversor Inteligente de Imagem para Relevo 3D",
                           style='Section.TLabel')
        subtitle.pack(pady=(0, 10))
        
    def create_notebook(self, parent):
        # Criar abas
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        
        # Aba 1: Configurações Básicas
        tab_basico = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(tab_basico, text="⚙️ Configurações")
        
        self.create_basic_tab(tab_basico)
        
        # Aba 2: Configurações Avançadas
        tab_avancado = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(tab_avancado, text="🔧 Avançado")
        
        self.create_advanced_tab(tab_avancado)
        
    def create_basic_tab(self, parent):
        # Frame com scroll
        canvas = tk.Canvas(parent, bg='#34495E', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Modern.TFrame')
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Conteúdo da aba básica
        content_frame = scrollable_frame
        
        # Seção: Imagem
        self.create_section(content_frame, "📁 Imagem de Entrada", 0)
        self.create_file_selection(content_frame, 1)
        
        # Seção: Processamento
        self.create_section(content_frame, "🔧 Método de Processamento", 2)
        self.create_processing_options(content_frame, 3)
        
        # Seção: Dimensões
        self.create_section(content_frame, "📐 Dimensões do Trabalho", 4)
        self.create_dimension_controls(content_frame, 5)
        
        # Seção: Geração
        self.create_generation_section(content_frame, 6)
        
    def create_advanced_tab(self, parent):
        content_frame = ttk.Frame(parent, style='Modern.TFrame')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Seção: Parâmetros Avançados
        self.create_section(content_frame, "⚙️ Parâmetros de Usinagem", 0)
        
        advanced_params = [
            ("🔍 Passo entre pontos (mm):", "1.0", "entry_passo"),
            ("⚡ Velocidade de avanço (mm/min):", "1200", "entry_feed"),
            ("🛡️ Safe Z (mm):", "5", "entry_safez")
        ]
        
        for i, (label, default, attr_name) in enumerate(advanced_params):
            ttk.Label(content_frame, text=label, style='Section.TLabel').grid(
                row=1+i, column=0, sticky='w', pady=5)
            
            entry = ttk.Entry(content_frame, width=15, style='Modern.TEntry')
            entry.insert(0, default)
            entry.grid(row=1+i, column=1, sticky='w', pady=5, padx=(10,0))
            setattr(self, attr_name, entry)
        
        # Seção: Tipo de Relevo
        self.create_section(content_frame, "🎨 Tipo de Relevo", 4)
        
        self.tipo_relevo = tk.StringVar(value="baixo")
        relief_frame = ttk.Frame(content_frame, style='Modern.TFrame')
        relief_frame.grid(row=5, column=0, columnspan=2, sticky='w', pady=10)
        
        ttk.Radiobutton(relief_frame, text="Baixo Relevo", 
                       variable=self.tipo_relevo, value="baixo",
                       style='Modern.TRadiobutton').pack(side='left', padx=(0, 20))
        ttk.Radiobutton(relief_frame, text="Alto Relevo", 
                       variable=self.tipo_relevo, value="alto",
                       style='Modern.TRadiobutton').pack(side='left')
        
    def create_section(self, parent, title, row):
        section_frame = ttk.Frame(parent, style='Modern.TFrame')
        section_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=15)
        
        ttk.Label(section_frame, text=title, style='Section.TLabel').pack(anchor='w')
        
        # Linha divisória
        separator = ttk.Separator(section_frame, orient='horizontal')
        separator.pack(fill='x', pady=(5, 0))
        
    def create_file_selection(self, parent, row):
        file_frame = ttk.Frame(parent, style='Modern.TFrame')
        file_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        
        self.entry_imagem = ttk.Entry(file_frame, width=50, style='Modern.TEntry')
        self.entry_imagem.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(file_frame, text="Procurar", 
                  command=self.selecionar_imagem,
                  style='Modern.TButton').pack(side='right')
        
    def create_processing_options(self, parent, row):
        processing_frame = ttk.Frame(parent, style='Modern.TFrame')
        processing_frame.grid(row=row, column=0, columnspan=2, sticky='w', pady=10)
        
        self.metodo_processamento = tk.StringVar(value="preservar_detalhes")
        self.uso_ia = tk.BooleanVar(value=True)
        
        # Checkbox IA
        ttk.Checkbutton(processing_frame, 
                       text="Usar processamento inteligente (Recomendado)",
                       variable=self.uso_ia,
                       style='Modern.TCheckbutton').pack(anchor='w', pady=5)
        
        # Métodos de processamento
        methods_frame = ttk.Frame(processing_frame, style='Modern.TFrame')
        methods_frame.pack(anchor='w', pady=10, padx=20)
        
        ttk.Radiobutton(methods_frame, text="🎯 Preservar Detalhes (Alta Qualidade)", 
                       variable=self.metodo_processamento, value="preservar_detalhes",
                       style='Modern.TRadiobutton').pack(anchor='w')
        ttk.Radiobutton(methods_frame, text="🌊 Relevo Natural (Balanço Ideal)", 
                       variable=self.metodo_processamento, value="relevo_natural",
                       style='Modern.TRadiobutton').pack(anchor='w')
        ttk.Radiobutton(methods_frame, text="⚡ Tradicional Rápido (Performance)", 
                       variable=self.metodo_processamento, value="tradicional",
                       style='Modern.TRadiobutton').pack(anchor='w')
        
    def create_dimension_controls(self, parent, row):
        dim_frame = ttk.Frame(parent, style='Modern.TFrame')
        dim_frame.grid(row=row, column=0, columnspan=2, sticky='w', pady=10)
        
        params = [
            ("Largura (mm):", "200", "entry_largura"),
            ("Altura (mm):", "150", "entry_altura"), 
            ("Profundidade máxima (mm):", "3", "entry_profundidade")
        ]
        
        for i, (label, default, attr_name) in enumerate(params):
            ttk.Label(dim_frame, text=label, style='Section.TLabel').grid(
                row=i, column=0, sticky='w', pady=5, padx=(0, 10))
            
            entry = ttk.Entry(dim_frame, width=10, style='Modern.TEntry')
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky='w', pady=5)
            setattr(self, attr_name, entry)
        
    def create_generation_section(self, parent, row):
        gen_frame = ttk.Frame(parent, style='Modern.TFrame')
        gen_frame.grid(row=row, column=0, columnspan=2, pady=30)
        
        ttk.Button(gen_frame, text="🚀 GERAR RELEVO 3D", 
                  command=self.gerar,
                  style='Accent.TButton',
                  width=25).pack(pady=20)
        
        # Status
        self.status_label = ttk.Label(gen_frame, 
                                     text="Pronto para processar",
                                     style='Section.TLabel')
        self.status_label.pack()
        
    def create_footer(self, parent):
        footer_frame = ttk.Frame(parent, style='Modern.TFrame')
        footer_frame.pack(fill='x', pady=(20, 0))
        
        ttk.Label(footer_frame, 
                 text="© 2025 CNC Studio Pro | Fidelidade Visual Garantida 🎨",
                 style='Section.TLabel').pack()
        
    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            self.status_label.config(text=f"Imagem carregada: {os.path.basename(caminho)}")
            
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
                'tipo_relevo': self.tipo_relevo.get(),
                'metodo_processamento': self.metodo_processamento.get()
            }

            self.status_label.config(text="Processando imagem...")
            self.root.update()
            
            # Processar imagem (usar a função processar_imagem_ia do código anterior)
            gcode_path, output_dir = self.processar_imagem_ia(img_path, **params)

            if gcode_path and output_dir:
                self.status_label.config(text="Processamento concluído!")
                messagebox.showinfo("Sucesso!", 
                    f"✅ Relevo 3D gerado com sucesso!\n\n"
                    f"📁 Pasta: {output_dir}\n"
                    f"📊 Análise visual gerada\n"
                    f"⚡ G-code pronto para usinagem")
            else:
                self.status_label.config(text="Erro no processamento")
                
        except Exception as e:
            self.status_label.config(text="Erro no processamento")
            messagebox.showerror("Erro", f"Falha: {str(e)}")

    def processar_imagem_ia(self, img_path, **params):
        """Função de processamento (adaptada do código anterior)"""
        try:
            output_dir = os.path.join(os.getcwd(), "CNC_Output")
            os.makedirs(output_dir, exist_ok=True)

            img = Image.open(img_path).convert("L")
            img_array_original = np.array(img) / 255.0
            
            img_ratio = img.width / img.height
            target_ratio = params['largura_mm'] / params['altura_mm']
            
            if img_ratio > target_ratio:
                new_width = int(params['largura_mm'] / params['passo'])
                new_height = int(new_width / img_ratio)
            else:
                new_height = int(params['altura_mm'] / params['passo'])
                new_width = int(new_height * img_ratio)
            
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            img_array = np.array(img_resized) / 255.0

            ai_processor = AICNCProcessor()
            
            if params['uso_ia']:
                z_map = ai_processor.processar_imagem_inteligente(
                    img_array, params['profundidade_max'], params['metodo_processamento']
                )
            else:
                if params['tipo_relevo'] == "baixo":
                    z_map = (1 - img_array) * params['profundidade_max']
                else:
                    z_map = img_array * params['profundidade_max']
            
            z_map = np.clip(z_map, 0, params['profundidade_max'])
            salvar_comparacao_visual(img_array, z_map, output_dir)

            gcode_path = os.path.join(output_dir, "relevo_3d.nc")
            success = gerar_gcode_otimizado(z_map, params['passo'], params['feedrate'], 
                                          params['safe_z'], gcode_path, new_width, new_height)

            return gcode_path, output_dir if success else (None, None)

        except Exception as e:
            print(f"Erro: {str(e)}")
            return None, None

# ==============================================
# EXECUÇÃO
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernCNCApp(root)
    root.mainloop()
    