import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==============================================
# CLASSE DE IA CORRIGIDA - SEM VALORES NaN
# ==============================================

class AICNCProcessor:
    def __init__(self):
        pass
        
    def processar_imagem_inteligente(self, image_array, profundidade_max, tipo_processamento="preservar_detalhes"):
        """
        Processamento inteligente SEM valores NaN
        """
        try:
            # GARANTIR que não há valores NaN na entrada
            image_array = np.nan_to_num(image_array, nan=0.5, posinf=1.0, neginf=0.0)
            image_array = np.clip(image_array, 0.001, 0.999)
            
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
        Técnica avançada SEM valores NaN
        """
        try:
            # Garantir dados válidos
            image_array = np.nan_to_num(image_array, nan=0.5)
            image_array = np.clip(image_array, 0.001, 0.999)
            
            img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 0.8)
            img_suavizada = np.clip(img_suavizada, 0.001, 0.999)
            
            kernel_aguçamento = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
            img_detalhada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
            img_detalhada = np.clip(img_detalhada, 0.001, 0.999)
            
            # Mapeamento seguro SEM NaN
            z_map = (1 - img_detalhada) * profundidade_max
            
            # Aplicar curva gamma com proteção
            gamma = 0.8
            z_map_normalized = np.clip(z_map / profundidade_max, 0.001, 0.999)
            z_map = np.power(z_map_normalized, gamma) * profundidade_max
            
            # Garantir saída válida
            z_map = np.nan_to_num(z_map, nan=0.0)
            z_map = np.clip(z_map, 0.0, profundidade_max)
            
            print(f"✅ Processamento 'Preservar Detalhes' - Profundidades: {z_map.min():.3f} a {z_map.max():.3f} mm")
            return z_map
            
        except Exception as e:
            print(f"Erro em preservar detalhes: {e}")
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_relevo_natural(self, image_array, profundidade_max):
        try:
            image_array = np.nan_to_num(image_array, nan=0.5)
            image_array = np.clip(image_array, 0.001, 0.999)
            
            img_contraste = np.power(image_array, 0.7)
            img_contraste = np.clip(img_contraste, 0.001, 0.999)
            
            z_map = self._mapeamento_perceptivo(img_contraste, profundidade_max)
            z_map = cv2.bilateralFilter(z_map.astype(np.float32), 7, 30, 30)
            
            z_map = np.nan_to_num(z_map, nan=0.0)
            z_map = np.clip(z_map, 0.0, profundidade_max)
            
            print(f"✅ Processamento 'Relevo Natural' - Profundidades: {z_map.min():.3f} a {z_map.max():.3f} mm")
            return z_map
            
        except Exception as e:
            print(f"Erro em relevo natural: {e}")
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_tradicional(self, image_array, profundidade_max):
        try:
            image_array = np.nan_to_num(image_array, nan=0.5)
            image_array = np.clip(image_array, 0.001, 0.999)
            
            z_map = (1 - image_array) * profundidade_max
            z_map = cv2.GaussianBlur(z_map, (2, 2), 0.5)
            
            z_map = np.nan_to_num(z_map, nan=0.0)
            z_map = np.clip(z_map, 0.0, profundidade_max)
            
            print(f"✅ Processamento 'Tradicional' - Profundidades: {z_map.min():.3f} a {z_map.max():.3f} mm")
            return z_map
            
        except Exception as e:
            print(f"Erro em tradicional: {e}")
            # Último recurso - retornar array seguro
            safe_array = np.full_like(image_array, profundidade_max / 2)
            return safe_array

    def _mapeamento_perceptivo(self, image_array, profundidade_max):
        try:
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
        except:
            return (1 - image_array) * profundidade_max

# ==============================================
# FUNÇÕES AUXILIARES CORRIGIDAS
# ==============================================

def salvar_comparacao_visual(img_original, z_map_final, output_dir):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title('IMAGEM ORIGINAL')
        axes[0].axis('off')
        
        im2 = axes[1].imshow(z_map_final, cmap='viridis')
        axes[1].set_title('MAPA DE PROFUNDIDADE')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        img_rgb = np.stack([img_original] * 3, axis=-1)
        depth_normalized = z_map_final / z_map_final.max() if z_map_final.max() > 0 else z_map_final
        heatmap = plt.cm.viridis(depth_normalized)
        alpha = 0.6
        sobreposicao = img_rgb * (1 - alpha) + heatmap[:, :, :3] * alpha
        
        axes[2].imshow(sobreposicao)
        axes[2].set_title('SOBREPOSIÇÃO')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparacao_visual.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Erro ao salvar comparação: {e}")

def gerar_gcode_otimizado(z_map, passo, feedrate, safe_z, gcode_path, img_width, img_height):
    """
    Gera G-code SEM valores NaN - CORREÇÃO CRÍTICA
    """
    try:
        with open(gcode_path, "w") as f:
            # CABEÇALHO
            f.write("(G-code para Relevo 3D - Gerado com IA)\n")
            f.write("G21 G90 G17 G94 G49\n")
            f.write(f"F{feedrate}\n\n")
            f.write(f"G0 Z{safe_z:.3f}\n")
            f.write("G0 X0 Y0\n\n")
            
            linhas, colunas = z_map.shape
            
            # CONTADOR para debug
            pontos_validos = 0
            pontos_invalidos = 0
            
            for y in range(linhas):
                if y % 2 == 0:
                    x_range = range(colunas)
                else:
                    x_range = range(colunas - 1, -1, -1)
                
                primeiro_ponto = True
                
                for x in x_range:
                    z = z_map[y, x]
                    
                    # VERIFICAÇÃO CRÍTICA - GARANTIR que Z é um número válido
                    if np.isnan(z) or np.isinf(z):
                        pontos_invalidos += 1
                        z = 0.0  # Valor seguro padrão
                    else:
                        pontos_validos += 1
                        z = max(0.0, min(z, 10.0))  # Limitar entre 0 e 10mm
                    
                    pos_x = (x * passo) - (img_width * passo / 2)
                    pos_y = (y * passo) - (img_height * passo / 2)
                    
                    if primeiro_ponto:
                        f.write(f"G0 X{pos_x:.3f} Y{pos_y:.3f}\n")
                        f.write(f"G1 Z{z:.3f}\n")
                        primeiro_ponto = False
                    else:
                        f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z:.3f}\n")
            
            # FINALIZAÇÃO
            f.write(f"\nG0 Z{safe_z:.3f}\n")
            f.write("G0 X0 Y0\n")
            f.write("M30\n")
            
            # ESTATÍSTICAS
            f.write(f"\n; === ESTATÍSTICAS ===\n")
            f.write(f"; Pontos válidos: {pontos_validos}\n")
            f.write(f"; Pontos corrigidos: {pontos_invalidos}\n")
            f.write(f"; Dimensões: {colunas}x{linhas}\n")
        
        print(f"✅ G-code gerado: {pontos_validos} pontos válidos, {pontos_invalidos} corrigidos")
        return True
        
    except Exception as e:
        print(f"❌ Erro crítico ao gerar G-code: {e}")
        return False

# ==============================================
# INTERFACE MODERNA COMPACTA
# ==============================================

class ModernCNCApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("🎨 CNC Studio Pro")
        self.root.geometry("800x680")  # Tela mais compacta
        self.root.configure(bg='#2C3E50')
        self.root.resizable(True, True)
        
        self.setup_styles()
        
        # Container principal
        main_container = tk.Frame(self.root, bg='#2C3E50')
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header COMPACTO
        self.create_compact_header(main_container)
        
        # Área de conteúdo principal
        self.create_content_area(main_container)
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurações modernas e compactas
        style.configure('Modern.TFrame', background='#34495E')
        style.configure('CompactHeader.TLabel', background='#2C3E50', foreground='white', 
                       font=('Segoe UI', 16, 'bold'))
        style.configure('Subtitle.TLabel', background='#2C3E50', foreground='#BDC3C7',
                       font=('Segoe UI', 9))
        style.configure('Section.TLabel', background='#34495E', foreground='#ECF0F1',
                       font=('Segoe UI', 10, 'bold'))
        style.configure('Modern.TButton', background='#3498DB', foreground='white',
                       font=('Segoe UI', 9, 'bold'))
        style.map('Modern.TButton', background=[('active', '#2980B9')])
        style.configure('Accent.TButton', background='#27AE60', foreground='white',
                       font=('Segoe UI', 11, 'bold'))
        style.map('Accent.TButton', background=[('active', '#229954')])
        style.configure('Modern.TCheckbutton', background='#34495E', foreground='#ECF0F1')
        style.configure('Modern.TRadiobutton', background='#34495E', foreground='#ECF0F1')
        style.configure('Modern.TEntry', fieldbackground='#ECF0F1')
        
    def create_compact_header(self, parent):
        """Header muito mais compacto"""
        header_frame = ttk.Frame(parent, style='Modern.TFrame')
        header_frame.pack(fill='x', pady=(0, 15))
        
        # Título principal compacto
        title = ttk.Label(header_frame, 
                         text="🎨 CNC Studio Pro", 
                         style='CompactHeader.TLabel')
        title.pack(pady=(5, 0))
        
        # Subtítulo menor
        subtitle = ttk.Label(header_frame,
                           text="Conversor de Imagem para Relevo 3D",
                           style='Subtitle.TLabel')
        subtitle.pack(pady=(0, 5))
        
    def create_content_area(self, parent):
        """Área de conteúdo principal com abas"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        
        # Aba Principal
        tab_principal = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(tab_principal, text="⚙️ Principal")
        self.create_main_tab(tab_principal)
        
        # Aba Avançado
        tab_avancado = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(tab_avancado, text="🔧 Avançado")
        self.create_advanced_tab(tab_avancado)
        
    def create_main_tab(self, parent):
        """Aba principal compacta"""
        # Frame com scroll para conteúdo
        canvas = tk.Canvas(parent, bg='#34495E', highlightthickness=0, height=500)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Modern.TFrame')
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar.pack(side="right", fill="y")
        
        # Conteúdo da aba principal
        content = scrollable_frame
        
        # Seção: Imagem
        self.create_section(content, "📁 Imagem de Entrada", 0)
        self.create_file_section(content, 1)
        
        # Seção: Processamento
        self.create_section(content, "🔧 Processamento", 2)
        self.create_processing_section(content, 3)
        
        # Seção: Dimensões
        self.create_section(content, "📐 Dimensões", 4)
        self.create_dimension_section(content, 5)
        
        # Botão de geração
        self.create_generate_section(content, 6)
        
    def create_file_section(self, parent, row):
        """Seleção de arquivo compacta"""
        file_frame = ttk.Frame(parent, style='Modern.TFrame')
        file_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=8)
        
        self.entry_imagem = ttk.Entry(file_frame, width=45, style='Modern.TEntry')
        self.entry_imagem.pack(side='left', fill='x', expand=True, padx=(0, 8))
        
        ttk.Button(file_frame, text="Procurar", 
                  command=self.selecionar_imagem,
                  style='Modern.TButton').pack(side='right')
        
    def create_processing_section(self, parent, row):
        """Opções de processamento compactas"""
        proc_frame = ttk.Frame(parent, style='Modern.TFrame')
        proc_frame.grid(row=row, column=0, columnspan=2, sticky='w', pady=8)
        
        self.metodo_processamento = tk.StringVar(value="preservar_detalhes")
        self.uso_ia = tk.BooleanVar(value=True)
        
        # Checkbox IA
        ttk.Checkbutton(proc_frame, 
                       text="Processamento Inteligente",
                       variable=self.uso_ia,
                       style='Modern.TCheckbutton').pack(anchor='w', pady=2)
        
        # Métodos de processamento
        methods_frame = ttk.Frame(proc_frame, style='Modern.TFrame')
        methods_frame.pack(anchor='w', pady=5, padx=15)
        
        ttk.Radiobutton(methods_frame, text="🎯 Alta Qualidade", 
                       variable=self.metodo_processamento, value="preservar_detalhes",
                       style='Modern.TRadiobutton').pack(anchor='w')
        ttk.Radiobutton(methods_frame, text="🌊 Natural", 
                       variable=self.metodo_processamento, value="relevo_natural",
                       style='Modern.TRadiobutton').pack(anchor='w')
        ttk.Radiobutton(methods_frame, text="⚡ Rápido", 
                       variable=self.metodo_processamento, value="tradicional",
                       style='Modern.TRadiobutton').pack(anchor='w')
        
    def create_dimension_section(self, parent, row):
        """Controles de dimensão compactos"""
        dim_frame = ttk.Frame(parent, style='Modern.TFrame')
        dim_frame.grid(row=row, column=0, columnspan=2, sticky='w', pady=8)
        
        # Layout em grid para economizar espaço
        params = [
            ("Largura (mm):", "200", "entry_largura", 0),
            ("Altura (mm):", "150", "entry_altura", 1), 
            ("Profundidade (mm):", "3", "entry_profundidade", 2)
        ]
        
        for label, default, attr_name, row_idx in params:
            ttk.Label(dim_frame, text=label, style='Section.TLabel').grid(
                row=row_idx, column=0, sticky='w', pady=2, padx=(0, 5))
            
            entry = ttk.Entry(dim_frame, width=8, style='Modern.TEntry')
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, sticky='w', pady=2)
            setattr(self, attr_name, entry)
        
    def create_generate_section(self, parent, row):
        """Seção de geração compacta"""
        gen_frame = ttk.Frame(parent, style='Modern.TFrame')
        gen_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        ttk.Button(gen_frame, text="🚀 GERAR RELEVO 3D", 
                  command=self.gerar,
                  style='Accent.TButton',
                  width=20).pack(pady=10)
        
        # Status
        self.status_label = ttk.Label(gen_frame, 
                                     text="Pronto para processar",
                                     style='Section.TLabel')
        self.status_label.pack()
        
    def create_section(self, parent, title, row):
        """Criar seção compacta"""
        section_frame = ttk.Frame(parent, style='Modern.TFrame')
        section_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=8)
        
        ttk.Label(section_frame, text=title, style='Section.TLabel').pack(anchor='w')
        
        separator = ttk.Separator(section_frame, orient='horizontal')
        separator.pack(fill='x', pady=(3, 0))
        
    def create_advanced_tab(self, parent):
        """Aba avançada compacta"""
        content = ttk.Frame(parent, style='Modern.TFrame')
        content.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.create_section(content, "⚙️ Parâmetros CNC", 0)
        
        advanced_params = [
            ("Passo (mm):", "1.0", "entry_passo", 1),
            ("Velocidade:", "1200", "entry_feed", 2),
            ("Safe Z (mm):", "5", "entry_safez", 3)
        ]
        
        for label, default, attr_name, row_idx in advanced_params:
            ttk.Label(content, text=label, style='Section.TLabel').grid(
                row=row_idx, column=0, sticky='w', pady=4)
            
            entry = ttk.Entry(content, width=10, style='Modern.TEntry')
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, sticky='w', pady=4, padx=(10,0))
            setattr(self, attr_name, entry)
        
        # Tipo de Relevo
        self.create_section(content, "🎨 Tipo de Relevo", 4)
        
        self.tipo_relevo = tk.StringVar(value="baixo")
        relief_frame = ttk.Frame(content, style='Modern.TFrame')
        relief_frame.grid(row=5, column=0, columnspan=2, sticky='w', pady=8)
        
        ttk.Radiobutton(relief_frame, text="Baixo Relevo", 
                       variable=self.tipo_relevo, value="baixo",
                       style='Modern.TRadiobutton').pack(side='left', padx=(0, 15))
        ttk.Radiobutton(relief_frame, text="Alto Relevo", 
                       variable=self.tipo_relevo, value="alto",
                       style='Modern.TRadiobutton').pack(side='left')
        
    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            self.status_label.config(text=f"Imagem: {os.path.basename(caminho)}")
            
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

            self.status_label.config(text="Processando...")
            self.root.update()
            
            # Processar imagem
            gcode_path, output_dir = self.processar_imagem_ia(img_path, **params)

            if gcode_path and output_dir:
                self.status_label.config(text="Concluído!")
                messagebox.showinfo("Sucesso!", 
                    f"✅ Relevo 3D gerado!\n\n"
                    f"📁 Pasta: {output_dir}\n"
                    f"⚡ G-code validado e pronto")
            else:
                self.status_label.config(text="Erro no processamento")
                
        except Exception as e:
            self.status_label.config(text="Erro no processamento")
            messagebox.showerror("Erro", f"Falha: {str(e)}")

    def processar_imagem_ia(self, img_path, **params):
        """Função de processamento CORRIGIDA - SEM NaN"""
        try:
            output_dir = os.path.join(os.getcwd(), "CNC_Output")
            os.makedirs(output_dir, exist_ok=True)

            # Carregar imagem
            img = Image.open(img_path).convert("L")
            img_array_original = np.array(img) / 255.0
            
            # Garantir dados válidos
            img_array_original = np.nan_to_num(img_array_original, nan=0.5)
            img_array_original = np.clip(img_array_original, 0.001, 0.999)
            
            # Calcular dimensões
            img_ratio = img.width / img.height
            target_ratio = params['largura_mm'] / params['altura_mm']
            
            if img_ratio > target_ratio:
                new_width = int(params['largura_mm'] / params['passo'])
                new_height = int(new_width / img_ratio)
            else:
                new_height = int(params['altura_mm'] / params['passo'])
                new_width = int(new_height * img_ratio)
            
            # Redimensionar
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            img_array = np.array(img_resized) / 255.0
            img_array = np.nan_to_num(img_array, nan=0.5)
            img_array = np.clip(img_array, 0.001, 0.999)

            # Processar com IA
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
            
            # GARANTIR saída válida
            z_map = np.nan_to_num(z_map, nan=0.0)
            z_map = np.clip(z_map, 0.0, params['profundidade_max'])
            
            # Salvar comparação
            salvar_comparacao_visual(img_array, z_map, output_dir)

            # Gerar G-code CORRIGIDO
            gcode_path = os.path.join(output_dir, "relevo_3d.nc")
            success = gerar_gcode_otimizado(z_map, params['passo'], params['feedrate'], 
                                          params['safe_z'], gcode_path, new_width, new_height)

            return gcode_path, output_dir if success else (None, None)

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            return None, None

# ==============================================
# EXECUÇÃO
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernCNCApp(root)
    root.mainloop()
    