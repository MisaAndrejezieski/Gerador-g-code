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
            # Fallback para método tradicional suave
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_preservando_detalhes(self, image_array, profundidade_max):
        """
        Técnica avançada que preserva MAXIMAMENTE os detalhes da imagem original
        """
        # 1. MANTER FIDELIDADE VISUAL COMPLETA
        # A imagem processada deve ser uma réplica 3D fiel da original
        
        # Suavizar levemente para reduzir ruído, mas manter detalhes
        img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 0.8)
        
        # 2. REALCE DE DETALHES PARA CNC
        # Usar filtro de aguçamento para destacar características importantes
        kernel_aguçamento = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        img_detalhada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
        
        # 3. MAPEAMENTO DIRETO E FIEL
        # Converter diretamente a intensidade para profundidade
        # Áreas CLARAS = MENOS profundidade (ficam mais altas)
        # Áreas ESCURAS = MAIS profundidade (ficam mais baixas)
        
        # Inverter para CNC: escuro = corta mais fundo
        z_map = (1 - img_detalhada) * profundidade_max
        
        # 4. AJUSTE DE CONTRASTE PARA OTIMIZAR USINAGEM
        # Aplicar curva gamma para melhor distribuição de profundidades
        gamma = 0.8  # Valor < 1 realça áreas escuras
        z_map = np.power(z_map / profundidade_max, gamma) * profundidade_max
        
        # 5. PRESERVAR BORDAS ORIGINAIS
        # Detectar bordas da imagem original
        bordas = cv2.Canny((image_array * 255).astype(np.uint8), 50, 150) / 255.0
        
        # Suavizar transições mas preservar bordas importantes
        z_map_suavizado = cv2.bilateralFilter(z_map.astype(np.float32), 5, 50, 50)
        
        # Combinar: manter bordas originais nítidas
        mascara_bordas = bordas > 0.1
        z_map_suavizado[mascara_bordas] = z_map[mascara_bordas]
        
        print("✅ Processamento 'Preservar Detalhes' aplicado")
        return z_map_suavizado

    def _processar_relevo_natural(self, image_array, profundidade_max):
        """
        Cria relevo natural baseado na percepção visual humana
        """
        # 1. ENFATIZAR CONTRASTES NATURAIS
        img_contraste = self._ajustar_contraste_perceptivo(image_array)
        
        # 2. MAPEAMENTO BASEADO EM PERCEPÇÃO VISUAL
        # Olho humano é mais sensível a médios tons
        z_map = self._mapeamento_perceptivo(img_contraste, profundidade_max)
        
        # 3. SUAVIZAÇÃO INTELIGENTE
        z_map = cv2.bilateralFilter(z_map.astype(np.float32), 7, 30, 30)
        
        print("✅ Processamento 'Relevo Natural' aplicado")
        return z_map

    def _processar_tradicional(self, image_array, profundidade_max):
        """
        Método tradicional melhorado
        """
        # Simples e direto - máximo de fidelidade
        z_map = (1 - image_array) * profundidade_max
        
        # Suavização mínima para evitar ruído
        z_map = cv2.GaussianBlur(z_map, (2, 2), 0.5)
        
        print("✅ Processamento 'Tradicional' aplicado")
        return z_map

    def _ajustar_contraste_perceptivo(self, image_array):
        """
        Ajusta contraste baseado em curva perceptiva
        """
        # Curve de ajuste para melhor percepção visual
        img_ajustada = np.power(image_array, 0.7)
        return img_ajustada

    def _mapeamento_perceptivo(self, image_array, profundidade_max):
        """
        Mapeamento que considera a sensibilidade visual humana
        """
        # Realçar médios tons onde o olho humano é mais sensível
        media = np.mean(image_array)
        desvio = np.std(image_array)
        
        # Criar curva de resposta não-linear
        z_map = np.zeros_like(image_array)
        
        # Áreas muito escuras: profundidade máxima
        mascara_escuro = image_array < (media - desvio/2)
        z_map[mascara_escuro] = profundidade_max * 0.9
        
        # Áreas muito claras: profundidade mínima
        mascara_claro = image_array > (media + desvio/2)
        z_map[mascara_claro] = profundidade_max * 0.1
        
        # Áreas médias: transição suave
        mascara_medio = ~(mascara_escuro | mascara_claro)
        if np.any(mascara_medio):
            intensidades_medio = image_array[mascara_medio]
            z_map[mascara_medio] = (1 - intensidades_medio) * profundidade_max * 0.8 + profundidade_max * 0.1
        
        return z_map

# ==============================================
# FUNÇÕES AUXILIARES MELHORADAS
# ==============================================

def salvar_comparacao_visual(img_original, z_map_final, output_dir):
    """
    Salva comparação visual entre original e resultado 3D
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Imagem Original
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title('IMAGEM ORIGINAL\n(Referência Visual)')
        axes[0].axis('off')
        
        # Mapa de Profundidade
        im2 = axes[1].imshow(z_map_final, cmap='viridis')
        axes[1].set_title('MAPA DE PROFUNDIDADE 3D\n(Resultado para CNC)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Sobreposição (visualização híbrida)
        img_rgb = np.stack([img_original] * 3, axis=-1)
        depth_normalized = z_map_final / z_map_final.max() if z_map_final.max() > 0 else z_map_final
        
        # Criar visualização de calor sobreposta
        heatmap = plt.cm.viridis(depth_normalized)
        alpha = 0.6
        sobreposicao = img_rgb * (1 - alpha) + heatmap[:, :, :3] * alpha
        
        axes[2].imshow(sobreposicao)
        axes[2].set_title('SOBREPOSIÇÃO\n(Original + Profundidade)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparacao_visual_detalhada.png'), 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Comparação visual salva com sucesso!")
        
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível salvar comparação visual: {e}")

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
        
        # Salvar o heightmap em cores para melhor visualização
        plt.figure(figsize=(10, 8))
        plt.imshow(z_map, cmap='viridis')
        plt.colorbar(label='Profundidade (mm)')
        plt.title('Mapa de Profundidade - Visualização Colorida')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "heightmap_colorido.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Imagens de processo salvas!")
    except Exception as e:
        print(f"⚠️ Aviso: Erro ao salvar imagens: {e}")

# ==============================================
# FUNÇÃO PRINCIPAL COMPLETAMENTE REPROJETADA
# ==============================================

def processar_imagem_ia(img_path, largura_mm, altura_mm, profundidade_max, passo, feedrate, safe_z, uso_ia=True, tipo_relevo="baixo", metodo_processamento="preservar_detalhes"):
    """
    Função principal COMPLETAMENTE REPROJETADA para máxima fidelidade visual
    """
    try:
        # Criar diretório de saída
        output_dir = os.path.join(os.getcwd(), "Imagens_Processadas_IA")
        os.makedirs(output_dir, exist_ok=True)

        print("📁 CARREGANDO IMAGEM ORIGINAL...")
        # Abrir imagem ORIGINAL - MÍNIMO de processamento
        img_original = Image.open(img_path).convert("L")
        
        # CONVERTER para array numpy PRESERVANDO valores originais
        img_array_original = np.array(img_original) / 255.0
        
        print("🎯 CALCULANDO DIMENSÕES...")
        # Calcular dimensões mantendo proporção EXATA da original
        img_ratio = img_original.width / img_original.height
        target_ratio = largura_mm / altura_mm
        
        if img_ratio > target_ratio:
            new_width = int(largura_mm / passo)
            new_height = int(new_width / img_ratio)
        else:
            new_height = int(altura_mm / passo)
            new_width = int(new_height * img_ratio)
        
        print(f"📐 REDIMENSIONANDO: {img_original.size} -> {new_width}x{new_height}")
        # Redimensionar com alta qualidade
        img_resized = img_original.resize((new_width, new_height), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0

        print("🔄 PROCESSANDO RELEVO 3D...")
        # PROCESSAMENTO INTELIGENTE
        ai_processor = AICNCProcessor()
        
        if uso_ia:
            print(f"🤖 USANDO MÉTODO: {metodo_processamento}")
            z_map = ai_processor.processar_imagem_inteligente(
                img_array, profundidade_max, metodo_processamento
            )
        else:
            print("🔧 USANDO MÉTODO TRADICIONAL")
            # Método tradicional DIRETO
            if tipo_relevo == "baixo":
                z_map = (1 - img_array) * profundidade_max
            else:
                z_map = img_array * profundidade_max
        
        # GARANTIR qualidade do resultado
        z_map = np.clip(z_map, 0, profundidade_max)
        
        # Salvar COMPARAÇÃO VISUAL detalhada
        salvar_comparacao_visual(img_array, z_map, output_dir)
        salvar_imagens_processo(img_array, z_map, output_dir, uso_ia)

        print("⚡ GERANDO G-CODE...")
        gcode_path = os.path.join(output_dir, "relevo_3d_fiel.nc")
        success = gerar_gcode_otimizado(z_map, passo, feedrate, safe_z, gcode_path, new_width, new_height)

        if success:
            print("🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
            print(f"📊 Estatísticas finais:")
            print(f"   - Profundidade mínima: {z_map.min():.2f}mm")
            print(f"   - Profundidade máxima: {z_map.max():.2f}mm") 
            print(f"   - Dimensões: {new_width}x{new_height} pontos")
            return gcode_path, output_dir
        else:
            return None, None

    except Exception as e:
        print(f"❌ ERRO NO PROCESSAMENTO: {str(e)}")
        messagebox.showerror("Erro", f"Erro no processamento: {str(e)}")
        return None, None

# ==============================================
# INTERFACE ATUALIZADA COM NOVAS OPÇÕES
# ==============================================

class GeradorCNCIA:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("Gerador de G-code CNC - FIDELIDADE VISUAL")
        self.root.geometry("750x800")
        self.root.resizable(True, True)
        
        self.setup_styles()
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Título
        title = ttk.Label(main_frame, text="🎨 CNC ROUTER - FIDELIDADE VISUAL GARANTIDA", 
                         font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        self.create_widgets(main_frame)
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Generate.TButton", font=("Segoe UI", 12, "bold"))
        
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
        
        # NOVO: Método de Processamento
        ttk.Label(parent, text="🔧 Método de Processamento:", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(20,10))
        row += 1
        
        self.metodo_processamento = tk.StringVar(value="preservar_detalhes")
        frame_metodo = ttk.Frame(parent)
        frame_metodo.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Radiobutton(frame_metodo, text="🎯 Preservar Detalhes (Recomendado)", 
                       variable=self.metodo_processamento, value="preservar_detalhes").pack(anchor="w")
        ttk.Radiobutton(frame_metago, text="🌊 Relevo Natural", 
                       variable=self.metodo_processamento, value="relevo_natural").pack(anchor="w")
        ttk.Radiobutton(frame_metodo, text="⚡ Tradicional Rápido", 
                       variable=self.metodo_processamento, value="tradicional").pack(anchor="w")
        row += 1
        
        # Configurações de IA
        ttk.Label(parent, text="🤖 Processamento com IA:", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(20,10))
        row += 1
        
        self.uso_ia = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Usar processamento inteligente (Recomendado para melhor qualidade)", 
                       variable=self.uso_ia).grid(row=row, column=0, sticky="w", pady=5)
        row += 1
        
        # Tipo de relevo (apenas se IA desativada)
        ttk.Label(parent, text="🎨 Tipo de Relevo (se IA desativada):", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(10,5))
        row += 1
        
        self.tipo_relevo = tk.StringVar(value="baixo")
        ttk.Radiobutton(parent, text="Baixo Relevo", variable=self.tipo_relevo, value="baixo").grid(row=row, column=0, sticky="w")
        ttk.Radiobutton(parent, text="Alto Relevo", variable=self.tipo_relevo, value="alto").grid(row=row, column=0, sticky="w")
        row += 1
        
        # Parâmetros de usinagem
        params = [
            ("📏 Largura (mm):", "200", "entry_largura"),
            ("📐 Altura (mm):", "150", "entry_altura"), 
            ("⏬ Profundidade máxima (mm):", "3", "entry_profundidade"),
            ("🔍 Passo entre pontos (mm):", "1.0", "entry_passo"),
            ("⚡ Velocidade de avanço (mm/min):", "1200", "entry_feed"),
            ("🛡️ Safe Z (mm):", "5", "entry_safez")
        ]
        
        for label, default, attr_name in params:
            ttk.Label(parent, text=label, style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(15,5))
            row += 1
            
            entry = ttk.Entry(parent, width=20, font=("Segoe UI", 10))
            entry.insert(0, default)
            entry.grid(row=row, column=0, sticky="w", pady=2, padx=(20,0))
            setattr(self, attr_name, entry)
            row += 1
        
        # Botão GERAR
        ttk.Label(parent, text="", style="Title.TLabel").grid(row=row, column=0, pady=(20,0))
        row += 1
        
        btn_gerar = ttk.Button(parent, text="🚀 GERAR G-CODE FIEL", 
                              command=self.gerar, 
                              style="Generate.TButton",
                              width=20)
        btn_gerar.grid(row=row, column=0, columnspan=2, pady=30)
        row += 1
        
        # Rodapé
        rodape = ttk.Label(parent, 
                          text="© 2025 - CNC Router | FIDELIDADE VISUAL GARANTIDA - Versão 3.0", 
                          font=("Segoe UI", 8), foreground="gray")
        rodape.grid(row=row, column=0, columnspan=2, pady=20)
        
        parent.columnconfigure(0, weight=1)
    
    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para processamento",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            messagebox.showinfo("Imagem Selecionada", 
                              f"Imagem carregada para processamento:\n{os.path.basename(caminho)}\n\n"
                              f"O resultado 3D será uma RÉPLICA FIEL da imagem original!")
            
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

            # Processar imagem
            messagebox.showinfo("Processando", 
                              f"Processando imagem com MÁXIMA FIDELIDADE VISUAL...\n\n"
                              f"Método: {params['metodo_processamento']}\n"
                              f"IA: {'ATIVADA' if params['uso_ia'] else 'Desativada'}\n\n"
                              f"O resultado será uma réplica 3D fiel da imagem original!")
            
            gcode_path, output_dir = processar_imagem_ia(img_path, **params)

            if gcode_path and output_dir:
                messagebox.showinfo("Sucesso!", 
                    f"✅ PROCESSAMENTO CONCLUÍDO!\n\n"
                    f"🎯 FIDELIDADE VISUAL GARANTIDA\n"
                    f"📁 Pasta: {output_dir}\n"
                    f"📊 Comparação: comparacao_visual_detalhada.png\n"
                    f"⚡ G-code: {os.path.basename(gcode_path)}\n\n"
                    f"Verifique a comparação visual gerada - o relevo 3D é uma RÉPLICA FIEL da imagem original!")

        except ValueError as e:
            messagebox.showerror("Erro", f"Valor inválido: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha: {str(e)}")

# ==============================================
# EXECUÇÃO
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GeradorCNCIA(root)
    root.mainloop()
    