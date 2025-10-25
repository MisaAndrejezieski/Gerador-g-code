import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import math
from scipy import ndimage

# ==============================================
# IA ESPECIALIZADA EM IMAGENS RELIGIOSAS (INTEGRADA)
# ==============================================

class SacredImageAI:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.important_features = {
            'face': 0.9,      # Rosto - máxima importância
            'crown': 0.85,    # Coroa
            'mantle': 0.8,    # Manto
            'hands': 0.75,    # Mãos
            'halo': 0.7,      # Auréola
            'details': 0.6    # Detalhes gerais
        }
        
    def processar_imagem_sacra(self, image_array, profundidade_max):
        """
        Processamento ESPECIALIZADO para imagens religiosas como Nossa Senhora
        """
        try:
            # Converter para PIL para processamento
            img_pil = Image.fromarray((image_array * 255).astype(np.uint8))
            
            # 1. DETECTAR E REALÇAR ROSTO
            img_pil = self._realcar_rosto(img_pil)
            
            # 2. MELHORAR DETALHES DO MANTO
            img_pil = self._realcar_manto(img_pil)
            
            # 3. DESTACAR COROA E AURÉOLA
            img_pil = self._realcar_coroa_halo(img_pil)
            
            # 4. APLICAR FILTROS ESPECÍFICOS
            img_pil = self._aplicar_filtros_sacros(img_pil)
            
            # Converter de volta para array
            img_processed = np.array(img_pil) / 255.0
            
            # 5. CRIAR MAPA DE PROFUNDIDADE INTELIGENTE
            z_map = self._criar_mapa_profundidade_inteligente(img_processed, profundidade_max)
            
            print("✅ IA sacra aplicada - Rosto, manto e coroa realçados")
            return z_map
            
        except Exception as e:
            print(f"Erro no processamento sacra: {e}")
            return self._processamento_basico(image_array, profundidade_max)

    def _realcar_rosto(self, img_pil):
        """Detecta e realça o rosto com máxima prioridade"""
        try:
            # Converter para OpenCV
            img_cv = np.array(img_pil)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY) if len(img_cv.shape) == 3 else img_cv
            
            # Detectar rostos
            faces = self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Encontrar o maior rosto (provavelmente o principal)
                main_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = main_face
                
                # Criar máscara para o rosto
                mask = np.zeros_like(gray)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                
                # Aplicar realce apenas no rosto
                img_array = np.array(img_pil)
                
                # Aumentar contraste no rosto
                roi = img_array[y:y+h, x:x+w]
                roi_enhanced = self._aumentar_contraste(roi, factor=1.8)
                img_array[y:y+h, x:x+w] = roi_enhanced
                
                # Aguçar detalhes faciais
                kernel_aguçamento = np.array([[-1, -1, -1],
                                            [-1,  12, -1],
                                            [-1, -1, -1]]) / 4.0
                roi_sharpened = cv2.filter2D(roi_enhanced, -1, kernel_aguçamento)
                img_array[y:y+h, x:x+w] = cv2.addWeighted(roi_enhanced, 0.7, roi_sharpened, 0.3, 0)
                
                img_pil = Image.fromarray(img_array)
                
                print(f"✅ Rosto detectado e realçado: {len(faces)} rosto(s) encontrado(s)")
            else:
                print("⚠️ Nenhum rosto detectado - usando heurística")
                img_pil = self._realcar_rosto_heuristica(img_pil)
                
        except Exception as e:
            print(f"Erro no realce do rosto: {e}")
            
        return img_pil

    def _realcar_rosto_heuristica(self, img_pil):
        """Realça rosto usando heurística quando detecção falha"""
        try:
            img_array = np.array(img_pil)
            h, w = img_array.shape[:2]
            
            # Supor que o rosto está no terço superior central
            face_region_height = h // 3
            face_region_width = w // 2
            start_x = w // 4
            start_y = h // 6
            
            # Realçar região do rosto
            roi = img_array[start_y:start_y+face_region_height, start_x:start_x+face_region_width]
            roi_enhanced = self._aumentar_contraste(roi, factor=1.6)
            img_array[start_y:start_y+face_region_height, start_x:start_x+face_region_width] = roi_enhanced
            
            return Image.fromarray(img_array)
            
        except:
            return img_pil

    def _realcar_manto(self, img_pil):
        """Realça detalhes do manto"""
        try:
            img_array = np.array(img_pil)
            
            # Aplicar filtro para destacar texturas (manto)
            kernel_textura = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
            
            img_enhanced = cv2.filter2D(img_array, -1, kernel_textura)
            
            # Combinar com original
            img_array = cv2.addWeighted(img_array, 0.6, img_enhanced, 0.4, 0)
            
            return Image.fromarray(img_array)
            
        except:
            return img_pil

    def _realcar_coroa_halo(self, img_pil):
        """Realça coroa e auréola"""
        try:
            img_array = np.array(img_pil)
            
            # Detectar áreas brilhantes (provavelmente coroa/auréola)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Aumentar brilho nas áreas da coroa/auréola
            img_array = img_array.astype(np.float32)
            bright_areas = bright_mask > 0
            img_array[bright_areas] *= 1.3  # Aumentar brilho
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
            
        except:
            return img_pil

    def _aplicar_filtros_sacros(self, img_pil):
        """Aplica filtros específicos para imagens sacras"""
        try:
            # Aumentar nitidez
            img_pil = img_pil.filter(ImageFilter.SHARPEN)
            
            # Aumentar contraste geral
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(1.3)
            
            # Aumentar nitidez novamente
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(2.0)
            
            return img_pil
            
        except:
            return img_pil

    def _criar_mapa_profundidade_inteligente(self, img_array, profundidade_max):
        """Cria mapa de profundidade preservando características importantes"""
        try:
            # Converter para escala de cinza se necessário
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (img_array * 255).astype(np.uint8)
            
            # 1. Mapa base
            z_base = (1 - gray / 255.0) * profundidade_max
            
            # 2. Detectar bordas importantes
            edges = cv2.Canny(gray, 50, 150) / 255.0
            
            # 3. Detectar regiões suaves (rosto)
            smooth_regions = cv2.GaussianBlur(gray, (15, 15), 5)
            smooth_mask = (cv2.Laplacian(smooth_regions, cv2.CV_64F).var() < 100)
            
            # 4. Combinar estratégias
            z_map = z_base.copy()
            
            # Realçar bordas (contornos importantes)
            z_map += edges * profundidade_max * 0.3
            
            # Suavizar rosto
            if np.any(smooth_mask):
                z_map[smooth_mask] = cv2.GaussianBlur(z_map, (5, 5), 1.5)[smooth_mask]
            
            # 5. Pós-processamento
            z_map = cv2.bilateralFilter(z_map.astype(np.float32), 5, 25, 25)
            
            return np.clip(z_map, 0, profundidade_max)
            
        except Exception as e:
            print(f"Erro no mapa de profundidade: {e}")
            return (1 - img_array) * profundidade_max

    def _aumentar_contraste(self, img_array, factor=1.5):
        """Aumenta contraste de uma região"""
        try:
            img_float = img_array.astype(np.float32)
            mean = np.mean(img_float)
            img_contrast = (img_float - mean) * factor + mean
            return np.clip(img_contrast, 0, 255).astype(np.uint8)
        except:
            return img_array

    def _processamento_basico(self, image_array, profundidade_max):
        """Fallback para processamento básico"""
        return (1 - image_array) * profundidade_max

# ==============================================
# PROCESSADOR DE IMAGENS COM IA AVANÇADA
# ==============================================

class AdvancedImageProcessor:
    def __init__(self):
        self.sacred_ai = SacredImageAI()
        self.enhancement_modes = {
            "sacred": "Imagens Sacras (Santos, Virgens)",
            "portrait": "Retratos e Rostos", 
            "landscape": "Paisagens e Natureza",
            "text": "Texto e Documentos",
            "art": "Arte e Pinturas",
            "wood_default": "Processamento Padrão Madeira"
        }
    
    def processar_imagem_avancado(self, image_path, profundidade_max, modo="sacred", params=None):
        """
        Processamento avançado com IA baseado no tipo de imagem
        """
        try:
            # Carregar imagem
            img = Image.open(image_path).convert("L")
            img_array = np.array(img) / 255.0
            
            print(f"🎯 Modo selecionado: {self.enhancement_modes[modo]}")
            
            if modo == "sacred":
                return self.sacred_ai.processar_imagem_sacra(img_array, profundidade_max)
            elif modo == "portrait":
                return self._processar_retrato(img_array, profundidade_max)
            elif modo == "landscape":
                return self._processar_paisagem(img_array, profundidade_max)
            elif modo == "text":
                return self._processar_texto(img_array, profundidade_max)
            elif modo == "art":
                return self._processar_arte(img_array, profundidade_max)
            else:
                return self._processar_padrao(img_array, profundidade_max)
                
        except Exception as e:
            print(f"Erro no processamento avançado: {e}")
            img = Image.open(image_path).convert("L")
            img_array = np.array(img) / 255.0
            return (1 - img_array) * profundidade_max

    def _processar_retrato(self, img_array, profundidade_max):
        """Processamento especializado para retratos"""
        # Similar ao sacra, mas focado apenas em rostos
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_pil = self.sacred_ai._realcar_rosto(img_pil)
        img_processed = np.array(img_pil) / 255.0
        return (1 - img_processed) * profundidade_max

    def _processar_paisagem(self, img_array, profundidade_max):
        """Processamento para paisagens"""
        # Realçar bordas e texturas naturais
        edges = cv2.Canny((img_array * 255).astype(np.uint8), 30, 100) / 255.0
        z_map = (1 - img_array) * profundidade_max * 0.7 + edges * profundidade_max * 0.3
        return np.clip(z_map, 0, profundidade_max)

    def _processar_texto(self, img_array, profundidade_max):
        """Processamento para texto"""
        # Maximizar contraste para texto
        _, binary = cv2.threshold((img_array * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        z_map = (1 - binary / 255.0) * profundidade_max
        return z_map

    def _processar_arte(self, img_array, profundidade_max):
        """Processamento para obras de arte"""
        # Preservar pinceladas e texturas artísticas
        img_enhanced = cv2.detailEnhance((img_array * 255).astype(np.uint8), sigma_s=10, sigma_r=0.15)
        z_map = (1 - img_enhanced / 255.0) * profundidade_max
        return z_map

    def _processar_padrao(self, img_array, profundidade_max):
        """Processamento padrão"""
        return (1 - img_array) * profundidade_max

# ==============================================
# CLASSE IA ESPECIALIZADA PARA MADEIRA (ORIGINAL)
# ==============================================

class WoodCarvingAI:
    def __init__(self):
        self.wood_grain_cache = {}
        self.advanced_processor = AdvancedImageProcessor()
        
    def processar_para_madeira(self, image_array, profundidade_max, tipo_madeira="medium", direcao_veio="horizontal", modo_ia="wood_default"):
        """
        Processamento OTIMIZADO para entalhe em madeira com IA avançada
        """
        try:
            # Se for modo de IA avançada, usar o processamento especializado
            if modo_ia != "wood_default":
                return self.advanced_processor.processar_imagem_avancado(
                    self._array_to_temp_image(image_array), 
                    profundidade_max, 
                    modo_ia
                )
            
            # Caso contrário, usar processamento original para madeira
            return self._processar_madeira_tradicional(image_array, profundidade_max, tipo_madeira, direcao_veio)
                
        except Exception as e:
            print(f"Erro no processamento para madeira: {e}")
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_madeira_tradicional(self, image_array, profundidade_max, tipo_madeira, direcao_veio):
        """Processamento tradicional para madeira"""
        # Garantir dados válidos
        image_array = self._preprocessar_imagem(image_array)
        
        # Aplicar filtros específicos para madeira
        if tipo_madeira == "soft":  # Madeiras macias (Pinho, Cedro)
            return self._processar_madeira_macia(image_array, profundidade_max, direcao_veio)
        elif tipo_madeira == "hard":  # Madeiras duras (Carvalho, Mogno)
            return self._processar_madeira_dura(image_array, profundidade_max, direcao_veio)
        else:  # Medium (Nogueira, Cerejeira)
            return self._processar_madeira_media(image_array, profundidade_max, direcao_veio)

    def _array_to_temp_image(self, image_array):
        """Converte array numpy para imagem temporária"""
        temp_img = Image.fromarray((image_array * 255).astype(np.uint8))
        temp_path = "temp_wood_image.png"
        temp_img.save(temp_path)
        return temp_path

    # ... (mantenha todos os outros métodos originais da WoodCarvingAI aqui)
    # _processar_madeira_macia, _processar_madeira_dura, _processar_madeira_media,
    # _aplicar_efeito_veio, _preprocessar_imagem, _processar_tradicional

# ==============================================
# INTERFACE PRINCIPAL ATUALIZADA
# ==============================================

class EnhancedWoodCarvingApp:
    def __init__(self, root):
        self.root = root
        self.wood_ai = WoodCarvingAI()
        self.gcode_gen = WoodGCodeGenerator()
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("🪵 Wood Carving Studio Pro - IA Avançada")
        self.root.geometry("1000x800")
        self.root.configure(bg='#8B4513')
        self.root.resizable(True, True)
        
        self.setup_styles()
        self.create_main_interface()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores de madeira
        style.configure('Wood.TFrame', background='#DEB887')
        style.configure('Wood.TLabel', background='#DEB887', foreground='#8B4513', font=('Segoe UI', 10))
        style.configure('WoodTitle.TLabel', background='#DEB887', foreground='#654321', 
                       font=('Segoe UI', 16, 'bold'))
        style.configure('Wood.TButton', background='#A0522D', foreground='white',
                       font=('Segoe UI', 9, 'bold'))
        style.map('Wood.TButton', background=[('active', '#8B4513')])
        
    def create_main_interface(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Wood.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title = ttk.Label(main_frame, text="🪵 Wood Carving Studio Pro - IA Avançada", style='WoodTitle.TLabel')
        title.pack(pady=(0, 20))
        
        # Abas
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # Aba Principal
        tab_principal = ttk.Frame(notebook, style='Wood.TFrame')
        notebook.add(tab_principal, text="⚙️ Configurações")
        
        # Aba Madeira
        tab_madeira = ttk.Frame(notebook, style='Wood.TFrame')
        notebook.add(tab_madeira, text="🪵 Tipo de Madeira")
        
        # Aba IA Avançada
        tab_ia = ttk.Frame(notebook, style='Wood.TFrame')
        notebook.add(tab_ia, text="🤖 IA Avançada")
        
        self.create_wood_settings_tab(tab_madeira)
        self.create_ai_advanced_tab(tab_ia)
        self.create_main_settings_tab(tab_principal)
        
    def create_ai_advanced_tab(self, parent):
        """Cria aba de processamento com IA avançada"""
        # Título
        ttk.Label(parent, text="Selecione o Tipo de Imagem:", 
                 style='Wood.TLabel', font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(15, 10))
        
        # Modo de processamento
        self.modo_processamento = tk.StringVar(value="wood_default")
        
        # Frame para os botões de modo
        mode_frame = ttk.Frame(parent, style='Wood.TFrame')
        mode_frame.pack(fill='x', pady=10, padx=20)
        
        modos = [
            ("🪵 Processamento Padrão Madeira", "wood_default"),
            ("🎭 Sacra (Santos/N.Sra)", "sacred"),
            ("👤 Retrato/Rostos", "portrait"),
            ("🏞️ Paisagem", "landscape"),
            ("📝 Texto", "text"),
            ("🎨 Arte", "art")
        ]
        
        for text, value in modos:
            btn = ttk.Radiobutton(mode_frame, text=text, variable=self.modo_processamento, 
                                 value=value, style='Wood.TLabel')
            btn.pack(anchor='w', pady=3)
        
        # Descrição do modo selecionado
        self.desc_label = ttk.Label(parent, text="💡 Processamento otimizado para entalhe em madeira", 
                                   style='Wood.TLabel', font=('Segoe UI', 9))
        self.desc_label.pack(anchor='w', pady=(15, 5), padx=20)
        
        # Configurações avançadas
        advanced_frame = ttk.Frame(parent, style='Wood.TFrame')
        advanced_frame.pack(fill='x', pady=15, padx=20)
        
        ttk.Label(advanced_frame, text="Configurações Avançadas:", 
                 style='Wood.TLabel', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Controle de intensidade do realce
        ttk.Label(advanced_frame, text="Intensidade do Realce:", style='Wood.TLabel').pack(anchor='w')
        
        self.intensidade_realce = tk.DoubleVar(value=1.0)
        scale_frame = ttk.Frame(advanced_frame, style='Wood.TFrame')
        scale_frame.pack(fill='x', pady=5)
        
        ttk.Scale(scale_frame, from_=0.5, to=2.0, variable=self.intensidade_realce,
                 orient='horizontal').pack(side='left', fill='x', expand=True)
        
        ttk.Label(scale_frame, textvariable=self.intensidade_realce, 
                 style='Wood.TLabel', width=4).pack(side='right', padx=(10, 0))
        
        # Dicas específicas
        tips_frame = ttk.Frame(parent, style='Wood.TFrame')
        tips_frame.pack(fill='x', pady=15, padx=20)
        
        tips_text = """
        🎯 DICAS PARA PROCESSAMENTO COM IA:
        
        • "Sacra": Ideal para Nossa Senhora, santos e imagens religiosas
        • "Retrato": Otimizado para rostos e características humanas
        • "Paisagem": Foca em relevos naturais e texturas
        • "Texto": Maximiza legibilidade de textos
        • "Arte": Preserva pinceladas e estilo artístico
        • "Padrão Madeira": Processamento tradicional para madeira
        
        💡 Para imagens religiosas, use o modo "Sacra" para melhor realce 
           de rostos, mantos e detalhes importantes.
        """
        
        tips_label = ttk.Label(tips_frame, text=tips_text, style='Wood.TLabel', 
                              justify='left', font=('Segoe UI', 9))
        tips_label.pack(anchor='w')
        
        # Atualizar descrição quando mudar o modo
        self.modo_processamento.trace('w', self._atualizar_descricao_modo)
    
    def _atualizar_descricao_modo(self, *args):
        """Atualiza a descrição do modo selecionado"""
        descricoes = {
            "wood_default": "💡 Processamento otimizado para entalhe em madeira",
            "sacred": "💡 Ideal para imagens religiosas, santos e virgens - realça rostos e detalhes",
            "portrait": "💡 Otimizado para retratos e destaque de rostos", 
            "landscape": "💡 Foca em relevos naturais e texturas do ambiente",
            "text": "💡 Maximiza legibilidade de texto e documentos",
            "art": "💡 Preserva pinceladas e estilo artístico"
        }
        
        modo = self.modo_processamento.get()
        self.desc_label.config(text=descricoes.get(modo, ""))

    def create_wood_settings_tab(self, parent):
        """Aba específica para configurações de madeira"""
        # ... (mantenha o código original desta função)

    def create_main_settings_tab(self, parent):
        """Aba principal de configurações"""
        # ... (mantenha o código original desta função)

    def selecionar_imagem(self):
        # ... (mantenha o código original desta função)

    def gerar_entalhe(self):
        """Função principal para gerar entalhe em madeira"""
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
                'safe_z': 5.0,
                'wood_type': self.tipo_madeira.get(),
                'grain_direction': self.direcao_veio.get(),
                'modo_ia': self.modo_processamento.get()
            }

            self.status_label.config(text="Processando imagem com IA...")
            self.root.update()
            
            # Processar
            success = self.processar_entalhe_madeira(img_path, params)
            
            if success:
                self.status_label.config(text="✅ Entalhe gerado com sucesso!")
                messagebox.showinfo("Sucesso!", "Entalhe em madeira gerado!\n\nVerifique a pasta 'WoodCarving_Output'")
            else:
                self.status_label.config(text="❌ Erro no processamento")
                
        except Exception as e:
            self.status_label.config(text="❌ Erro no processamento")
            messagebox.showerror("Erro", f"Falha: {str(e)}")

    def processar_entalhe_madeira(self, img_path, params):
        """Processamento completo para entalhe em madeira com IA"""
        try:
            output_dir = os.path.join(os.getcwd(), "WoodCarving_Output")
            os.makedirs(output_dir, exist_ok=True)

            # Carregar e preparar imagem
            img = Image.open(img_path).convert("L")
            img_array_original = np.array(img) / 255.0
            
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
            
            # PROCESSAMENTO COM IA AVANÇADA
            modo_ia = params.get('modo_ia', 'wood_default')
            
            if modo_ia == 'wood_default':
                # Processamento tradicional para madeira
                z_map = self.wood_ai.processar_para_madeira(
                    img_array, 
                    params['profundidade_max'],
                    params['wood_type'],
                    params['grain_direction'],
                    modo_ia
                )
            else:
                # Processamento com IA especializada
                z_map = self.wood_ai.advanced_processor.processar_imagem_avancado(
                    img_path,
                    params['profundidade_max'],
                    modo_ia
                )
                
                # Redimensionar o z_map para corresponder às dimensões desejadas
                if z_map.shape != (new_height, new_width):
                    z_map = cv2.resize(z_map, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Garantir dados válidos
            z_map = np.nan_to_num(z_map, nan=0.0)
            z_map = np.clip(z_map, 0.0, params['profundidade_max'])
            
            # Salvar visualização
            self.salvar_visualizacao_madeira(img_array, z_map, output_dir, params)
            
            # Gerar G-code para madeira
            gcode_filename = f"entalhe_madeira_{modo_ia}.nc"
            gcode_path = os.path.join(output_dir, gcode_filename)
            params['output_path'] = gcode_path
            
            success = self.gcode_gen.gerar_gcode_madeira(z_map, params)
            
            # Salvar dados atuais para possível análise posterior
            self.current_z_map = z_map
            self.current_params = params.copy()
            
            return success

        except Exception as e:
            print(f"Erro no processamento madeira: {str(e)}")
            return False

    def salvar_visualizacao_madeira(self, img_original, z_map, output_dir, params):
        """Salva visualização específica para madeira"""
        # ... (mantenha o código original desta função)

# ==============================================
# CLASSES ORIGINAIS (MANTIDAS)
# ==============================================

class WoodGCodeGenerator:
    # ... (mantenha toda a classe original)

class WoodFeasibilityAnalyzer:
    # ... (mantenha toda a classe original)

# ==============================================
# EXECUÇÃO PRINCIPAL
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedWoodCarvingApp(root)
    
    # Centralizar janela
    window_width = 1000
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()
    