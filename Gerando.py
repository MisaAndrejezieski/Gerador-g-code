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
# CLASSE IA ESPECIALIZADA PARA MADEIRA
# ==============================================

class WoodCarvingAI:
    def __init__(self):
        self.wood_grain_cache = {}
        self.advanced_processor = AdvancedImageProcessor()
        
    def processar_para_madeira(self, image_array, profundidade_max, tipo_madeira="medium", direcao_veio="horizontal", modo_ia="wood_default"):
        """
        Processamento OTIMIZADO para entalhe em madeira
        """
        try:
            # Garantir dados válidos
            image_array = self._preprocessar_imagem(image_array)
            
            # Aplicar filtros específicos para madeira
            if tipo_madeira == "soft":  # Madeiras macias (Pinho, Cedro)
                return self._processar_madeira_macia(image_array, profundidade_max, direcao_veio)
            elif tipo_madeira == "hard":  # Madeiras duras (Carvalho, Mogno)
                return self._processar_madeira_dura(image_array, profundidade_max, direcao_veio)
            else:  # Medium (Nogueira, Cerejeira)
                return self._processar_madeira_media(image_array, profundidade_max, direcao_veio)
                
        except Exception as e:
            print(f"Erro no processamento para madeira: {e}")
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_madeira_macia(self, image_array, profundidade_max, direcao_veio):
        """
        Para madeiras macias - cortes mais suaves, menos detalhes finos
        """
        # Suavização mais agressiva
        img_suavizada = cv2.GaussianBlur(image_array, (5, 5), 1.2)
        
        # Realçar contornos médios (evitar detalhes muito finos)
        edges = cv2.Canny((img_suavizada * 255).astype(np.uint8), 50, 150) / 255.0
        
        # Combinar com imagem suavizada
        z_map = (1 - img_suavizada) * profundidade_max * 0.7 + edges * profundidade_max * 0.3
        
        # Aplicar veio da madeira
        z_map = self._aplicar_efeito_veio(z_map, direcao_veio, intensidade=0.1)
        
        return np.clip(z_map, 0, profundidade_max)

    def _processar_madeira_dura(self, image_array, profundidade_max, direcao_veio):
        """
        Para madeiras duras - permite mais detalhes e definição
        """
        # Suavização leve
        img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 0.8)
        
        # Realçar detalhes finos
        kernel_aguçamento = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        img_detalhada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
        
        # Detecção de bordas para detalhes
        edges = cv2.Canny((img_detalhada * 255).astype(np.uint8), 70, 180) / 255.0
        
        # Mapeamento mais agressivo para detalhes
        z_map = (1 - img_detalhada) * profundidade_max * 0.6 + edges * profundidade_max * 0.4
        
        # Efeito de veio mais sutil
        z_map = self._aplicar_efeito_veio(z_map, direcao_veio, intensidade=0.05)
        
        return np.clip(z_map, 0, profundidade_max)

    def _processar_madeira_media(self, image_array, profundidade_max, direcao_veio):
        """
        Para madeiras de densidade média - equilíbrio entre detalhe e suavidade
        """
        # Suavização moderada
        img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 1.0)
        
        # Aguçamento moderado
        kernel_aguçamento = np.array([[0, -0.5, 0],
                                    [-0.5, 3, -0.5],
                                    [0, -0.5, 0]])
        img_aprimorada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
        
        # Mapeamento balanceado
        z_map = (1 - img_aprimorada) * profundidade_max
        
        # Curva gamma para melhor distribuição
        gamma = 0.7
        z_map_normalized = np.clip(z_map / profundidade_max, 0.001, 0.999)
        z_map = np.power(z_map_normalized, gamma) * profundidade_max
        
        # Efeito de veio
        z_map = self._aplicar_efeito_veio(z_map, direcao_veio, intensidade=0.08)
        
        return np.clip(z_map, 0, profundidade_max)

    def _aplicar_efeito_veio(self, z_map, direcao, intensidade=0.1):
        """
        Adiciona efeito de veio da madeira ao mapa de profundidade
        """
        try:
            rows, cols = z_map.shape
            
            # Criar padrão de veio
            if direcao == "horizontal":
                veio = np.sin(np.linspace(0, 4*np.pi, cols))
                veio = np.tile(veio, (rows, 1))
            elif direcao == "vertical":
                veio = np.sin(np.linspace(0, 4*np.pi, rows))
                veio = np.tile(veio.reshape(-1, 1), (1, cols))
            else:  # diagonal
                x = np.linspace(0, 4*np.pi, cols)
                y = np.linspace(0, 4*np.pi, rows)
                X, Y = np.meshgrid(x, y)
                veio = np.sin(X + Y)
            
            # Aplicar veio com intensidade controlada
            z_map_com_veio = z_map * (1 + veio * intensidade * 0.5)
            
            return np.clip(z_map_com_veio, z_map.min(), z_map.max())
            
        except:
            return z_map

    def _preprocessar_imagem(self, image_array):
        """Pré-processamento robusto"""
        image_array = np.nan_to_num(image_array, nan=0.5)
        image_array = np.clip(image_array, 0.001, 0.999)
        return image_array

    def _processar_tradicional(self, image_array, profundidade_max):
        """Fallback seguro"""
        image_array = self._preprocessar_imagem(image_array)
        z_map = (1 - image_array) * profundidade_max
        return np.clip(z_map, 0, profundidade_max)

# ==============================================
# GERADOR G-CODE PARA MADEIRA
# ==============================================

class WoodGCodeGenerator:
    def __init__(self):
        self.wood_configs = {
            "soft": {"feedrate": 2000, "stepover": 0.8, "depth_increment": 0.5},
            "medium": {"feedrate": 1500, "stepover": 0.6, "depth_increment": 0.3},
            "hard": {"feedrate": 1000, "stepover": 0.4, "depth_increment": 0.2}
        }
    
    def gerar_gcode_madeira(self, z_map, params):
        """
        Gera G-code otimizado para entalhe em madeira
        """
        try:
            gcode_path = params['output_path']
            wood_type = params.get('wood_type', 'medium')
            config = self.wood_configs[wood_type]
            
            with open(gcode_path, "w", encoding='utf-8') as f:
                # CABEÇALHO ESPECÍFICO PARA MADEIRA
                f.write("(G-code para Entalhe em Madeira - Gerado com IA)\n")
                f.write("(OTIMIZADO PARA USINAGEM EM MADEIRA)\n")
                f.write("G21 G90 G17 G94 G49 G40\n")
                f.write(f"F{config['feedrate']}\n")
                f.write("S10000 (RPM SPINDLE)\n\n")
                
                # POSICIONAMENTO SEGURO
                f.write(f"G0 Z{params['safe_z']:.3f}\n")
                f.write("G0 X0 Y0\n")
                f.write("M3 (LIGAR SPINDLE)\n")
                f.write("G4 P2 (AGUARDAR SPINDLE)\n\n")
                
                # GERAR PERCURSO OTIMIZADO
                self._gerar_percurso_madeira(f, z_map, params, config)
                
                # FINALIZAÇÃO
                f.write(f"\nG0 Z{params['safe_z']:.3f}\n")
                f.write("M5 (DESLIGAR SPINDLE)\n")
                f.write("G0 X0 Y0\n")
                f.write("M30\n\n")
                
                # ESTATÍSTICAS
                f.write("; === ESTATÍSTICAS MADEIRA ===\n")
                f.write(f"; Tipo: {wood_type.upper()}\n")
                f.write(f"; Velocidade: {config['feedrate']} mm/min\n")
                f.write(f"; RPM: 10000\n")
                f.write(f"; Dimensões: {z_map.shape[1]}x{z_map.shape[0]} pontos\n")
            
            print(f"✅ G-code para madeira {wood_type} gerado: {gcode_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao gerar G-code madeira: {e}")
            return False

    def _gerar_percurso_madeira(self, f, z_map, params, config):
        """Gera percurso otimizado para madeira"""
        rows, cols = z_map.shape
        passo = params['passo']
        safe_z = params['safe_z']
        
        # ESTRATÉGIA: Varredura em múltiplas passes para madeira dura
        max_depth = np.max(z_map)
        if params.get('wood_type') == 'hard' and max_depth > 2:
            num_passes = max(2, int(max_depth / config['depth_increment']))
        else:
            num_passes = 1
        
        for pass_num in range(num_passes):
            if num_passes > 1:
                f.write(f"\n(PASSE {pass_num + 1} de {num_passes})\n")
            
            depth_factor = (pass_num + 1) / num_passes
            
            for y in range(rows):
                # Alternar direção (zig-zag)
                if y % 2 == 0:
                    x_range = range(cols)
                else:
                    x_range = range(cols - 1, -1, -1)
                
                primeiro_ponto = True
                
                for x in x_range:
                    z_original = z_map[y, x]
                    
                    # Para múltiplos passes, calcular profundidade deste passe
                    if num_passes > 1:
                        z_target = z_original * depth_factor
                    else:
                        z_target = z_original
                    
                    # VALIDAÇÃO CRÍTICA
                    if np.isnan(z_target) or np.isinf(z_target):
                        z_target = 0.0
                    else:
                        z_target = max(0.0, min(z_target, params['profundidade_max']))
                    
                    pos_x = (x * passo) - (cols * passo / 2)
                    pos_y = (y * passo) - (rows * passo / 2)
                    
                    if primeiro_ponto:
                        f.write(f"G0 X{pos_x:.3f} Y{pos_y:.3f}\n")
                        f.write(f"G1 Z{z_target:.3f}\n")
                        primeiro_ponto = False
                    else:
                        f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z_target:.3f}\n")

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
        # Tipo de Madeira
        ttk.Label(parent, text="Selecione o Tipo de Madeira:", style='Wood.TLabel').pack(anchor='w', pady=(10,5))
        
        self.tipo_madeira = tk.StringVar(value="medium")
        
        wood_frame = ttk.Frame(parent, style='Wood.TFrame')
        wood_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(wood_frame, text="🔸 Macia (Pinho, Cedro)", 
                       variable=self.tipo_madeira, value="soft", style='Wood.TLabel').pack(side='left', padx=10)
        ttk.Radiobutton(wood_frame, text="🔸 Média (Nogueira, Cerejeira)", 
                       variable=self.tipo_madeira, value="medium", style='Wood.TLabel').pack(side='left', padx=10)
        ttk.Radiobutton(wood_frame, text="🔸 Dura (Carvalho, Mogno)", 
                       variable=self.tipo_madeira, value="hard", style='Wood.TLabel').pack(side='left', padx=10)
        
        # Direção do Veio
        ttk.Label(parent, text="Direção do Veio:", style='Wood.TLabel').pack(anchor='w', pady=(15,5))
        
        self.direcao_veio = tk.StringVar(value="horizontal")
        
        grain_frame = ttk.Frame(parent, style='Wood.TFrame')
        grain_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(grain_frame, text="➡️ Horizontal", 
                       variable=self.direcao_veio, value="horizontal", style='Wood.TLabel').pack(side='left', padx=10)
        ttk.Radiobutton(grain_frame, text="⬇️ Vertical", 
                       variable=self.direcao_veio, value="vertical", style='Wood.TLabel').pack(side='left', padx=10)
        ttk.Radiobutton(grain_frame, text="↘️ Diagonal", 
                       variable=self.direcao_veio, value="diagonal", style='Wood.TLabel').pack(side='left', padx=10)
        
        # Dicas para cada tipo de madeira
        tips_frame = ttk.Frame(parent, style='Wood.TFrame')
        tips_frame.pack(fill='x', pady=15)
        
        tips_text = """
        💡 DICAS PARA ENTALHE:
        
        • MACIA: Ideal para peças grandes, menos detalhadas
        • MÉDIA: Equilíbrio entre detalhe e facilidade de usinagem  
        • DURA: Para peças pequenas e altamente detalhadas
        • Use ferramentas afiadas e refrigerante para madeiras duras
        """
        
        tips_label = ttk.Label(tips_frame, text=tips_text, style='Wood.TLabel', justify='left')
        tips_label.pack(anchor='w')
        
    def create_main_settings_tab(self, parent):
        """Aba principal de configurações"""
        # Seleção de arquivo
        ttk.Label(parent, text="Imagem para Entalhe:", style='Wood.TLabel').pack(anchor='w', pady=(10,5))
        
        file_frame = ttk.Frame(parent, style='Wood.TFrame')
        file_frame.pack(fill='x', pady=5)
        
        self.entry_imagem = ttk.Entry(file_frame, width=50)
        self.entry_imagem.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(file_frame, text="Procurar", command=self.selecionar_imagem, style='Wood.TButton').pack(side='right')
        
        # Parâmetros de usinagem
        params_frame = ttk.Frame(parent, style='Wood.TFrame')
        params_frame.pack(fill='x', pady=15)
        
        # Grid de parâmetros
        self.entry_largura = self.create_parameter(params_frame, "Largura (mm):", "200", 0)
        self.entry_altura = self.create_parameter(params_frame, "Altura (mm):", "150", 1)
        self.entry_profundidade = self.create_parameter(params_frame, "Profundidade Max (mm):", "4", 2)
        self.entry_passo = self.create_parameter(params_frame, "Passo (mm):", "1.0", 3)
        
        # Botão de geração
        generate_btn = ttk.Button(parent, text="🪚 GERAR ENTALHE EM MADEIRA", 
                                 command=self.gerar_entalhe, style='Wood.TButton')
        generate_btn.pack(pady=20)
        
        # Status
        self.status_label = ttk.Label(parent, text="Pronto para criar entalhe em madeira", style='Wood.TLabel')
        self.status_label.pack()
        
    def create_parameter(self, parent, label, default, row):
        """Cria um campo de parâmetro"""
        ttk.Label(parent, text=label, style='Wood.TLabel').grid(row=row, column=0, sticky='w', pady=5, padx=(0, 10))
        entry = ttk.Entry(parent, width=10)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky='w', pady=5)
        return entry
        
    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para entalhe",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            self.status_label.config(text=f"Imagem carregada: {os.path.basename(caminho)}")
            
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
                    params['grain_direction']
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
            
            return success

        except Exception as e:
            print(f"Erro no processamento madeira: {str(e)}")
            return False

    def salvar_visualizacao_madeira(self, img_original, z_map, output_dir, params):
        """Salva visualização específica para madeira"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Análise para Entalhe em Madeira ({params["wood_type"].upper()}) - Modo: {params["modo_ia"]}', fontsize=16)
            
            # Imagem original
            axes[0,0].imshow(img_original, cmap='gray')
            axes[0,0].set_title('Imagem Original')
            axes[0,0].axis('off')
            
            # Mapa de profundidade
            im = axes[0,1].imshow(z_map, cmap='terrain')
            axes[0,1].set_title('Mapa de Profundidade (mm)')
            axes[0,1].axis('off')
            plt.colorbar(im, ax=axes[0,1])
            
            # Perfil 3D
            from mpl_toolkits.mplot3d import Axes3D
            x = np.arange(z_map.shape[1])
            y = np.arange(z_map.shape[0])
            X, Y = np.meshgrid(x, y)
            
            ax3d = fig.add_subplot(2, 2, (3, 4), projection='3d')
            surf = ax3d.plot_surface(X, Y, z_map, cmap='terrain', alpha=0.8)
            ax3d.set_title('Visualização 3D do Entalhe')
            
            # Histograma de profundidades
            axes[1,1].hist(z_map.flatten(), bins=50, alpha=0.7, color='brown')
            axes[1,1].set_title('Distribuição de Profundidades')
            axes[1,1].set_xlabel('Profundidade (mm)')
            axes[1,1].set_ylabel('Frequência')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'analise_entalhe_madeira.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Erro ao salvar visualização: {e}")

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
    