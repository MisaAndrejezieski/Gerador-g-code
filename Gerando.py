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
import joblib
from collections import defaultdict

# ==============================================
# SISTEMA DE APRENDIZADO MÁQUINA (SEM PANDAS)
# ==============================================

class LearningSystem:
    def __init__(self):
        self.feedback_data = []
        self.model_path = "wood_learning_model.pkl"
        self.dataset_path = "learning_dataset.json"
        self.model = None
        self.load_model()
        self.load_feedback()
        
    def extract_features(self, image_array, params):
        """Extrai características da imagem para aprendizado"""
        try:
            features = {}
            
            # Estatísticas da imagem
            features['mean_intensity'] = float(np.mean(image_array))
            features['std_intensity'] = float(np.std(image_array))
            features['contrast'] = float(np.max(image_array) - np.min(image_array))
            
            # Características de textura
            edges = cv2.Canny((image_array * 255).astype(np.uint8), 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            # Histogram features (simplificado)
            hist, _ = np.histogram(image_array, bins=5)
            for i, val in enumerate(hist):
                features[f'hist_bin_{i}'] = float(val)
            
            # Parâmetros do usuário
            wood_type_map = {'soft': 0, 'medium': 1, 'hard': 2}
            grain_map = {'horizontal': 0, 'vertical': 1, 'diagonal': 2}
            
            features['wood_type'] = wood_type_map.get(params.get('wood_type', 'medium'), 1)
            features['grain_direction'] = grain_map.get(params.get('grain_direction', 'horizontal'), 0)
            features['profundidade_max'] = float(params.get('profundidade_max', 4.0))
            features['passo'] = float(params.get('passo', 1.0))
            
            return features
            
        except Exception as e:
            print(f"Erro ao extrair features: {e}")
            return {}
    
    def add_feedback(self, image_array, params, z_map_result, user_rating, user_notes=""):
        """Adiciona feedback do usuário para aprendizado"""
        try:
            features = self.extract_features(image_array, params)
            
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'params': params,
                'user_rating': user_rating,
                'user_notes': user_notes,
                'z_map_stats': {
                    'mean_depth': float(np.mean(z_map_result)),
                    'max_depth': float(np.max(z_map_result)),
                    'std_depth': float(np.std(z_map_result))
                }
            }
            
            self.feedback_data.append(feedback_entry)
            self.save_feedback()
            
            # Treinar modelo incrementalmente
            if len(self.feedback_data) >= 3:  # Reduzido para 3 amostras
                self.train_model_simple()
                
            print(f"✅ Feedback salvo (Rating: {user_rating}/5)")
            
        except Exception as e:
            print(f"Erro ao salvar feedback: {e}")
    
    def train_model_simple(self):
        """Treina modelo simples baseado em médias"""
        try:
            if len(self.feedback_data) < 2:
                return
                
            # Análise simples baseada em ratings
            high_rated = [entry for entry in self.feedback_data if entry['user_rating'] >= 4]
            low_rated = [entry for entry in self.feedback_data if entry['user_rating'] <= 2]
            
            if high_rated:
                # Calcular médias dos parâmetros bem avaliados
                self.optimal_params = self.calculate_average_params(high_rated)
                print("✅ Modelo simples treinado com amostras bem avaliadas")
            else:
                self.optimal_params = None
                
        except Exception as e:
            print(f"Erro no treinamento simples: {e}")
    
    def calculate_average_params(self, entries):
        """Calcula parâmetros médios baseado em entries"""
        avg_params = defaultdict(float)
        count = len(entries)
        
        for entry in entries:
            for key, value in entry['features'].items():
                avg_params[key] += value / count
        
        return dict(avg_params)
    
    def predict_optimal_params(self, image_array, current_params):
        """Sugere parâmetros otimizados baseado no aprendizado"""
        try:
            if not hasattr(self, 'optimal_params') or self.optimal_params is None:
                return current_params
            
            features = self.extract_features(image_array, current_params)
            if not features:
                return current_params
            
            # Ajuste simples baseado em features
            optimized_params = current_params.copy()
            
            # Exemplo: se muita densidade de bordas, reduzir profundidade
            if features.get('edge_density', 0) > 0.3:
                optimized_params['profundidade_max'] *= 0.8
            elif features.get('edge_density', 0) < 0.1:
                optimized_params['profundidade_max'] *= 1.2
                
            return optimized_params
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return current_params
    
    def load_model(self):
        """Carrega modelo salvo"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ Modelo de aprendizado carregado")
        except:
            self.model = None
    
    def save_feedback(self):
        """Salva dados de feedback em JSON"""
        try:
            with open(self.dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            print("✅ Feedback salvo no dataset")
        except Exception as e:
            print(f"Erro ao salvar feedback: {e}")
    
    def load_feedback(self):
        """Carrega feedback do arquivo JSON"""
        try:
            if os.path.exists(self.dataset_path):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                print(f"✅ {len(self.feedback_data)} amostras de feedback carregadas")
        except Exception as e:
            print(f"Erro ao carregar feedback: {e}")
            self.feedback_data = []
    
    def get_learning_stats(self):
        """Retorna estatísticas do aprendizado"""
        if not self.feedback_data:
            return {"total_samples": 0, "average_rating": 0, "model_trained": False}
        
        ratings = [entry['user_rating'] for entry in self.feedback_data]
        return {
            "total_samples": len(self.feedback_data),
            "average_rating": float(np.mean(ratings)),
            "model_trained": hasattr(self, 'optimal_params') and self.optimal_params is not None
        }

# ==============================================
# IA ESPECIALIZADA EM IMAGENS RELIGIOSAS
# ==============================================

class SacredImageAI:
    def __init__(self):
        try:
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_detector = None
            print("⚠️ Detector de rostos não disponível")
        
    def processar_imagem_sacra(self, image_array, profundidade_max):
        """Processamento ESPECIALIZADO para imagens religiosas"""
        try:
            img_pil = Image.fromarray((image_array * 255).astype(np.uint8))
            
            # 1. DETECTAR E REALÇAR ROSTO
            img_pil = self._realcar_rosto(img_pil)
            
            # 2. MELHORAR DETALHES
            img_pil = self._realcar_manto(img_pil)
            img_pil = self._realcar_coroa_halo(img_pil)
            img_pil = self._aplicar_filtros_sacros(img_pil)
            
            img_processed = np.array(img_pil) / 255.0
            z_map = self._criar_mapa_profundidade_inteligente(img_processed, profundidade_max)
            
            print("✅ IA sacra aplicada")
            return z_map
            
        except Exception as e:
            print(f"Erro no processamento sacra: {e}")
            return (1 - image_array) * profundidade_max

    def _realcar_rosto(self, img_pil):
        """Detecta e realça o rosto"""
        try:
            if self.face_detector is None:
                return img_pil
                
            img_cv = np.array(img_pil)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY) if len(img_cv.shape) == 3 else img_cv
            
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                main_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = main_face
                
                img_array = np.array(img_pil)
                roi = img_array[y:y+h, x:x+w]
                roi_enhanced = self._aumentar_contraste(roi, factor=1.8)
                img_array[y:y+h, x:x+w] = roi_enhanced
                img_pil = Image.fromarray(img_array)
                
            return img_pil
        except:
            return img_pil

    def _realcar_manto(self, img_pil):
        """Realça detalhes do manto"""
        try:
            img_array = np.array(img_pil)
            kernel_textura = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img_enhanced = cv2.filter2D(img_array, -1, kernel_textura)
            img_array = cv2.addWeighted(img_array, 0.6, img_enhanced, 0.4, 0)
            return Image.fromarray(img_array)
        except:
            return img_pil

    def _realcar_coroa_halo(self, img_pil):
        """Realça coroa e auréola"""
        try:
            img_array = np.array(img_pil)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            img_array = img_array.astype(np.float32)
            bright_areas = bright_mask > 0
            img_array[bright_areas] *= 1.3
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        except:
            return img_pil

    def _aplicar_filtros_sacros(self, img_pil):
        """Aplica filtros específicos"""
        try:
            img_pil = img_pil.filter(ImageFilter.SHARPEN)
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(2.0)
            return img_pil
        except:
            return img_pil

    def _criar_mapa_profundidade_inteligente(self, img_array, profundidade_max):
        """Cria mapa de profundidade inteligente"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (img_array * 255).astype(np.uint8)
            
            z_base = (1 - gray / 255.0) * profundidade_max
            edges = cv2.Canny(gray, 50, 150) / 255.0
            z_map = z_base + edges * profundidade_max * 0.3
            z_map = cv2.bilateralFilter(z_map.astype(np.float32), 5, 25, 25)
            return np.clip(z_map, 0, profundidade_max)
        except:
            return (1 - img_array) * profundidade_max

    def _aumentar_contraste(self, img_array, factor=1.5):
        """Aumenta contraste"""
        try:
            img_float = img_array.astype(np.float32)
            mean = np.mean(img_float)
            img_contrast = (img_float - mean) * factor + mean
            return np.clip(img_contrast, 0, 255).astype(np.uint8)
        except:
            return img_array

# ==============================================
# PROCESSADOR DE IMAGENS COM IA AVANÇADA
# ==============================================

class AdvancedImageProcessor:
    def __init__(self):
        self.sacred_ai = SacredImageAI()
        
    def processar_imagem_avancado(self, image_path, profundidade_max, modo="sacred"):
        """Processamento avançado com IA"""
        try:
            img = Image.open(image_path).convert("L")
            img_array = np.array(img) / 255.0
            
            print(f"🎯 Modo selecionado: {modo}")
            
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
                return (1 - img_array) * profundidade_max
                
        except Exception as e:
            print(f"Erro no processamento: {e}")
            img = Image.open(image_path).convert("L")
            img_array = np.array(img) / 255.0
            return (1 - img_array) * profundidade_max

    def _processar_retrato(self, img_array, profundidade_max):
        """Processamento para retratos"""
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_pil = self.sacred_ai._realcar_rosto(img_pil)
        img_processed = np.array(img_pil) / 255.0
        return (1 - img_processed) * profundidade_max

    def _processar_paisagem(self, img_array, profundidade_max):
        """Processamento para paisagens"""
        edges = cv2.Canny((img_array * 255).astype(np.uint8), 30, 100) / 255.0
        z_map = (1 - img_array) * profundidade_max * 0.7 + edges * profundidade_max * 0.3
        return np.clip(z_map, 0, profundidade_max)

    def _processar_texto(self, img_array, profundidade_max):
        """Processamento para texto"""
        _, binary = cv2.threshold((img_array * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        z_map = (1 - binary / 255.0) * profundidade_max
        return z_map

    def _processar_arte(self, img_array, profundidade_max):
        """Processamento para arte"""
        img_enhanced = cv2.detailEnhance((img_array * 255).astype(np.uint8), sigma_s=10, sigma_r=0.15)
        z_map = (1 - img_enhanced / 255.0) * profundidade_max
        return z_map

# ==============================================
# CLASSE IA ESPECIALIZADA PARA MADEIRA
# ==============================================

class WoodCarvingAI:
    def __init__(self):
        self.advanced_processor = AdvancedImageProcessor()
        
    def processar_para_madeira(self, image_array, profundidade_max, tipo_madeira="medium", direcao_veio="horizontal"):
        """Processamento OTIMIZADO para entalhe em madeira"""
        try:
            image_array = self._preprocessar_imagem(image_array)
            
            if tipo_madeira == "soft":
                return self._processar_madeira_macia(image_array, profundidade_max, direcao_veio)
            elif tipo_madeira == "hard":
                return self._processar_madeira_dura(image_array, profundidade_max, direcao_veio)
            else:
                return self._processar_madeira_media(image_array, profundidade_max, direcao_veio)
                
        except Exception as e:
            print(f"Erro no processamento: {e}")
            return self._processar_tradicional(image_array, profundidade_max)

    def _processar_madeira_macia(self, image_array, profundidade_max, direcao_veio):
        """Para madeiras macias"""
        img_suavizada = cv2.GaussianBlur(image_array, (5, 5), 1.2)
        edges = cv2.Canny((img_suavizada * 255).astype(np.uint8), 50, 150) / 255.0
        z_map = (1 - img_suavizada) * profundidade_max * 0.7 + edges * profundidade_max * 0.3
        z_map = self._aplicar_efeito_veio(z_map, direcao_veio, 0.1)
        return np.clip(z_map, 0, profundidade_max)

    def _processar_madeira_dura(self, image_array, profundidade_max, direcao_veio):
        """Para madeiras duras"""
        img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 0.8)
        kernel_aguçamento = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_detalhada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
        edges = cv2.Canny((img_detalhada * 255).astype(np.uint8), 70, 180) / 255.0
        z_map = (1 - img_detalhada) * profundidade_max * 0.6 + edges * profundidade_max * 0.4
        z_map = self._aplicar_efeito_veio(z_map, direcao_veio, 0.05)
        return np.clip(z_map, 0, profundidade_max)

    def _processar_madeira_media(self, image_array, profundidade_max, direcao_veio):
        """Para madeiras médias"""
        img_suavizada = cv2.GaussianBlur(image_array, (3, 3), 1.0)
        kernel_aguçamento = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        img_aprimorada = cv2.filter2D(img_suavizada, -1, kernel_aguçamento)
        z_map = (1 - img_aprimorada) * profundidade_max
        gamma = 0.7
        z_map_normalized = np.clip(z_map / profundidade_max, 0.001, 0.999)
        z_map = np.power(z_map_normalized, gamma) * profundidade_max
        z_map = self._aplicar_efeito_veio(z_map, direcao_veio, 0.08)
        return np.clip(z_map, 0, profundidade_max)

    def _aplicar_efeito_veio(self, z_map, direcao, intensidade=0.1):
        """Adiciona efeito de veio"""
        try:
            rows, cols = z_map.shape
            if direcao == "horizontal":
                veio = np.sin(np.linspace(0, 4*np.pi, cols))
                veio = np.tile(veio, (rows, 1))
            elif direcao == "vertical":
                veio = np.sin(np.linspace(0, 4*np.pi, rows))
                veio = np.tile(veio.reshape(-1, 1), (1, cols))
            else:
                x = np.linspace(0, 4*np.pi, cols)
                y = np.linspace(0, 4*np.pi, rows)
                X, Y = np.meshgrid(x, y)
                veio = np.sin(X + Y)
            
            z_map_com_veio = z_map * (1 + veio * intensidade * 0.5)
            return np.clip(z_map_com_veio, z_map.min(), z_map.max())
        except:
            return z_map

    def _preprocessar_imagem(self, image_array):
        """Pré-processamento"""
        image_array = np.nan_to_num(image_array, nan=0.5)
        return np.clip(image_array, 0.001, 0.999)

    def _processar_tradicional(self, image_array, profundidade_max):
        """Fallback"""
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
        """Gera G-code otimizado"""
        try:
            gcode_path = params['output_path']
            wood_type = params.get('wood_type', 'medium')
            config = self.wood_configs[wood_type]
            
            with open(gcode_path, "w", encoding='utf-8') as f:
                f.write("(G-code para Entalhe em Madeira)\n")
                f.write("G21 G90 G17 G94 G49 G40\n")
                f.write(f"F{config['feedrate']}\n")
                f.write("S10000\n\n")
                
                f.write(f"G0 Z{params['safe_z']:.3f}\n")
                f.write("G0 X0 Y0\n")
                f.write("M3\n")
                f.write("G4 P2\n\n")
                
                self._gerar_percurso_madeira(f, z_map, params, config)
                
                f.write(f"\nG0 Z{params['safe_z']:.3f}\n")
                f.write("M5\n")
                f.write("G0 X0 Y0\n")
                f.write("M30\n")
            
            print(f"✅ G-code gerado: {gcode_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro G-code: {e}")
            return False

    def _gerar_percurso_madeira(self, f, z_map, params, config):
        """Gera percurso"""
        rows, cols = z_map.shape
        passo = params['passo']
        safe_z = params['safe_z']
        
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
                if y % 2 == 0:
                    x_range = range(cols)
                else:
                    x_range = range(cols - 1, -1, -1)
                
                primeiro_ponto = True
                
                for x in x_range:
                    z_original = z_map[y, x]
                    
                    if num_passes > 1:
                        z_target = z_original * depth_factor
                    else:
                        z_target = z_original
                    
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
# INTERFACE PRINCIPAL COM APRENDIZADO
# ==============================================

class EnhancedWoodCarvingApp:
    def __init__(self, root):
        self.root = root
        self.wood_ai = WoodCarvingAI()
        self.gcode_gen = WoodGCodeGenerator()
        self.learning_system = LearningSystem()
        self.current_z_map = None
        self.current_params = None
        self.current_image_array = None
        
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("🪵 Wood Carving Studio Pro - COM APRENDIZADO")
        self.root.geometry("1000x800")
        self.root.configure(bg='#8B4513')
        
        self.setup_styles()
        self.create_main_interface()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Wood.TFrame', background='#DEB887')
        style.configure('Wood.TLabel', background='#DEB887', foreground='#8B4513', font=('Segoe UI', 10))
        style.configure('WoodTitle.TLabel', background='#DEB887', foreground='#654321', font=('Segoe UI', 16, 'bold'))
        style.configure('Wood.TButton', background='#A0522D', foreground='white', font=('Segoe UI', 9, 'bold'))
        style.map('Wood.TButton', background=[('active', '#8B4513')])
        
    def create_main_interface(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Wood.TFrame')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Título
        title = ttk.Label(main_frame, text="🪵 Wood Carving Studio Pro - COM SISTEMA DE APRENDIZADO", style='WoodTitle.TLabel')
        title.pack(pady=(0, 20))
        
        # Abas
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # Criar abas
        tab_principal = ttk.Frame(notebook, style='Wood.TFrame')
        tab_madeira = ttk.Frame(notebook, style='Wood.TFrame')
        tab_ia = ttk.Frame(notebook, style='Wood.TFrame')
        tab_learning = ttk.Frame(notebook, style='Wood.TFrame')
        
        notebook.add(tab_principal, text="⚙️ Principal")
        notebook.add(tab_madeira, text="🪵 Madeira")
        notebook.add(tab_ia, text="🤖 IA")
        notebook.add(tab_learning, text="🧠 Aprendizado")
        
        self.create_main_tab(tab_principal)
        self.create_wood_tab(tab_madeira)
        self.create_ai_tab(tab_ia)
        self.create_learning_tab(tab_learning)
        
    def create_main_tab(self, parent):
        """Aba principal de configurações"""
        ttk.Label(parent, text="Configurações Principais", style='Wood.TLabel', font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(10, 20))
        
        # Seleção de arquivo
        file_frame = ttk.Frame(parent, style='Wood.TFrame')
        file_frame.pack(fill='x', pady=10)
        
        ttk.Label(file_frame, text="Imagem para Entalhe:", style='Wood.TLabel').pack(anchor='w')
        
        file_subframe = ttk.Frame(file_frame, style='Wood.TFrame')
        file_subframe.pack(fill='x', pady=5)
        
        self.entry_imagem = ttk.Entry(file_subframe, font=('Segoe UI', 10))
        self.entry_imagem.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(file_subframe, text="📁 Procurar", command=self.selecionar_imagem, style='Wood.TButton').pack(side='right')
        
        # Parâmetros de usinagem
        params_frame = ttk.Frame(parent, style='Wood.TFrame')
        params_frame.pack(fill='x', pady=15)
        
        ttk.Label(params_frame, text="Parâmetros de Usinagem:", style='Wood.TLabel', font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Grid de parâmetros
        param_grid = ttk.Frame(params_frame, style='Wood.TFrame')
        param_grid.pack(fill='x')
        
        self.entry_largura = self.create_parameter(param_grid, "Largura (mm):", "200", 0)
        self.entry_altura = self.create_parameter(param_grid, "Altura (mm):", "150", 1)
        self.entry_profundidade = self.create_parameter(param_grid, "Profundidade Máx (mm):", "4.0", 2)
        self.entry_passo = self.create_parameter(param_grid, "Passo (mm):", "1.0", 3)
        self.entry_safe_z = self.create_parameter(param_grid, "Altura Segurança (mm):", "5.0", 4)
        
        # Botões principais
        button_frame = ttk.Frame(parent, style='Wood.TFrame')
        button_frame.pack(fill='x', pady=20)
        
        ttk.Button(button_frame, text="🪚 GERAR ENTALHE", 
                  command=self.gerar_entalhe, style='Wood.TButton').pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text="📊 PRÉ-VISUALIZAR", 
                  command=self.previsualizar, style='Wood.TButton').pack(side='left')
        
        # Status
        self.status_label = ttk.Label(parent, text="Pronto para criar entalhe em madeira", 
                                     style='Wood.TLabel', font=('Segoe UI', 10))
        self.status_label.pack(pady=10)
        
    def create_wood_tab(self, parent):
        """Aba de configurações de madeira"""
        ttk.Label(parent, text="Configurações da Madeira", style='Wood.TLabel', font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(10, 20))
        
        # Tipo de Madeira
        ttk.Label(parent, text="Tipo de Madeira:", style='Wood.TLabel').pack(anchor='w', pady=(10,5))
        
        self.tipo_madeira = tk.StringVar(value="medium")
        
        wood_frame = ttk.Frame(parent, style='Wood.TFrame')
        wood_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(wood_frame, text="🔸 Macia (Pinho, Cedro)", 
                       variable=self.tipo_madeira, value="soft", style='Wood.TLabel').pack(anchor='w', pady=2)
        ttk.Radiobutton(wood_frame, text="🔸 Média (Nogueira, Cerejeira)", 
                       variable=self.tipo_madeira, value="medium", style='Wood.TLabel').pack(anchor='w', pady=2)
        ttk.Radiobutton(wood_frame, text="🔸 Dura (Carvalho, Mogno)", 
                       variable=self.tipo_madeira, value="hard", style='Wood.TLabel').pack(anchor='w', pady=2)
        
        # Direção do Veio
        ttk.Label(parent, text="Direção do Veio:", style='Wood.TLabel').pack(anchor='w', pady=(15,5))
        
        self.direcao_veio = tk.StringVar(value="horizontal")
        
        grain_frame = ttk.Frame(parent, style='Wood.TFrame')
        grain_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(grain_frame, text="➡️ Horizontal", 
                       variable=self.direcao_veio, value="horizontal", style='Wood.TLabel').pack(side='left', padx=(0, 20))
        ttk.Radiobutton(grain_frame, text="⬇️ Vertical", 
                       variable=self.direcao_veio, value="vertical", style='Wood.TLabel').pack(side='left', padx=(0, 20))
        ttk.Radiobutton(grain_frame, text="↘️ Diagonal", 
                       variable=self.direcao_veio, value="diagonal", style='Wood.TLabel').pack(side='left')
        
    def create_ai_tab(self, parent):
        """Aba de IA avançada"""
        ttk.Label(parent, text="Processamento com IA", style='Wood.TLabel', font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(10, 20))
        
        # Modo de processamento
        ttk.Label(parent, text="Modo de Processamento:", style='Wood.TLabel').pack(anchor='w', pady=(10,5))
        
        self.modo_processamento = tk.StringVar(value="wood_default")
        
        modes = [
            ("🪵 Padrão Madeira", "wood_default"),
            ("🎭 Imagens Sacras", "sacred"),
            ("👤 Retratos", "portrait"), 
            ("🏞️ Paisagens", "landscape"),
            ("📝 Texto", "text"),
            ("🎨 Arte", "art")
        ]
        
        for text, value in modes:
            ttk.Radiobutton(parent, text=text, variable=self.modo_processamento, 
                           value=value, style='Wood.TLabel').pack(anchor='w', pady=2)
        
        # Descrição do modo
        self.modo_desc = ttk.Label(parent, text="Processamento otimizado para entalhe em madeira", 
                                  style='Wood.TLabel', font=('Segoe UI', 9))
        self.modo_desc.pack(anchor='w', pady=15)
        
        self.modo_processamento.trace('w', self.atualizar_descricao_modo)
        
    def create_learning_tab(self, parent):
        """Aba do sistema de aprendizado"""
        ttk.Label(parent, text="Sistema de Aprendizado", style='Wood.TLabel', font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(10, 20))
        
        # Estatísticas
        stats_frame = ttk.Frame(parent, style='Wood.TFrame')
        stats_frame.pack(fill='x', pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="Carregando estatísticas...", 
                                    style='Wood.TLabel', justify='left')
        self.stats_label.pack(anchor='w')
        
        # Feedback do último trabalho
        ttk.Label(parent, text="Avaliar Último Trabalho:", style='Wood.TLabel').pack(anchor='w', pady=(20,5))
        
        feedback_frame = ttk.Frame(parent, style='Wood.TFrame')
        feedback_frame.pack(fill='x', pady=5)
        
        self.feedback_rating = tk.IntVar(value=5)
        
        rating_frame = ttk.Frame(feedback_frame, style='Wood.TFrame')
        rating_frame.pack(fill='x', pady=5)
        
        for i in range(1, 6):
            ttk.Radiobutton(rating_frame, text="⭐" * i, variable=self.feedback_rating, 
                           value=i, style='Wood.TLabel').pack(side='left', padx=(0, 10))
        
        # Notas
        ttk.Label(feedback_frame, text="Observações (opcional):", style='Wood.TLabel').pack(anchor='w', pady=(10,5))
        self.feedback_notes = tk.Text(feedback_frame, height=3, width=50, font=('Segoe UI', 9))
        self.feedback_notes.pack(fill='x', pady=5)
        
        ttk.Button(feedback_frame, text="💾 Salvar Avaliação", 
                  command=self.salvar_avaliacao, style='Wood.TButton').pack(pady=10)
        
        # Botões de gerenciamento
        manage_frame = ttk.Frame(parent, style='Wood.TFrame')
        manage_frame.pack(fill='x', pady=20)
        
        ttk.Button(manage_frame, text="🔄 Atualizar Estatísticas", 
                  command=self.atualizar_estatisticas, style='Wood.TButton').pack(side='left', padx=(0, 10))
        
        ttk.Button(manage_frame, text="🗑️ Limpar Dados", 
                  command=self.limpar_dados_aprendizado, style='Wood.TButton').pack(side='left')
        
        self.atualizar_estatisticas()
        
    def create_parameter(self, parent, label, default, row):
        """Cria campo de parâmetro"""
        ttk.Label(parent, text=label, style='Wood.TLabel').grid(row=row, column=0, sticky='w', pady=8, padx=(0, 15))
        entry = ttk.Entry(parent, width=12, font=('Segoe UI', 10))
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky='w', pady=8)
        return entry
        
    def atualizar_descricao_modo(self, *args):
        """Atualiza descrição do modo de IA"""
        descricoes = {
            "wood_default": "💡 Processamento otimizado para entalhe em madeira",
            "sacred": "💡 Ideal para imagens religiosas - realça rostos e detalhes",
            "portrait": "💡 Otimizado para retratos - foco em características faciais",
            "landscape": "💡 Para paisagens - destaca texturas naturais", 
            "text": "💡 Maximiza legibilidade de textos",
            "art": "💡 Preserva estilo artístico"
        }
        self.modo_desc.config(text=descricoes.get(self.modo_processamento.get(), ""))
        
    def selecionar_imagem(self):
        """Seleciona arquivo de imagem"""
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para entalhe",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            self.status_label.config(text=f"Imagem carregada: {os.path.basename(caminho)}")
            
    def previsualizar(self):
        """Pré-visualização do processamento"""
        try:
            img_path = self.entry_imagem.get()
            if not img_path or not os.path.exists(img_path):
                messagebox.showwarning("Aviso", "Selecione uma imagem válida primeiro.")
                return
                
            # Carregar imagem para preview
            img = Image.open(img_path).convert("L")
            img_array = np.array(img) / 255.0
            
            # Mostrar preview simples
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_array, cmap='gray')
            plt.title('Imagem Original')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            z_preview = (1 - img_array) * 4.0
            plt.imshow(z_preview, cmap='terrain')
            plt.title('Preview do Entalhe')
            plt.axis('off')
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()
            
            self.status_label.config(text="Preview gerado")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro no preview: {str(e)}")
            
    def gerar_entalhe(self):
        """Gera o entalhe completo"""
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
                'safe_z': float(self.entry_safe_z.get()),
                'wood_type': self.tipo_madeira.get(),
                'grain_direction': self.direcao_veio.get(),
                'modo_ia': self.modo_processamento.get()
            }

            self.status_label.config(text="Processando imagem...")
            self.root.update()
            
            # Processar
            success = self.processar_entalhe_madeira(img_path, params)
            
            if success:
                self.status_label.config(text="✅ Entalhe gerado com sucesso!")
                messagebox.showinfo("Sucesso!", 
                                  "Entalhe em madeira gerado com sucesso!\n\n" +
                                  "Arquivos salvos na pasta 'WoodCarving_Output'\n\n" +
                                  "Avalie o resultado na aba 'Aprendizado'!")
            else:
                self.status_label.config(text="❌ Erro no processamento")
                
        except ValueError as e:
            self.status_label.config(text="❌ Erro nos parâmetros")
            messagebox.showerror("Erro", f"Verifique os valores:\n{str(e)}")
        except Exception as e:
            self.status_label.config(text="❌ Erro no processamento")
            messagebox.showerror("Erro", f"Falha:\n{str(e)}")

    def processar_entalhe_madeira(self, img_path, params):
        """Processamento completo do entalhe"""
        try:
            output_dir = os.path.join(os.getcwd(), "WoodCarving_Output")
            os.makedirs(output_dir, exist_ok=True)

            # Carregar imagem
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
            
            # PROCESSAMENTO COM IA
            modo_ia = params.get('modo_ia', 'wood_default')
            
            if modo_ia == 'wood_default':
                z_map = self.wood_ai.processar_para_madeira(
                    img_array, 
                    params['profundidade_max'],
                    params['wood_type'],
                    params['grain_direction']
                )
            else:
                z_map = self.wood_ai.advanced_processor.processar_imagem_avancado(
                    img_path,
                    params['profundidade_max'],
                    modo_ia
                )
                
                if z_map.shape != (new_height, new_width):
                    z_map = cv2.resize(z_map, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Garantir dados válidos
            z_map = np.nan_to_num(z_map, nan=0.0)
            z_map = np.clip(z_map, 0.0, params['profundidade_max'])
            
            # Salvar visualização
            self.salvar_visualizacao(img_array, z_map, output_dir, params)
            
            # Gerar G-code
            gcode_filename = f"entalhe_{datetime.now().strftime('%H%M%S')}.nc"
            gcode_path = os.path.join(output_dir, gcode_filename)
            params['output_path'] = gcode_path
            
            success = self.gcode_gen.gerar_gcode_madeira(z_map, params)
            
            # Salvar para feedback
            self.current_z_map = z_map
            self.current_params = params.copy()
            self.current_image_array = img_array
            
            return success

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            return False

    def salvar_visualizacao(self, img_original, z_map, output_dir, params):
        """Salva visualização do resultado"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Análise do Entalhe - {params["wood_type"].upper()}', fontsize=16)
            
            # Imagem original
            axes[0,0].imshow(img_original, cmap='gray')
            axes[0,0].set_title('Imagem Original')
            axes[0,0].axis('off')
            
            # Mapa de profundidade
            im = axes[0,1].imshow(z_map, cmap='terrain')
            axes[0,1].set_title('Mapa de Profundidade (mm)')
            axes[0,1].axis('off')
            plt.colorbar(im, ax=axes[0,1])
            
            # Visualização 3D
            try:
                from mpl_toolkits.mplot3d import Axes3D
                x = np.arange(z_map.shape[1])
                y = np.arange(z_map.shape[0])
                X, Y = np.meshgrid(x, y)
                
                ax3d = fig.add_subplot(2, 2, (3, 4), projection='3d')
                surf = ax3d.plot_surface(X, Y, z_map, cmap='terrain', alpha=0.8)
                ax3d.set_title('Visualização 3D')
            except:
                # Fallback se 3D não funcionar
                axes[1,0].imshow(z_map, cmap='terrain')
                axes[1,0].set_title('Mapa de Profundidade')
                axes[1,0].axis('off')
            
            # Histograma
            axes[1,1].hist(z_map.flatten(), bins=50, alpha=0.7, color='brown')
            axes[1,1].set_title('Distribuição de Profundidades')
            axes[1,1].set_xlabel('Profundidade (mm)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'analise_completa.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Erro ao salvar visualização: {e}")

    def salvar_avaliacao(self):
        """Salva avaliação do usuário para aprendizado"""
        if self.current_z_map is None:
            messagebox.showwarning("Aviso", "Gere um entalhe primeiro antes de avaliar.")
            return
            
        try:
            rating = self.feedback_rating.get()
            notes = self.feedback_notes.get("1.0", tk.END).strip()
            
            self.learning_system.add_feedback(
                self.current_image_array,
                self.current_params,
                self.current_z_map,
                rating,
                notes
            )
            
            messagebox.showinfo("Sucesso", f"Avaliação {rating}⭐ salva!\nO sistema está aprendendo...")
            
            # Limpar campos
            self.feedback_notes.delete("1.0", tk.END)
            self.atualizar_estatisticas()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar: {str(e)}")
            
    def atualizar_estatisticas(self):
        """Atualiza estatísticas do aprendizado"""
        stats = self.learning_system.get_learning_stats()
        
        text = f"""📊 Estatísticas do Aprendizado:

• Amostras de treinamento: {stats['total_samples']}
• Avaliação média: {stats['average_rating']:.1f} ⭐
• Modelo treinado: {'✅ Sim' if stats['model_trained'] else '❌ Não'}

💡 O sistema aprende com suas avaliações!"""
        
        self.stats_label.config(text=text)
        
    def limpar_dados_aprendizado(self):
        """Limpa dados de aprendizado"""
        if messagebox.askyesno("Confirmar", "Limpar todos os dados de aprendizado?"):
            self.learning_system.feedback_data = []
            self.learning_system.optimal_params = None
            try:
                if os.path.exists(self.learning_system.dataset_path):
                    os.remove(self.learning_system.dataset_path)
            except:
                pass
            self.atualizar_estatisticas()
            messagebox.showinfo("Sucesso", "Dados limpos!")

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
    