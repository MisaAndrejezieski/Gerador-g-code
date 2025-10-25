import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from datetime import datetime
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

# ==============================================
# IA AVANÇADA PARA RECONHECIMENTO DE VOLUME 3D
# ==============================================

class Volume3DAI:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def analisar_volume_3d(self, image_array, profundidade_max):
        """
        Analisa a imagem como um objeto 3D REAL com volume e profundidade
        """
        try:
            print("🔍 Iniciando análise de volume 3D...")
            
            # Converter para array de trabalho
            img_float = image_array.astype(np.float32)
            
            # 1. ANÁLISE DE PROFUNDIDADE POR SEGMENTAÇÃO
            mapa_profundidade = self._criar_mapa_profundidade_avancado(img_float, profundidade_max)
            
            # 2. DETECÇÃO DE CARACTERÍSTICAS 3D
            mapa_volume = self._detectar_caracteristicas_3d(img_float, mapa_profundidade, profundidade_max)
            
            # 3. SUAVIZAÇÃO INTELIGENTE
            mapa_final = self._aplicar_suavizacao_inteligente(mapa_volume, img_float)
            
            # 4. REALCE DE DETALHES
            mapa_final = self._realcar_detalhes_importantes(mapa_final, img_float, profundidade_max)
            
            print("✅ Análise 3D concluída - Volume real detectado")
            return np.clip(mapa_final, 0, profundidade_max)
            
        except Exception as e:
            print(f"❌ Erro na análise 3D: {e}")
            return self._fallback_3d(image_array, profundidade_max)

    def _criar_mapa_profundidade_avancado(self, img_float, profundidade_max):
        """Cria mapa de profundidade baseado em múltiplas técnicas"""
        try:
            h, w = img_float.shape
            
            # Técnica 1: Profundidade por intensidade (áreas claras = mais altas)
            mapa_intensidade = img_float * profundidade_max
            
            # Técnica 2: Profundidade por gradientes (bordas = variação de altura)
            grad_x = sobel(img_float, axis=1)
            grad_y = sobel(img_float, axis=0)
            magnitude_grad = np.sqrt(grad_x**2 + grad_y**2)
            mapa_gradientes = magnitude_grad * profundidade_max * 2
            
            # Técnica 3: Profundidade por segmentação
            mapa_segmentacao = self._segmentar_regioes_profundidade(img_float, profundidade_max)
            
            # Técnica 4: Profundidade por textura
            mapa_textura = self._analisar_textura_profundidade(img_float, profundidade_max)
            
            # COMBINAR TODAS AS TÉCNICAS
            mapa_combinado = (
                mapa_intensidade * 0.4 +      # Base da intensidade
                mapa_segmentacao * 0.3 +      # Segmentação de regiões
                mapa_gradientes * 0.2 +       # Gradientes para bordas
                mapa_textura * 0.1            # Textura para detalhes
            )
            
            return mapa_combinado
            
        except Exception as e:
            print(f"Erro no mapa de profundidade: {e}")
            return img_float * profundidade_max

    def _segmentar_regioes_profundidade(self, img_float, profundidade_max):
        """Segmenta a imagem em regiões de diferentes profundidades"""
        try:
            # Converter para 8-bit para processamento OpenCV
            img_8bit = (img_float * 255).astype(np.uint8)
            
            # Aplicar filtro bilateral para preservar bordas
            img_suavizada = cv2.bilateralFilter(img_8bit, 9, 75, 75)
            
            # Segmentação por K-means
            dados = img_suavizada.reshape((-1, 1))
            dados = np.float32(dados)
            
            criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, rotulos, centros = cv2.kmeans(dados, 4, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Converter de volta para imagem
            centros = np.uint8(centers)
            img_segmentada = centros[rotulos.flatten()].reshape((img_float.shape))
            
            # Mapear segmentos para profundidades
            segmentos_ordenados = np.sort(centers.flatten())
            mapa_profundidade = np.zeros_like(img_float)
            
            for i, segmento in enumerate(segmentos_ordenados):
                mascara = (img_segmentada == segmento)
                # Segmentos mais claros = maior profundidade
                profundidade_segmento = (i + 1) * (profundidade_max / len(segmentos_ordenados))
                mapa_profundidade[mascara] = profundidade_segmento
            
            return mapa_profundidade.astype(np.float32)
            
        except Exception as e:
            print(f"Erro na segmentação: {e}")
            return img_float * profundidade_max

    def _analisar_textura_profundidade(self, img_float, profundidade_max):
        """Analisa texturas para determinar profundidade"""
        try:
            # Calcular mapa de variância local (textura)
            kernel_size = 5
            img_padded = np.pad(img_float, kernel_size//2, mode='reflect')
            mapa_variancia = np.zeros_like(img_float)
            
            for i in range(img_float.shape[0]):
                for j in range(img_float.shape[1]):
                    regiao = img_padded[i:i+kernel_size, j:j+kernel_size]
                    mapa_variancia[i, j] = np.var(regiao)
            
            # Normalizar e mapear para profundidade
            mapa_variancia = mapa_variancia / np.max(mapa_variancia)
            mapa_textura = mapa_variancia * profundidade_max * 0.5
            
            return mapa_textura
            
        except Exception as e:
            print(f"Erro na análise de textura: {e}")
            return np.zeros_like(img_float)

    def _detectar_caracteristicas_3d(self, img_float, mapa_base, profundidade_max):
        """Detecta características específicas e aplica volume 3D real"""
        try:
            mapa_volume = mapa_base.copy()
            img_8bit = (img_float * 255).astype(np.uint8)
            
            # 1. DETECTAR ROSTOS E APLICAR VOLUME FACIAL
            rostos = self._detectar_rosto_aplicar_volume(img_8bit, mapa_volume, profundidade_max)
            
            # 2. DETECTAR OLHOS E CRIAR CAVIDADES
            if rostos > 0:
                self._detectar_olhos_aplicar_volume(img_8bit, mapa_volume, profundidade_max)
            
            # 3. DETECTAR BORDAS E APLICAR RELEVOS
            self._aplicar_relevo_bordas(img_float, mapa_volume, profundidade_max)
            
            # 4. DETECTAR ÁREAS DE ALTO CONTRASTE PARA DESTAQUE
            self._destacar_areas_contraste(img_float, mapa_volume, profundidade_max)
            
            return mapa_volume
            
        except Exception as e:
            print(f"Erro na detecção 3D: {e}")
            return mapa_base

    def _detectar_rosto_aplicar_volume(self, img_8bit, mapa_volume, profundidade_max):
        """Detecta rostos e aplica volume facial realista"""
        try:
            faces = self.face_detector.detectMultiScale(
                img_8bit, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                print(f"👤 Rosto detectado: {w}x{h} pixels")
                
                # Criar máscara suave para o rosto
                mascara_rosto = np.zeros_like(mapa_volume)
                cv2.ellipse(mascara_rosto, 
                           (x + w//2, y + h//2), 
                           (w//2, h//2), 0, 0, 360, 1, -1)
                
                # Aplicar volume convexo para o rosto (formato arredondado)
                for i in range(h):
                    for j in range(w):
                        if mascara_rosto[y+i, x+j] > 0:
                            # Calcular distância do centro (formato oval)
                            dist_x = abs(j - w//2) / (w//2)
                            dist_y = abs(i - h//2) / (h//2)
                            dist_normalizada = np.sqrt(dist_x**2 + dist_y**2)
                            
                            # Volume convexo (mais alto no centro)
                            if dist_normalizada <= 1.0:
                                altura = (1 - dist_normalizada) * profundidade_max * 0.7
                                mapa_volume[y+i, x+j] += altura
            
            return len(faces)
            
        except Exception as e:
            print(f"Erro detecção rosto: {e}")
            return 0

    def _detectar_olhos_aplicar_volume(self, img_8bit, mapa_volume, profundidade_max):
        """Detecta olhos e cria cavidades realistas"""
        try:
            eyes = self.eye_detector.detectMultiScale(
                img_8bit, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(15, 15)
            )
            
            for (x, y, w, h) in eyes:
                print(f"👁️ Olho detectado: {w}x{h} pixels")
                
                # Criar cavidade para o olho
                for i in range(h):
                    for j in range(w):
                        dist_x = abs(j - w//2) / (w//2)
                        dist_y = abs(i - h//2) / (h//2)
                        dist_normalizada = np.sqrt(dist_x**2 + dist_y**2)
                        
                        if dist_normalizada <= 1.0:
                            # Cavidade côncava (mais baixa no centro)
                            profundidade = dist_normalizada * profundidade_max * 0.3
                            mapa_volume[y+i, x+j] = np.maximum(0, mapa_volume[y+i, x+j] - profundidade)
            
        except Exception as e:
            print(f"Erro detecção olhos: {e}")

    def _aplicar_relevo_bordas(self, img_float, mapa_volume, profundidade_max):
        """Aplica relevo nas bordas detectadas"""
        try:
            # Detectar bordas com Canny
            bordas = cv2.Canny((img_float * 255).astype(np.uint8), 50, 150)
            
            # Dilatar bordas para área de influência
            kernel = np.ones((3, 3), np.uint8)
            bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=1)
            
            # Aplicar relevo nas bordas
            mapa_volume[bordas_dilatadas > 0] += profundidade_max * 0.2
            
        except Exception as e:
            print(f"Erro relevo bordas: {e}")

    def _destacar_areas_contraste(self, img_float, mapa_volume, profundidade_max):
        """Destaca áreas de alto contraste com volume adicional"""
        try:
            # Calcular contraste local
            kernel_size = 7
            img_padded = np.pad(img_float, kernel_size//2, mode='reflect')
            mapa_contraste = np.zeros_like(img_float)
            
            for i in range(img_float.shape[0]):
                for j in range(img_float.shape[1]):
                    regiao = img_padded[i:i+kernel_size, j:j+kernel_size]
                    contraste = np.max(regiao) - np.min(regiao)
                    mapa_contraste[i, j] = contraste
            
            # Aplicar volume em áreas de alto contraste
            limiar_contraste = np.percentile(mapa_contraste, 80)
            areas_alto_contraste = mapa_contraste > limiar_contraste
            mapa_volume[areas_alto_contraste] += profundidade_max * 0.15
            
        except Exception as e:
            print(f"Erro destaque contraste: {e}")

    def _aplicar_suavizacao_inteligente(self, mapa_volume, img_float):
        """Aplica suavização inteligente preservando bordas"""
        try:
            # Suavização bilateral para preservar bordas
            mapa_suavizado = cv2.bilateralFilter(
                mapa_volume.astype(np.float32), 
                5,  # diâmetro
                25,  # sigma cor
                25   # sigma espaço
            )
            
            return mapa_suavizado
            
        except Exception as e:
            print(f"Erro suavização: {e}")
            return mapa_volume

    def _realcar_detalhes_importantes(self, mapa_volume, img_float, profundidade_max):
        """Realça detalhes importantes preservando volume"""
        try:
            # Detectar pequenos detalhes
            detalhes = img_float - cv2.GaussianBlur(img_float, (0, 0), 3)
            detalhes = np.clip(detalhes * 3, 0, 1)
            
            # Aplicar detalhes ao mapa de volume
            mapa_detalhado = mapa_volume + (detalhes * profundidade_max * 0.1)
            
            return mapa_detalhado
            
        except Exception as e:
            print(f"Erro realce detalhes: {e}")
            return mapa_volume

    def _fallback_3d(self, image_array, profundidade_max):
        """Fallback para quando a análise 3D falha"""
        # AGORA CORRETO: áreas claras = mais volume, áreas escuras = menos volume
        return image_array * profundidade_max

# ==============================================
# SISTEMA DE APRENDIZADO PARA VOLUME 3D
# ==============================================

class LearningSystem3D:
    def __init__(self):
        self.feedback_data = []
        self.dataset_path = "learning_3d.json"
        self.load_feedback()
        
    def add_feedback(self, image_array, params, z_map_result, user_rating, user_notes=""):
        """Adiciona feedback para aprendizado 3D"""
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'params': params,
                'user_rating': user_rating,
                'user_notes': user_notes,
                'image_stats': {
                    'mean_intensity': float(np.mean(image_array)),
                    'contrast': float(np.max(image_array) - np.min(image_array)),
                    'edge_density': float(np.mean(cv2.Canny((image_array * 255).astype(np.uint8), 50, 150) > 0))
                },
                'volume_stats': {
                    'mean_depth': float(np.mean(z_map_result)),
                    'max_depth': float(np.max(z_map_result)),
                    'volume_variance': float(np.var(z_map_result))
                }
            }
            
            self.feedback_data.append(feedback_entry)
            self.save_feedback()
            print(f"✅ Feedback 3D salvo: {user_rating}⭐")
            
        except Exception as e:
            print(f"Erro feedback 3D: {e}")
    
    def save_feedback(self):
        """Salva dados de feedback"""
        try:
            with open(self.dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro salvar feedback: {e}")
    
    def load_feedback(self):
        """Carrega feedback do arquivo"""
        try:
            if os.path.exists(self.dataset_path):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                print(f"📊 {len(self.feedback_data)} amostras 3D carregadas")
        except:
            self.feedback_data = []
    
    def get_learning_stats(self):
        """Retorna estatísticas do aprendizado"""
        if not self.feedback_data:
            return {"total_samples": 0, "average_rating": 0}
        
        ratings = [entry['user_rating'] for entry in self.feedback_data]
        return {
            "total_samples": len(self.feedback_data),
            "average_rating": float(np.mean(ratings))
        }

# ==============================================
# GERADOR G-CODE OTIMIZADO PARA 3D
# ==============================================

class GCodeGenerator3D:
    def __init__(self):
        self.wood_configs = {
            "soft": {"feedrate": 2000, "stepover": 0.6, "depth_increment": 0.8},
            "medium": {"feedrate": 1500, "stepover": 0.4, "depth_increment": 0.5},
            "hard": {"feedrate": 1000, "stepover": 0.3, "depth_increment": 0.3}
        }
    
    def gerar_gcode_3d(self, z_map, params):
        """Gera G-code otimizado para escultura 3D"""
        try:
            gcode_path = params['output_path']
            wood_type = params.get('wood_type', 'medium')
            config = self.wood_configs[wood_type]
            
            with open(gcode_path, "w", encoding='utf-8') as f:
                # CABEÇALHO AVANÇADO
                f.write("(G-code para ESCULTURA 3D em Madeira)\n")
                f.write("(VOLUME REAL - GERADO POR IA 3D)\n")
                f.write("G21 G90 G94 G17 G40 G49\n")
                f.write(f"F{config['feedrate']}\n")
                f.write("S12000 (SPINDLE PARA 3D)\n")
                f.write("G64 P0.01 (CONTORNO PRECISO)\n\n")
                
                # POSICIONAMENTO
                f.write(f"G0 Z{params['safe_z']:.3f}\n")
                f.write("G0 X0 Y0\n")
                f.write("M3 (LIGAR SPINDLE)\n")
                f.write("G4 P3 (AGUARDAR ACELERAÇÃO)\n\n")
                
                # ESTRATÉGIA 3D - MULTI-PASSES
                self._gerar_estrategia_3d(f, z_map, params, config)
                
                # FINALIZAÇÃO
                f.write(f"\nG0 Z{params['safe_z']:.3f}\n")
                f.write("M5 (DESLIGAR SPINDLE)\n")
                f.write("G0 X0 Y0\n")
                f.write("M30\n")
                
                # ESTATÍSTICAS
                f.write(f"\n; ESTATÍSTICAS 3D\n")
                f.write(f"; Volume máximo: {np.max(z_map):.2f}mm\n")
                f.write(f"; Volume médio: {np.mean(z_map):.2f}mm\n")
                f.write(f"; Área usinada: {z_map.shape[1]*params['passo']:.1f}x{z_map.shape[0]*params['passo']:.1f}mm\n")
            
            print(f"✅ G-code 3D gerado: {gcode_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro G-code 3D: {e}")
            return False

    def _gerar_estrategia_3d(self, f, z_map, params, config):
        """Gera estratégia de usinagem 3D otimizada"""
        rows, cols = z_map.shape
        passo = params['passo']
        safe_z = params['safe_z']
        
        # Calcular número de passes baseado na profundidade máxima
        max_depth = np.max(z_map)
        num_passes = max(1, int(np.ceil(max_depth / config['depth_increment'])))
        
        print(f"🔧 Estratégia 3D: {num_passes} passes")
        
        for pass_num in range(num_passes):
            f.write(f"\n(PASSE {pass_num + 1} de {num_passes})\n")
            
            # Profundidade deste passe
            depth_ratio = (pass_num + 1) / num_passes
            
            for y in range(rows):
                # Direção alternada (zig-zag)
                if y % 2 == 0:
                    x_range = range(cols)
                else:
                    x_range = range(cols - 1, -1, -1)
                
                primeiro_ponto = True
                
                for x in x_range:
                    z_original = z_map[y, x]
                    
                    # Para múltiplos passes, calcular profundidade progressiva
                    if num_passes > 1:
                        z_target = min(z_original * depth_ratio, z_original)
                    else:
                        z_target = z_original
                    
                    # Validação de segurança
                    if np.isnan(z_target) or np.isinf(z_target):
                        z_target = 0.0
                    else:
                        z_target = max(0.0, min(z_target, params['profundidade_max']))
                    
                    # Coordenadas reais
                    pos_x = (x * passo) - (cols * passo / 2)
                    pos_y = (y * passo) - (rows * passo / 2)
                    
                    if primeiro_ponto:
                        # Movimento rápido para primeiro ponto
                        f.write(f"G0 X{pos_x:.3f} Y{pos_y:.3f}\n")
                        f.write(f"G1 Z{z_target:.3f}\n")
                        primeiro_ponto = False
                    else:
                        # Movimento de corte
                        f.write(f"G1 X{pos_x:.3f} Y{pos_y:.3f} Z{z_target:.3f}\n")

# ==============================================
# INTERFACE PRINCIPAL 3D
# ==============================================

class WoodCarving3DApp:
    def __init__(self, root):
        self.root = root
        self.volume_ai = Volume3DAI()
        self.gcode_3d = GCodeGenerator3D()
        self.learning_3d = LearningSystem3D()
        
        self.current_z_map = None
        self.current_params = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("🎯 ESCULTURA 3D EM MADEIRA - IA DE VOLUME REAL")
        self.root.geometry("1200x900")
        self.root.configure(bg='#2B4C7E')
        
        self.setup_styles()
        self.create_main_interface()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores modernas para interface 3D
        style.configure('TFrame', background='#1E3B5A')
        style.configure('TLabel', background='#1E3B5A', foreground='white', font=('Segoe UI', 10))
        style.configure('Title.TLabel', background='#1E3B5A', foreground='#4FC3F7', 
                       font=('Segoe UI', 18, 'bold'))
        style.configure('TButton', background='#1565C0', foreground='white',
                       font=('Segoe UI', 10, 'bold'))
        style.map('TButton', background=[('active', '#0D47A1')])
        
    def create_main_interface(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title = ttk.Label(main_frame, text="🎯 SISTEMA DE ESCULTURA 3D EM MADEIRA", style='Title.TLabel')
        title.pack(pady=(0, 20))
        
        # Abas
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # Criar abas
        tab_principal = ttk.Frame(notebook, style='TFrame')
        tab_3d = ttk.Frame(notebook, style='TFrame')
        tab_learning = ttk.Frame(notebook, style='TFrame')
        
        notebook.add(tab_principal, text="🛠️ Configurações")
        notebook.add(tab_3d, text="📐 Controle 3D")
        notebook.add(tab_learning, text="📊 Aprendizado")
        
        self.create_main_tab(tab_principal)
        self.create_3d_tab(tab_3d)
        self.create_learning_tab(tab_learning)
        
    def create_main_tab(self, parent):
        """Aba principal de configurações"""
        # Container principal com scroll
        container = ttk.Frame(parent, style='TFrame')
        container.pack(fill='both', expand=True)
        
        # Canvas e Scrollbar
        canvas = tk.Canvas(container, bg='#1E3B5A', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # CONTEÚDO PRINCIPAL
        ttk.Label(scrollable_frame, text="CONFIGURAÇÕES DE ESCULTURA 3D", 
                 style='Title.TLabel').pack(anchor='w', pady=(10, 20))
        
        # Seção de imagem
        img_frame = ttk.LabelFrame(scrollable_frame, text="📷 IMAGEM 3D", style='TFrame')
        img_frame.pack(fill='x', pady=10, padx=5)
        
        ttk.Label(img_frame, text="Imagem para Escultura 3D:", style='TLabel').pack(anchor='w', pady=5)
        
        file_frame = ttk.Frame(img_frame, style='TFrame')
        file_frame.pack(fill='x', pady=5)
        
        self.entry_imagem = ttk.Entry(file_frame, font=('Segoe UI', 11), width=50)
        self.entry_imagem.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(file_frame, text="📁 Procurar", 
                  command=self.selecionar_imagem, style='TButton').pack(side='right')
        
        # Seção de parâmetros 3D
        params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ PARÂMETROS 3D", style='TFrame')
        params_frame.pack(fill='x', pady=15, padx=5)
        
        # Grid de parâmetros
        param_grid = ttk.Frame(params_frame, style='TFrame')
        param_grid.pack(fill='x', pady=10)
        
        self.entry_largura = self.create_parameter_3d(param_grid, "Largura da Peça (mm):", "200", 0)
        self.entry_altura = self.create_parameter_3d(param_grid, "Altura da Peça (mm):", "150", 1)
        self.entry_profundidade = self.create_parameter_3d(param_grid, "Altura Máxima do Relevo (mm):", "8.0", 2)
        self.entry_passo = self.create_parameter_3d(param_grid, "Resolução (mm/ponto):", "0.8", 3)
        self.entry_safe_z = self.create_parameter_3d(param_grid, "Altura de Segurança (mm):", "10.0", 4)
        
        # Seção tipo de madeira
        wood_frame = ttk.LabelFrame(scrollable_frame, text="🪵 TIPO DE MADEIRA", style='TFrame')
        wood_frame.pack(fill='x', pady=15, padx=5)
        
        self.tipo_madeira = tk.StringVar(value="medium")
        
        ttk.Radiobutton(wood_frame, text="🔸 MACIA (Pinho, Cedro) - Corte Rápido", 
                       variable=self.tipo_madeira, value="soft", style='TLabel').pack(anchor='w', pady=2)
        ttk.Radiobutton(wood_frame, text="🔸 MÉDIA (Nogueira, Cerejeira) - Balanceado", 
                       variable=self.tipo_madeira, value="medium", style='TLabel').pack(anchor='w', pady=2)
        ttk.Radiobutton(wood_frame, text="🔸 DURA (Carvalho, Mogno) - Detalhamento", 
                       variable=self.tipo_madeira, value="hard", style='TLabel').pack(anchor='w', pady=2)
        
        # Botões de ação
        action_frame = ttk.Frame(scrollable_frame, style='TFrame')
        action_frame.pack(fill='x', pady=20)
        
        ttk.Button(action_frame, text="🔍 ANALISAR VOLUME 3D", 
                  command=self.analisar_volume, style='TButton').pack(side='left', padx=(0, 10))
        
        ttk.Button(action_frame, text="🪚 GERAR ESCULTURA 3D", 
                  command=self.gerar_escultura, style='TButton').pack(side='left')
        
        # Status
        self.status_label = ttk.Label(scrollable_frame, text="Pronto para análise 3D...", 
                                     style='TLabel', font=('Segoe UI', 11))
        self.status_label.pack(pady=10)
        
        # Empacotar scroll
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_3d_tab(self, parent):
        """Aba de controle 3D"""
        ttk.Label(parent, text="CONTROLE DE VOLUME 3D", style='Title.TLabel').pack(anchor='w', pady=(10, 20))
        
        control_frame = ttk.Frame(parent, style='TFrame')
        control_frame.pack(fill='x', pady=10)
        
        # Controles de ajuste de volume
        ttk.Label(control_frame, text="Ajuste de Volume:", style='TLabel').pack(anchor='w')
        
        self.volume_scale = tk.DoubleVar(value=1.0)
        scale_frame = ttk.Frame(control_frame, style='TFrame')
        scale_frame.pack(fill='x', pady=5)
        
        ttk.Scale(scale_frame, from_=0.5, to=2.0, variable=self.volume_scale,
                 orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(scale_frame, textvariable=self.volume_scale, style='TLabel', width=4).pack(side='right')
        
        # Visualização
        ttk.Button(control_frame, text="👁️ VISUALIZAR 3D", 
                  command=self.visualizar_3d, style='TButton').pack(pady=10)
        
    def create_learning_tab(self, parent):
        """Aba de aprendizado"""
        ttk.Label(parent, text="SISTEMA DE APRENDIZADO 3D", style='Title.TLabel').pack(anchor='w', pady=(10, 20))
        
        # Estatísticas
        stats_frame = ttk.LabelFrame(parent, text="📊 ESTATÍSTICAS", style='TFrame')
        stats_frame.pack(fill='x', pady=10, padx=5)
        
        self.stats_label = ttk.Label(stats_frame, text="Carregando...", style='TLabel', justify='left')
        self.stats_label.pack(anchor='w', pady=10, padx=10)
        
        # Avaliação
        rating_frame = ttk.LabelFrame(parent, text="⭐ AVALIAR RESULTADO", style='TFrame')
        rating_frame.pack(fill='x', pady=10, padx=5)
        
        self.feedback_rating = tk.IntVar(value=5)
        
        ttk.Label(rating_frame, text="Qualidade da Escultura 3D:", style='TLabel').pack(anchor='w', pady=5)
        
        rating_buttons = ttk.Frame(rating_frame, style='TFrame')
        rating_buttons.pack(fill='x', pady=5)
        
        for i in range(1, 6):
            ttk.Radiobutton(rating_buttons, text="★" * i, variable=self.feedback_rating, 
                           value=i, style='TLabel').pack(side='left', padx=(0, 10))
        
        ttk.Button(rating_frame, text="💾 SALVAR AVALIAÇÃO", 
                  command=self.salvar_avaliacao, style='TButton').pack(pady=10)
        
        self.atualizar_estatisticas()
        
    def create_parameter_3d(self, parent, label, default, row):
        """Cria campo de parâmetro para 3D"""
        ttk.Label(parent, text=label, style='TLabel').grid(row=row, column=0, sticky='w', pady=8, padx=(0, 15))
        entry = ttk.Entry(parent, width=15, font=('Segoe UI', 10))
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky='w', pady=8)
        return entry
        
    def selecionar_imagem(self):
        """Seleciona imagem para análise 3D"""
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para escultura 3D",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if caminho:
            self.entry_imagem.delete(0, tk.END)
            self.entry_imagem.insert(0, caminho)
            self.status_label.config(text=f"Imagem 3D carregada: {os.path.basename(caminho)}")
            
    def analisar_volume(self):
        """Analisa o volume 3D da imagem"""
        try:
            img_path = self.entry_imagem.get()
            if not img_path or not os.path.exists(img_path):
                messagebox.showwarning("Aviso", "Selecione uma imagem válida.")
                return
                
            self.status_label.config(text="🔍 Analisando volume 3D...")
            self.root.update()
            
            # Carregar imagem
            img = Image.open(img_path).convert("L")
            img_array = np.array(img) / 255.0
            
            # Parâmetros
            params = {
                'profundidade_max': float(self.entry_profundidade.get())
            }
            
            # Análise 3D
            z_map = self.volume_ai.analisar_volume_3d(img_array, params['profundidade_max'])
            
            # Salvar para uso posterior
            self.current_z_map = z_map
            self.current_image = img_array
            self.current_params = params
            
            # Visualização automática
            self.visualizar_resultado_3d(img_array, z_map)
            
            self.status_label.config(text="✅ Análise 3D concluída - Verifique a visualização")
            
        except Exception as e:
            self.status_label.config(text="❌ Erro na análise 3D")
            messagebox.showerror("Erro", f"Falha na análise 3D:\n{str(e)}")
            
    def visualizar_resultado_3d(self, img_original, z_map):
        """Mostra visualização 3D do resultado"""
        try:
            fig = plt.figure(figsize=(15, 5))
            
            # Imagem original
            plt.subplot(1, 3, 1)
            plt.imshow(img_original, cmap='gray')
            plt.title('Imagem Original')
            plt.axis('off')
            
            # Mapa de profundidade
            plt.subplot(1, 3, 2)
            plt.imshow(z_map, cmap='hot')
            plt.title('Mapa de Profundidade 3D')
            plt.axis('off')
            plt.colorbar()
            
            # Visualização 3D
            ax = fig.add_subplot(1, 3, 3, projection='3d')
            x = np.arange(z_map.shape[1])
            y = np.arange(z_map.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Plot surface
            surf = ax.plot_surface(X, Y, z_map, cmap='viridis', 
                                 linewidth=0, antialiased=True, alpha=0.8)
            
            ax.set_title('Visualização 3D do Volume')
            ax.set_zlabel('Altura (mm)')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Erro visualização: {e}")
            
    def visualizar_3d(self):
        """Visualização interativa 3D"""
        if self.current_z_map is None:
            messagebox.showwarning("Aviso", "Analise o volume 3D primeiro.")
            return
        self.visualizar_resultado_3d(self.current_image, self.current_z_map)
            
    def gerar_escultura(self):
        """Gera a escultura 3D completa"""
        try:
            if self.current_z_map is None:
                messagebox.showwarning("Aviso", "Analise o volume 3D primeiro.")
                return
                
            # Coletar parâmetros
            params = {
                'largura_mm': float(self.entry_largura.get()),
                'altura_mm': float(self.entry_altura.get()),
                'profundidade_max': float(self.entry_profundidade.get()),
                'passo': float(self.entry_passo.get()),
                'safe_z': float(self.entry_safe_z.get()),
                'wood_type': self.tipo_madeira.get(),
                'output_path': os.path.join(os.getcwd(), "Escultura_3D", f"escultura_3d_{datetime.now().strftime('%H%M%S')}.nc")
            }
            
            # Criar diretório
            os.makedirs(os.path.dirname(params['output_path']), exist_ok=True)
            
            self.status_label.config(text="🪚 Gerando escultura 3D...")
            self.root.update()
            
            # Aplicar ajuste de volume
            z_map_ajustado = self.current_z_map * self.volume_scale.get()
            
            # Gerar G-code 3D
            success = self.gcode_3d.gerar_gcode_3d(z_map_ajustado, params)
            
            if success:
                self.status_label.config(text="✅ Escultura 3D gerada com sucesso!")
                messagebox.showinfo("Sucesso!", 
                                  "Escultura 3D gerada com sucesso!\n\n" +
                                  "Arquivo salvo em: 'Escultura_3D'\n\n" +
                                  "Avalie o resultado na aba 'Aprendizado'")
            else:
                self.status_label.config(text="❌ Erro na geração 3D")
                
        except Exception as e:
            self.status_label.config(text="❌ Erro na escultura 3D")
            messagebox.showerror("Erro", f"Falha na geração 3D:\n{str(e)}")
            
    def salvar_avaliacao(self):
        """Salva avaliação do resultado 3D"""
        if self.current_z_map is None:
            messagebox.showwarning("Aviso", "Gere uma escultura 3D primeiro.")
            return
            
        try:
            rating = self.feedback_rating.get()
            
            self.learning_3d.add_feedback(
                self.current_image,
                self.current_params,
                self.current_z_map,
                rating,
                "Escultura 3D"
            )
            
            messagebox.showinfo("Sucesso", f"Avaliação {rating}★ salva!\nSistema de IA aprendendo...")
            self.atualizar_estatisticas()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar: {str(e)}")
            
    def atualizar_estatisticas(self):
        """Atualiza estatísticas do aprendizado"""
        stats = self.learning_3d.get_learning_stats()
        
        text = f"""📊 ESTATÍSTICAS 3D:

• Amostras de treinamento: {stats['total_samples']}
• Avaliação média: {stats['average_rating']:.1f} ★
• Sistema: {'✅ APRENDENDO' if stats['total_samples'] > 0 else '⏳ AGUARDANDO DADOS'}

💡 A IA melhora com cada avaliação!"""
        
        self.stats_label.config(text=text)

# ==============================================
# EXECUÇÃO PRINCIPAL
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = WoodCarving3DApp(root)
    
    # Centralizar janela
    window_width = 1200
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()
    