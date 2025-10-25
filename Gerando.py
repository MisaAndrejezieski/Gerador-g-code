import json
import time
from datetime import datetime
import math
from scipy import signal
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches

# ==============================================
# ANALISADOR DE VIABILIDADE PARA MADEIRA
# ==============================================

class WoodFeasibilityAnalyzer:
    def __init__(self):
        self.wood_properties = {
            "soft": {
                "max_detail": 0.8,    # Detalhamento máximo (0-1)
                "max_depth": 10.0,    # Profundidade máxima (mm)
                "min_feature_size": 1.5,  # Tamanho mínimo de característica (mm)
                "recommended_bits": ["V-bit 60°", "Ballnose 3mm"]
            },
            "medium": {
                "max_detail": 0.9,
                "max_depth": 8.0,
                "min_feature_size": 1.0,
                "recommended_bits": ["V-bit 90°", "Ballnose 2mm", "Endmill 1mm"]
            },
            "hard": {
                "max_detail": 0.95,
                "max_depth": 6.0,
                "min_feature_size": 0.5,
                "recommended_bits": ["V-bit 120°", "Ballnose 1mm", "Endmill 0.5mm"]
            }
        }
    
    def analisar_viabilidade(self, z_map, params, wood_type):
        """Analisa se o design é viável para o tipo de madeira selecionado"""
        analysis = {
            "viable": True,
            "warnings": [],
            "recommendations": [],
            "statistics": {}
        }
        
        props = self.wood_properties[wood_type]
        
        # Estatísticas básicas
        analysis["statistics"]["max_depth"] = np.max(z_map)
        analysis["statistics"]["min_depth"] = np.min(z_map)
        analysis["statistics"]["avg_depth"] = np.mean(z_map)
        analysis["statistics"]["depth_range"] = np.max(z_map) - np.min(z_map)
        
        # Verificar profundidade máxima
        if analysis["statistics"]["max_depth"] > props["max_depth"]:
            analysis["warnings"].append(
                f"Profundidade máxima ({analysis['statistics']['max_depth']:.1f}mm) "
                f"excede recomendação para {wood_type} ({props['max_depth']}mm)"
            )
            analysis["viable"] = False
        
        # Analisar detalhes finos
        detail_analysis = self._analisar_detalhes_finos(z_map, params['passo'])
        analysis["statistics"].update(detail_analysis)
        
        if detail_analysis["smallest_feature"] < props["min_feature_size"]:
            analysis["warnings"].append(
                f"Detalhes muito finos ({detail_analysis['smallest_feature']:.1f}mm) "
                f"para madeira {wood_type} (mínimo: {props['min_feature_size']}mm)"
            )
        
        # Verificar inclinações íngremes
        slope_analysis = self._analisar_inclinacoes(z_map, params['passo'])
        analysis["statistics"].update(slope_analysis)
        
        if slope_analysis["max_slope"] > 75:  # graus
            analysis["warnings"].append(
                f"Inclinações muito íngremes ({slope_analysis['max_slope']:.1f}°) "
                f"podem causar fraturas na madeira"
            )
        
        # Gerar recomendações
        analysis["recommendations"] = self._gerar_recomendacoes(analysis, props, wood_type)
        
        return analysis
    
    def _analisar_detalhes_finos(self, z_map, passo):
        """Analisa a presença de detalhes muito finos"""
        # Calcular gradientes para identificar bordas finas
        grad_x = np.gradient(z_map, axis=1)
        grad_y = np.gradient(z_map, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identificar características pequenas
        threshold = np.percentile(grad_magnitude, 95)
        high_detail_mask = grad_magnitude > threshold
        
        # Estimar tamanho das menores características
        if np.any(high_detail_mask):
            labeled_array, num_features = ndimage.label(high_detail_mask)
            feature_sizes = []
            
            for i in range(1, num_features + 1):
                feature_mask = labeled_array == i
                feature_size = np.sum(feature_mask) * passo
                if feature_size > 0:
                    feature_sizes.append(feature_size)
            
            smallest_feature = min(feature_sizes) if feature_sizes else 0
        else:
            smallest_feature = float('inf')
        
        return {
            "smallest_feature": smallest_feature,
            "detail_density": np.mean(high_detail_mask),
            "gradient_variance": np.var(grad_magnitude)
        }
    
    def _analisar_inclinacoes(self, z_map, passo):
        """Analisa inclinações no mapa de profundidade"""
        grad_x = np.gradient(z_map, axis=1) / passo
        grad_y = np.gradient(z_map, axis=0) / passo
        
        slopes_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slopes_deg = np.degrees(slopes_rad)
        
        return {
            "max_slope": np.max(slopes_deg),
            "avg_slope": np.mean(slopes_deg),
            "steep_area_ratio": np.mean(slopes_deg > 45)
        }
    
    def _gerar_recomendacoes(self, analysis, props, wood_type):
        """Gera recomendações baseadas na análise"""
        recommendations = []
        stats = analysis["statistics"]
        
        # Recomendações de ferramentas
        recommendations.append(f"Ferramentas recomendadas: {', '.join(props['recommended_bits'])}")
        
        # Recomendações de estratégia
        if stats["depth_range"] > props["max_depth"] * 0.7:
            recommendations.append("Considere usar múltiplos passes para profundidades variadas")
        
        if stats["detail_density"] > 0.3:
            recommendations.append("Alta densidade de detalhes - use ferramentas menores e velocidade reduzida")
        
        if stats["steep_area_ratio"] > 0.2:
            recommendations.append("Áreas íngremes detectadas - considere suavizar inclinações")
        
        # Recomendações de velocidade
        if wood_type == "hard":
            recommendations.append("Madeira dura: Reduza velocidade em 30% e use refrigerante")
        elif wood_type == "soft":
            recommendations.append("Madeira macia: Aumente velocidade em 20% para melhor acabamento")
        
        return recommendations

# ==============================================
# GERADOR DE ESTRATÉGIAS DE USINAGEM
# ==============================================

class MachiningStrategyGenerator:
    def __init__(self):
        self.strategies = {
            "high_speed_roughing": {
                "description": "Desbaste rápido para remoção de material",
                "stepover": 0.8,
                "depth_per_pass": 2.0,
                "feedrate_multiplier": 1.2
            },
            "detail_finishing": {
                "description": "Acabamento fino para detalhes",
                "stepover": 0.3,
                "depth_per_pass": 0.5,
                "feedrate_multiplier": 0.7
            },
            "contour_following": {
                "description": "Seguimento de contornos para relevos complexos",
                "stepover": 0.4,
                "depth_per_pass": 1.0,
                "feedrate_multiplier": 0.9
            }
        }
    
    def gerar_estrategia_completa(self, z_map, wood_type, complexity):
        """Gera estratégia completa de usinagem"""
        strategy = {
            "roughing": self._gerar_estrategia_desbaste(z_map, wood_type),
            "finishing": self._gerar_estrategia_acabamento(z_map, wood_type, complexity),
            "tool_paths": [],
            "estimated_time": 0
        }
        
        # Calcular tempo estimado
        strategy["estimated_time"] = self._calcular_tempo_estimado(strategy, wood_type)
        
        return strategy
    
    def _gerar_estrategia_desbaste(self, z_map, wood_type):
        """Gera estratégia de desbaste"""
        max_depth = np.max(z_map)
        
        if wood_type == "soft":
            depth_per_pass = 3.0
            num_passes = max(1, math.ceil(max_depth / depth_per_pass))
        elif wood_type == "hard":
            depth_per_pass = 1.0
            num_passes = max(1, math.ceil(max_depth / depth_per_pass))
        else:  # medium
            depth_per_pass = 2.0
            num_passes = max(1, math.ceil(max_depth / depth_per_pass))
        
        return {
            "type": "roughing",
            "depth_per_pass": depth_per_pass,
            "num_passes": num_passes,
            "stepover": 0.7,
            "feedrate": "alta" if wood_type == "soft" else "media"
        }
    
    def _gerar_estrategia_acabamento(self, z_map, wood_type, complexity):
        """Gera estratégia de acabamento"""
        detail_level = self._calcular_nivel_detalhe(z_map)
        
        if complexity == "high" or detail_level > 0.6:
            # Alto detalhe - múltiplas estratégias
            strategies = [
                {
                    "type": "contour_finishing",
                    "stepover": 0.2,
                    "feedrate": "baixa",
                    "purpose": "Contornos principais"
                },
                {
                    "type": "detail_finishing", 
                    "stepover": 0.1,
                    "feedrate": "muito_baixa",
                    "purpose": "Detalhes finos"
                }
            ]
        else:
            # Detalhe médio/baixo
            strategies = [
                {
                    "type": "finishing",
                    "stepover": 0.3,
                    "feedrate": "media",
                    "purpose": "Acabamento geral"
                }
            ]
        
        return strategies
    
    def _calcular_nivel_detalhe(self, z_map):
        """Calcula o nível de detalhe do mapa"""
        grad_x = np.gradient(z_map, axis=1)
        grad_y = np.gradient(z_map, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalizar e retornar métrica de detalhe
        return np.mean(grad_magnitude) / np.max(grad_magnitude) if np.max(grad_magnitude) > 0 else 0
    
    def _calcular_tempo_estimado(self, strategy, wood_type):
        """Calcula tempo estimado de usinagem"""
        base_time = 60  # minutos base
        
        # Ajustar por tipo de madeira
        if wood_type == "hard":
            base_time *= 1.5
        elif wood_type == "soft":
            base_time *= 0.8
        
        # Ajustar por complexidade
        roughing_time = base_time * 0.4
        finishing_time = base_time * 0.6 * len(strategy["finishing"])
        
        return roughing_time + finishing_time

# ==============================================
# SISTEMA DE RELATÓRIOS PROFISSIONAIS
# ==============================================

class ProfessionalReportGenerator:
    def __init__(self):
        self.template = """
# RELATÓRIO DE ENTALHE EM MADEIRA
**Data:** {date}
**Projeto:** {project_name}

## 📊 RESUMO EXECUTIVO
{executive_summary}

## 🪵 CONFIGURAÇÕES DA MADEIRA
{t_wood_settings}

## ⚙️ PARÂMETROS TÉCNICOS
{t_technical_params}

## 📈 ANÁLISE DE VIABILIDADE
{t_feasibility_analysis}

## 🛠️ ESTRATÉGIA DE USINAGEM
{t_machining_strategy}

## ⚠️ RECOMENDAÇÕES E ALERTAS
{t_recommendations}

## 📋 LISTA DE MATERIAIS
{t_materials_list}

---
*Relatório gerado automaticamente por Wood Carving Studio Pro*
"""
    
    def gerar_relatorio_completo(self, analysis_data, output_path):
        """Gera relatório profissional completo"""
        try:
            report_content = self.template.format(
                date=datetime.now().strftime("%d/%m/%Y %H:%M"),
                project_name=analysis_data.get("project_name", "Entalhe em Madeira"),
                executive_summary=self._gerar_resumo_executivo(analysis_data),
                t_wood_settings=self._formatar_config_madeira(analysis_data),
                t_technical_params=self._formatar_parametros_tecnicos(analysis_data),
                t_feasibility_analysis=self._formatar_analise_viabilidade(analysis_data),
                t_machining_strategy=self._formatar_estrategia_usinagem(analysis_data),
                t_recommendations=self._formatar_recomendacoes(analysis_data),
                t_materials_list=self._formatar_lista_materiais(analysis_data)
            )
            
            # Salvar relatório
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Gerar versão JSON para análise
            json_path = output_path.replace('.txt', '_data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Erro ao gerar relatório: {e}")
            return False
    
    def _gerar_resumo_executivo(self, data):
        """Gera resumo executivo do projeto"""
        viability = data["feasibility_analysis"]
        stats = viability["statistics"]
        
        summary = f"""
• **Status:** {'✅ VIÁVEL' if viability['viable'] else '❌ NÃO VIÁVEL'}
• **Dimensões:** {data['params']['largura_mm']} x {data['params']['altura_mm']} mm
• **Profundidade Máxima:** {stats['max_depth']:.1f} mm
• **Tipo de Madeira:** {data['wood_type'].upper()}
• **Tempo Estimado:** {data['machining_strategy']['estimated_time']:.0f} min
• **Complexidade:** {'ALTA' if stats['detail_density'] > 0.4 else 'MÉDIA' if stats['detail_density'] > 0.2 else 'BAIXA'}
"""
        return summary
    
    def _formatar_config_madeira(self, data):
        """Formata configurações da madeira"""
        return f"""
• **Tipo:** {data['wood_type'].upper()}
• **Direção do Veio:** {data['grain_direction']}
• **Ferramentas Recomendadas:** {', '.join(data.get('recommended_tools', ['V-bit 90°', 'Ballnose']))}
"""
    
    def _formatar_analise_viabilidade(self, data):
        """Formata análise de viabilidade"""
        viability = data["feasibility_analysis"]
        stats = viability["statistics"]
        
        content = f"""
## 📊 Estatísticas Principais
• Profundidade Máxima: {stats['max_depth']:.1f} mm
• Profundidade Média: {stats['avg_depth']:.1f} mm  
• Menor Característica: {stats['smallest_feature']:.1f} mm
• Densidade de Detalhes: {stats['detail_density']:.1%}
• Inclinação Máxima: {stats['max_slope']:.1f}°

## ⚠️ Alertas
{chr(10).join(['• ' + warning for warning in viability['warnings']]) or '• Nenhum alerta crítico'}
"""
        return content

# ==============================================
# INTERFACE AVANÇADA COM RELATÓRIOS
# ==============================================

class AdvancedWoodCarvingApp(WoodCarvingApp):
    def __init__(self, root):
        super().__init__(root)
        self.analyzer = WoodFeasibilityAnalyzer()
        self.strategy_gen = MachiningStrategyGenerator()
        self.report_gen = ProfessionalReportGenerator()
        
        # Adicionar abas avançadas
        self.setup_advanced_interface()
    
    def setup_advanced_interface(self):
        """Adiciona funcionalidades avançadas à interface"""
        # Adicionar aba de Análise
        notebook = self.root.winfo_children()[0].winfo_children()[1]  # Acessar notebook
        
        # Aba de Análise
        tab_analise = ttk.Frame(notebook, style='Wood.TFrame')
        notebook.add(tab_analise, text="📊 Análise")
        self.create_analysis_tab(tab_analise)
        
        # Aba de Relatórios
        tab_relatorios = ttk.Frame(notebook, style='Wood.TFrame')
        notebook.add(tab_relatorios, text="📋 Relatório")
        self.create_reports_tab(tab_relatorios)
    
    def create_analysis_tab(self, parent):
        """Cria aba de análise detalhada"""
        # Frame de análise
        analysis_frame = ttk.Frame(parent, style='Wood.TFrame')
        analysis_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Botão de análise
        self.analyze_btn = ttk.Button(analysis_frame, text="🔍 ANALISAR VIABILIDADE",
                                     command=self.analisar_projeto, style='Wood.TButton')
        self.analyze_btn.pack(pady=10)
        
        # Área de resultados
        self.analysis_text = tk.Text(analysis_frame, height=20, width=80, bg='#F5F5DC', fg='#333333')
        self.analysis_text.pack(fill='both', expand=True, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.analysis_text)
        scrollbar.pack(side='right', fill='y')
        self.analysis_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.analysis_text.yview)
    
    def create_reports_tab(self, parent):
        """Cria aba de relatórios"""
        report_frame = ttk.Frame(parent, style='Wood.TFrame')
        report_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Controles de relatório
        ttk.Label(report_frame, text="Nome do Projeto:", style='Wood.TLabel').pack(anchor='w', pady=5)
        self.entry_project_name = ttk.Entry(report_frame, width=40)
        self.entry_project_name.pack(fill='x', pady=5)
        self.entry_project_name.insert(0, "Meu Entalhe em Madeira")
        
        # Botões
        btn_frame = ttk.Frame(report_frame, style='Wood.TFrame')
        btn_frame.pack(fill='x', pady=15)
        
        ttk.Button(btn_frame, text="📄 GERAR RELATÓRIO", 
                  command=self.gerar_relatorio, style='Wood.TButton').pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text="💾 SALVAR ANÁLISE", 
                  command=self.salvar_analise, style='Wood.TButton').pack(side='left', padx=5)
        
        # Visualização do relatório
        self.report_text = tk.Text(report_frame, height=15, width=80, bg='#F5F5DC', fg='#333333')
        self.report_text.pack(fill='both', expand=True, pady=10)
    
    def analisar_projeto(self):
        """Executa análise completa do projeto"""
        try:
            if not hasattr(self, 'current_z_map'):
                messagebox.showwarning("Aviso", "Gere um entalhe primeiro para analisar.")
                return
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, "🔍 Analisando viabilidade...\n\n")
            self.root.update()
            
            # Executar análise
            analysis = self.analyzer.analisar_viabilidade(
                self.current_z_map, 
                self.current_params,
                self.tipo_madeira.get()
            )
            
            # Gerar estratégia
            complexity = "high" if analysis["statistics"]["detail_density"] > 0.4 else "medium"
            strategy = self.strategy_gen.gerar_estrategia_completa(
                self.current_z_map, 
                self.tipo_madeira.get(),
                complexity
            )
            
            # Exibir resultados
            self.exibir_resultados_analise(analysis, strategy)
            
            # Salvar dados atuais para relatório
            self.current_analysis = analysis
            self.current_strategy = strategy
            
        except Exception as e:
            self.analysis_text.insert(tk.END, f"❌ Erro na análise: {str(e)}")
    
    def exibir_resultados_analise(self, analysis, strategy):
        """Exibe resultados da análise na interface"""
        text = self.analysis_text
        
        # Cabeçalho
        text.insert(tk.END, "="*60 + "\n")
        text.insert(tk.END, "📊 RELATÓRIO DE ANÁLISE DE VIABILIDADE\n")
        text.insert(tk.END, "="*60 + "\n\n")
        
        # Status de viabilidade
        if analysis["viable"]:
            text.insert(tk.END, "✅ PROJETO VIÁVEL\n", "green")
        else:
            text.insert(tk.END, "❌ PROJETO NÃO VIÁVEL\n", "red")
        
        text.insert(tk.END, "\n")
        
        # Estatísticas
        stats = analysis["statistics"]
        text.insert(tk.END, "📈 ESTATÍSTICAS PRINCIPAIS:\n")
        text.insert(tk.END, f"• Profundidade máxima: {stats['max_depth']:.1f} mm\n")
        text.insert(tk.END, f"• Profundidade mínima: {stats['min_depth']:.1f} mm\n")
        text.insert(tk.END, f"• Variação total: {stats['depth_range']:.1f} mm\n")
        text.insert(tk.END, f"• Menor característica: {stats['smallest_feature']:.1f} mm\n")
        text.insert(tk.END, f"• Densidade de detalhes: {stats['detail_density']:.1%}\n")
        text.insert(tk.END, f"• Inclinação máxima: {stats['max_slope']:.1f}°\n")
        
        text.insert(tk.END, "\n")
        
        # Alertas
        if analysis["warnings"]:
            text.insert(tk.END, "⚠️ ALERTAS:\n")
            for warning in analysis["warnings"]:
                text.insert(tk.END, f"• {warning}\n")
            text.insert(tk.END, "\n")
        
        # Estratégia
        text.insert(tk.END, "🛠️ ESTRATÉGIA RECOMENDADA:\n")
        text.insert(tk.END, f"• Tempo estimado: {strategy['estimated_time']:.0f} minutos\n")
        text.insert(tk.END, f"• Passes de desbaste: {strategy['roughing']['num_passes']}\n")
        text.insert(tk.END, f"• Estratégias de acabamento: {len(strategy['finishing'])}\n")
        
        text.insert(tk.END, "\n")
        
        # Recomendações
        if analysis["recommendations"]:
            text.insert(tk.END, "💡 RECOMENDAÇÕES:\n")
            for rec in analysis["recommendations"]:
                text.insert(tk.END, f"• {rec}\n")
        
        # Configurar cores
        text.tag_configure("green", foreground="green")
        text.tag_configure("red", foreground="red")
    
    def gerar_relatorio(self):
        """Gera relatório profissional completo"""
        try:
            if not hasattr(self, 'current_analysis'):
                messagebox.showwarning("Aviso", "Execute uma análise primeiro.")
                return
            
            # Preparar dados para relatório
            report_data = {
                "project_name": self.entry_project_name.get(),
                "wood_type": self.tipo_madeira.get(),
                "grain_direction": self.direcao_veio.get(),
                "params": self.current_params,
                "feasibility_analysis": self.current_analysis,
                "machining_strategy": self.current_strategy,
                "timestamp": datetime.now().isoformat()
            }
            
            # Gerar relatório
            output_dir = os.path.join(os.getcwd(), "WoodCarving_Output")
            os.makedirs(output_dir, exist_ok=True)
            
            report_path = os.path.join(output_dir, "relatorio_entalhe.txt")
            
            success = self.report_gen.gerar_relatorio_completo(report_data, report_path)
            
            if success:
                # Exibir relatório na interface
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, report_content)
                
                messagebox.showinfo("Sucesso", f"Relatório gerado:\n{report_path}")
            else:
                messagebox.showerror("Erro", "Falha ao gerar relatório.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar relatório: {str(e)}")
    
    def salvar_analise(self):
        """Salva análise em arquivo JSON"""
        try:
            if not hasattr(self, 'current_analysis'):
                messagebox.showwarning("Aviso", "Execute uma análise primeiro.")
                return
            
            output_dir = os.path.join(os.getcwd(), "WoodCarving_Output")
            os.makedirs(output_dir, exist_ok=True)
            
            analysis_data = {
                "analysis": self.current_analysis,
                "strategy": self.current_strategy,
                "params": self.current_params,
                "wood_type": self.tipo_madeira.get(),
                "timestamp": datetime.now().isoformat()
            }
            
            file_path = os.path.join(output_dir, "analise_projeto.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Sucesso", f"Análise salva:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar análise: {str(e)}")
    
    def processar_entalhe_madeira(self, img_path, params):
        """Sobrescreve método para salvar dados atuais"""
        success = super().processar_entalhe_madeira(img_path, params)
        
        if success and hasattr(self, 'current_z_map'):
            # Salvar parâmetros atuais para análise
            self.current_params = params.copy()
            
        return success

# ==============================================
# EXECUÇÃO DO SISTEMA COMPLETO
# ==============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedWoodCarvingApp(root)
    
    # Configurar ícone e título
    root.title("🪵 Wood Carving Studio Pro - Edição Avançada")
    
    # Centralizar janela
    window_width = 1000
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()
    