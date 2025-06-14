"""
Görselleştirme modülü - Paclitaxel doz optimizasyonu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    def __init__(self):
        # Modern görselleştirme stili
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Türkçe karakter desteği
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
        
    def plot_dose_response_curves(self, data_processor, model, selected_cell_lines=None, max_lines=10):
        """Doz-yanıt eğrilerini çiz"""
        print("Doz-yanıt eğrileri çiziliyor...")
        
        # Hücre hatlarını seç
        all_cell_lines = data_processor.get_cell_lines()
        if selected_cell_lines is None:
            # Rastgele seçim yap
            selected_cell_lines = np.random.choice(
                all_cell_lines, 
                min(max_lines, len(all_cell_lines)), 
                replace=False
            )
        
        plt.figure(figsize=(15, 10))
        
        # Renk paleti
        colors = sns.color_palette("husl", len(selected_cell_lines))
        
        for i, cell_line in enumerate(selected_cell_lines):
            # Gerçek veriyi al
            cell_data = data_processor.df[data_processor.df['ARXSPAN_ID'] == cell_line]
            
            # Orijinal doz değerlerini al (ölçeklenmemiş)
            original_data = pd.read_excel('Book1 (1).xlsx', sheet_name='DOZ X CANLILIK')
            original_cell_data = original_data[
                (original_data['DRUG_NAME'] == 'PACLITAXEL') & 
                (original_data['ARXSPAN_ID'] == cell_line)
            ]
            
            if len(original_cell_data) > 0:
                # Gerçek veri noktalarını çiz
                plt.scatter(
                    original_cell_data['dose'], 
                    original_cell_data['viability'],
                    color=colors[i], 
                    alpha=0.7, 
                    s=50,
                    label=f'{cell_line} (Gerçek)'
                )
                
                # Model tahminlerini çiz
                dose_range, predicted_viability = model.predict_dose_response_curve(
                    data_processor, cell_line
                )
                
                if dose_range is not None:
                    plt.plot(
                        dose_range, 
                        predicted_viability,
                        color=colors[i], 
                        linestyle='--', 
                        alpha=0.8,
                        linewidth=2
                    )
        
        plt.xscale('log')
        plt.xlabel('Doz (µM)', fontsize=14, fontweight='bold')
        plt.ylabel('Hücre Canlılığı', fontsize=14, fontweight='bold')
        plt.title('Paclitaxel Doz-Yanıt Eğrileri', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # %80 ölüm çizgisi (%20 canlılık)
        plt.axhline(y=0.2, color='red', linestyle=':', alpha=0.7, 
                   label='%80 Ölüm Hedefi')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig('paclitaxel_dose_response_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, model):
        """Özellik önemini görselleştir"""
        print("Özellik önem grafiği çiziliyor...")
        
        feature_importance = model.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(feature_importance['Feature'], feature_importance['Importance'], 
                      color=['#2E86AB', '#A23B72', '#F18F01'])
        
        # Değerleri çubukların üzerine yaz
        for bar, value in zip(bars, feature_importance['Importance']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Özellik Önemleri', fontsize=16, fontweight='bold')
        plt.xlabel('Özellikler', fontsize=14)
        plt.ylabel('Önem Skoru', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show() 