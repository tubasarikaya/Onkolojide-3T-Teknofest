"""
Raporlama modülü - Paclitaxel doz optimizasyonu
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

class Reporter:
    def __init__(self):
        self.optimal_dose_results = []
        self.performance_metrics = {}
        
    def calculate_model_performance(self, y_true, y_pred):
        """Model performans metriklerini hesapla"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        self.performance_metrics = {
            'R2_Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Data_Points': len(y_true),
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.performance_metrics
        
    def add_optimal_dose(self, cell_line, optimal_dose, ci_lower, ci_upper, predicted_viability=None):
        """Optimal doz sonucunu ekle"""
        self.optimal_dose_results.append({
            'Cell_Line': cell_line,
            'Optimal_Dose_µM': optimal_dose,
            'CI_Lower_µM': ci_lower,
            'CI_Upper_µM': ci_upper,
            'CI_Width_µM': ci_upper - ci_lower,
            'Predicted_Viability': predicted_viability,
            'Confidence_Level': '95%'
        })
        
    def generate_comprehensive_report(self):
        """Kapsamlı rapor oluştur"""
        print("\n" + "="*80)
        print("🧬 PACLİTAXEL DOZ OPTİMİZASYONU ANALİZ RAPORU")
        print("="*80)
        
        # Model performansı
        if self.performance_metrics:
            print("\n📊 MODEL PERFORMANSI:")
            print("-" * 40)
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
        
        # Optimal doz sonuçları
        if self.optimal_dose_results:
            results_df = pd.DataFrame(self.optimal_dose_results)
            
            print("\n🎯 OPTIMAL DOZ ÖZETİ:")
            print("-" * 40)
            print(f"Analiz edilen hücre hattı sayısı: {len(results_df)}")
            print(f"Ortalama optimal doz: {results_df['Optimal_Dose_µM'].mean():.6f} µM")
            print(f"Medyan optimal doz: {results_df['Optimal_Dose_µM'].median():.6f} µM")
            print(f"Doz aralığı: {results_df['Optimal_Dose_µM'].min():.6f} - {results_df['Optimal_Dose_µM'].max():.6f} µM")
            
            # CSV dosyasına kaydet
            results_df.to_csv('paclitaxel_optimal_doses.csv', index=False)
            print(f"\n💾 Sonuçlar 'paclitaxel_optimal_doses.csv' dosyasına kaydedildi.")
            
            return results_df
        
        return None 