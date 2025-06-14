"""
Veri işleme modülü - Paclitaxel doz optimizasyonu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
from scipy.optimize import curve_fit

class DataProcessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.df = None
        
    def load_data(self, file_path='Book1 (1).xlsx'):
        """Excel dosyasından veri yükle"""
        # DOZ X CANLILIK sayfasını oku
        self.df = pd.read_excel(file_path, sheet_name='DOZ X CANLILIK')
        
        print("Yüklenen sütunlar:", self.df.columns.tolist())
        print(f"Toplam veri sayısı: {len(self.df)}")
        print("\nİlk 5 satır:")
        print(self.df.head())
        return self
        
    def preprocess(self, drug_name='PACLITAXEL'):
        """Veriyi ön işle"""
        if self.df is None:
            raise ValueError("Veri yüklenmedi. Önce load_data() çağırın.")
            
        # Sadece belirtilen ilacı filtrele
        self.df = self.df[self.df['DRUG_NAME'] == drug_name].copy()
        print(f"\n{drug_name} verisi filtrelendi: {len(self.df)} satır")
        
        # Eksik değerleri temizle
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['dose', 'viability', 'ARXSPAN_ID'])
        print(f"Eksik değerler temizlendi: {len(self.df)} satır kaldı")
        
        # Numeric sütunları dönüştür
        self.df['dose'] = pd.to_numeric(self.df['dose'], errors='coerce')
        self.df['viability'] = pd.to_numeric(self.df['viability'], errors='coerce')
        
        # Geçersiz değerleri temizle
        self.df = self.df.dropna(subset=['dose', 'viability'])
        print(f"Geçersiz değerler temizlendi: {len(self.df)} satır kaldı")
        
        # Özellik mühendisliği
        self.df['log_dose'] = np.log10(self.df['dose'] + 1e-10)  # Log dönüşümü
        
        # Hücre hatlarını kodla
        self.df['cell_line_encoded'] = self.label_encoder.fit_transform(self.df['ARXSPAN_ID'])
        
        # Özellikleri ölçekle
        features_to_scale = ['dose', 'log_dose']
        self.df[features_to_scale] = self.scaler.fit_transform(self.df[features_to_scale])
        
        print(f"\nİşlenmiş veri özeti:")
        print(f"- Hücre hattı sayısı: {self.df['ARXSPAN_ID'].nunique()}")
        print(f"- Doz aralığı: {self.df['dose'].min():.4f} - {self.df['dose'].max():.4f}")
        print(f"- Canlılık aralığı: {self.df['viability'].min():.3f} - {self.df['viability'].max():.3f}")
        
        # IC50 hesapla
        self.calculate_ic50()
        
        # Toksisite indeksi hesapla
        self.calculate_toxicity_index()
        
        return self
        
    def sigmoid_4pl(self, x, top, bottom, ic50, hill_slope):
        """4-parametreli sigmoid fonksiyonu (Hill denklemi)"""
        return bottom + (top - bottom) / (1 + (x / ic50) ** hill_slope)
        
    def calculate_ic50(self):
        """Her hücre hattı için IC50 hesapla"""
        ic50_results = []
        
        for cell_line in self.df['ARXSPAN_ID'].unique():
            cell_data = self.df[self.df['ARXSPAN_ID'] == cell_line].copy()
            cell_data = cell_data.sort_values('dose')
            
            # Orijinal doz değerlerini kullan (ölçeklenmemiş)
            doses_orig = pd.read_excel('Book1 (1).xlsx', sheet_name='DOZ X CANLILIK')
            doses_orig = doses_orig[doses_orig['DRUG_NAME'] == 'PACLITAXEL']
            doses_orig = doses_orig[doses_orig['ARXSPAN_ID'] == cell_line]
            
            x = doses_orig['dose'].values
            y = cell_data['viability'].values
            
            try:
                # 4-parametreli sigmoid eğrisi fit et
                popt, _ = curve_fit(
                    self.sigmoid_4pl,
                    x, y,
                    p0=[1.0, 0.0, np.median(x), 1.0],  # top, bottom, ic50, hill_slope
                    bounds=([0.5, 0.0, min(x), 0.1], [1.5, 0.5, max(x), 10.0]),
                    maxfev=5000
                )
                
                ic50 = popt[2]
                r_squared = self.calculate_r_squared(y, self.sigmoid_4pl(x, *popt))
                
                ic50_results.append({
                    'Cell_Line': cell_line,
                    'IC50_µM': ic50,
                    'R_squared': r_squared,
                    'Hill_Slope': popt[3],
                    'Top_Plateau': popt[0],
                    'Bottom_Plateau': popt[1]
                })
                
            except Exception as e:
                print(f"IC50 hesaplanamadı - {cell_line}: {str(e)}")
                ic50_results.append({
                    'Cell_Line': cell_line,
                    'IC50_µM': np.nan,
                    'R_squared': np.nan,
                    'Hill_Slope': np.nan,
                    'Top_Plateau': np.nan,
                    'Bottom_Plateau': np.nan
                })
        
        # IC50 sonuçlarını kaydet
        ic50_df = pd.DataFrame(ic50_results)
        ic50_df.to_csv('paclitaxel_ic50_results.csv', index=False)
        print(f"\nIC50 sonuçları kaydedildi: {len(ic50_df)} hücre hattı")
        print(f"Başarılı IC50 hesaplaması: {ic50_df['IC50_µM'].notna().sum()} hücre hattı")
        
    def calculate_r_squared(self, y_actual, y_pred):
        """R-kare hesapla"""
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return 1 - (ss_res / ss_tot)
        
    def calculate_toxicity_index(self):
        """Toksisite indeksi hesapla"""
        toxicity_results = []
        
        for cell_line in self.df['ARXSPAN_ID'].unique():
            cell_data = self.df[self.df['ARXSPAN_ID'] == cell_line]
            
            # En yüksek dozdaki ortalama canlılık
            max_dose_viability = cell_data[cell_data['dose'] == cell_data['dose'].max()]['viability'].mean()
            
            # Toksisite indeksi (1 - canlılık)
            toxicity_index = 1 - max_dose_viability
            
            toxicity_results.append({
                'Cell_Line': cell_line,
                'Max_Dose_Viability': max_dose_viability,
                'Toxicity_Index': toxicity_index
            })
        
        # Toksisite sonuçlarını kaydet
        toxicity_df = pd.DataFrame(toxicity_results)
        toxicity_df.to_csv('paclitaxel_toxicity_index.csv', index=False)
        print(f"Toksisite indeksi kaydedildi: {len(toxicity_df)} hücre hattı")
        
    def get_features_target(self):
        """Özellikler ve hedef değişkeni döndür"""
        if self.df is None:
            raise ValueError("Veri işlenmedi. preprocess() çağırın.")
            
        feature_columns = ['dose', 'cell_line_encoded', 'log_dose']
        X = self.df[feature_columns]
        y = self.df['viability']
        
        return X, y
        
    def get_cell_lines(self):
        """Benzersiz hücre hatlarını döndür"""
        if self.df is None:
            raise ValueError("Veri işlenmedi. preprocess() çağırın.")
            
        return self.df['ARXSPAN_ID'].unique() 