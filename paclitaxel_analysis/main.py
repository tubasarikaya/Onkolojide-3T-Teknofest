"""
Paclitaxel Doz Optimizasyonu - Ana Çalıştırma Dosyası
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Modülleri içe aktar
from data_processor import DataProcessor
from model import DoseResponseModel  
from visualizer import Visualizer
from reporter import Reporter

def main():
    print("🧬 PACLİTAXEL DOZ OPTİMİZASYONU ANALİZİ BAŞLIYOR...")
    print("=" * 60)
    
    try:
        # Bileşenleri başlat
        print("\n1️⃣ Sistem bileşenleri başlatılıyor...")
        data_processor = DataProcessor()
        model = DoseResponseModel()
        visualizer = Visualizer()
        reporter = Reporter()
        
        # Veri yükleme ve ön işleme
        print("\n2️⃣ Excel verisi yükleniyor ve işleniyor...")
        data_processor.load_data('Book1 (1).xlsx')
        data_processor.preprocess('PACLITAXEL')  # Sadece Paclitaxel verisi
        
        # Özellik ve hedef değişkenleri al
        X, y = data_processor.get_features_target()
        print(f"\n📊 Model eğitimi için hazır veri:")
        print(f"   • Özellik sayısı: {X.shape[1]}")
        print(f"   • Veri noktası sayısı: {X.shape[0]}")
        print(f"   • Hücre hattı sayısı: {len(data_processor.get_cell_lines())}")
        
        # Model eğitimi
        print("\n3️⃣ XGBoost modeli eğitiliyor...")
        model.train(X, y)
        
        # Model performansını değerlendir
        print("\n4️⃣ Model performansı değerlendiriliyor...")
        y_pred = model.predict(X)
        performance_metrics = reporter.calculate_model_performance(y, y_pred)
        
        # Görselleştirmeler oluştur
        print("\n5️⃣ Görselleştirmeler hazırlanıyor...")
        
        # Doz-yanıt eğrileri (rastgele 15 hücre hattı)
        visualizer.plot_dose_response_curves(data_processor, model, max_lines=15)
        
        # Özellik önem grafiği
        visualizer.plot_feature_importance(model)
        
        # Optimal doz hesaplama
        print("\n6️⃣ Optimal dozlar hesaplanıyor...")
        cell_lines = data_processor.get_cell_lines()
        
        # İlk 50 hücre hattı için optimal doz hesapla
        sample_cell_lines = cell_lines[:50] if len(cell_lines) > 50 else cell_lines
        
        print(f"   • {len(sample_cell_lines)} hücre hattı için optimal doz hesaplanacak...")
        
        successful_calculations = 0
        for i, cell_line in enumerate(sample_cell_lines):
            try:
                optimal_dose, ci_lower, ci_upper = model.find_optimal_dose(
                    data_processor, cell_line, target_viability=0.2
                )
                
                if optimal_dose is not None:
                    reporter.add_optimal_dose(cell_line, optimal_dose, ci_lower, ci_upper)
                    successful_calculations += 1
                
                # İlerleme göstergesi
                if (i + 1) % 10 == 0:
                    print(f"     ✓ {i + 1}/{len(sample_cell_lines)} hücre hattı tamamlandı")
                    
            except Exception as e:
                print(f"     ⚠️ {cell_line} için optimal doz hesaplanamadı: {str(e)}")
                continue
        
        print(f"\n   ✅ {successful_calculations}/{len(sample_cell_lines)} hücre hattı için başarılı hesaplama")
        
        # Kapsamlı rapor oluştur
        print("\n7️⃣ Kapsamlı analiz raporu hazırlanıyor...")
        results_df = reporter.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("🎉 ANALİZ BAŞARIYLA TAMAMLANDI!")
        print("=" * 60)
        
        print("\n📁 Oluşturulan dosyalar:")
        output_files = [
            "paclitaxel_optimal_doses.csv",
            "paclitaxel_ic50_results.csv",
            "paclitaxel_toxicity_index.csv", 
            "paclitaxel_dose_response_curves.png",
            "feature_importance.png"
        ]
        
        for file in output_files:
            print(f"   ✓ {file}")
            
        return {
            'model': model,
            'data_processor': data_processor,
            'performance_metrics': performance_metrics,
            'results_df': results_df
        }
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_quick_analysis():
    """Hızlı analiz için basitleştirilmiş versiyon"""
    print("🚀 HIZLI PACLİTAXEL ANALİZİ")
    print("=" * 40)
    
    # Sadece temel analizi çalıştır
    data_processor = DataProcessor()
    data_processor.load_data('Book1 (1).xlsx')
    data_processor.preprocess('PACLITAXEL')
    
    X, y = data_processor.get_features_target()
    print(f"Veri yüklendi: {X.shape[0]} satır, {X.shape[1]} özellik")
    
    # Basit model eğitimi
    model = DoseResponseModel()
    model.train(X, y)
    
    # Sadece 5 hücre hattı için örnek analiz
    sample_cells = data_processor.get_cell_lines()[:5]
    print(f"\nÖrnek 5 hücre hattı için optimal doz:")
    
    for cell_line in sample_cells:
        try:
            optimal_dose, ci_lower, ci_upper = model.find_optimal_dose(
                data_processor, cell_line
            )
            if optimal_dose:
                print(f"  {cell_line}: {optimal_dose:.6f} µM [{ci_lower:.6f}-{ci_upper:.6f}]")
        except:
            print(f"  {cell_line}: Hesaplanamadı")

if __name__ == "__main__":
    # Tam analiz için main(), hızlı test için run_quick_analysis() kullanın
    
    print("Hangi analizi çalıştırmak istiyorsunuz?")
    print("1. Tam analiz (20-30 dakika)")  
    print("2. Hızlı analiz (2-3 dakika)")
    
    choice = input("Seçiminizi yapın (1 veya 2): ").strip()
    
    if choice == "2":
        run_quick_analysis()
    else:
        results = main()
        
        if results:
            print("\n🔬 Analiz tamamlandı! Sonuçlar yukarıda özetlenmiştir.")
            print("📊 Detaylı grafikleri ve CSV dosyalarını kontrol edin.")
        else:
            print("\n❌ Analiz sırasında hata oluştu. Lütfen hata mesajlarını kontrol edin.") 