"""
Konfigürasyon parametreleri - Paclitaxel doz optimizasyonu
"""

# Veri parametreleri
DATA_FILE = 'Book1 (1).xlsx'
FEATURE_COLUMNS = [
    'dose',                    # Doz konsantrasyonu (0.0004 - 0.1024 µM)
    'cell_line_encoded',       # Kodlanmış hücre hattı ID
    'log_dose'                 # Log dönüştürülmüş doz (özellik mühendisliği)
]
TARGET_COLUMN = 'viability'    # Hedef değişken (0-1 arası)

# Model parametreleri
RANDOM_STATE = 42
N_SPLITS = 5
TARGET_EFFICACY = 0.2  # %80 hücre ölümü için canlılık = 0.2

# XGBoost parametreleri
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'random_state': RANDOM_STATE
}

# Basitleştirilmiş hyperparameter grid
PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5]
}

# Doz aralığı parametreleri (Paclitaxel için)
MIN_DOSE = 0.0004  # Minimum doz (µM)
MAX_DOSE = 0.1024  # Maksimum doz (µM)
N_DOSE_POINTS = 1000

# Bootstrap parametreleri
N_BOOTSTRAP = 1000
CI_LOWER = 2.5
CI_UPPER = 97.5

# Görselleştirme parametreleri
FIGURE_SIZE = (12, 8)
DPI = 300
STYLE = 'seaborn-v0_8'

# Çıktı dosyaları
DOSE_RESPONSE_PLOT = 'paclitaxel_dose_response_curves.png'
FEATURE_IMPORTANCE_PLOT = 'feature_importance.png'
OPTIMAL_DOSES_CSV = 'paclitaxel_optimal_doses.csv'
IC50_RESULTS_CSV = 'paclitaxel_ic50_results.csv'
TOXICITY_INDEX_CSV = 'paclitaxel_toxicity_index.csv' 