"""
Model eğitimi ve tahmin modülü - Paclitaxel doz optimizasyonu
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

class DoseResponseModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_names = ['dose', 'cell_line_encoded', 'log_dose']
        
    def train(self, X, y):
        """Modeli eğit"""
        print("Model eğitimi başlıyor...")
        
        # XGBoost model parametreleri
        xgb_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Hyperparameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Model oluştur
        model = xgb.XGBRegressor(**xgb_params)
        
        # Cross-validation ile grid search
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=kf,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Model performansını değerlendir
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Model eğitimi tamamlandı!")
        print(f"En iyi R² skoru: {grid_search.best_score_:.3f}")
        print(f"Eğitim R² skoru: {r2:.3f}")
        print(f"Eğitim RMSE: {rmse:.3f}")
        print(f"En iyi parametreler: {self.best_params}")
        
        return self
        
    def predict(self, X):
        """Tahmin yap"""
        if self.model is None:
            raise ValueError("Model eğitilmedi. train() fonksiyonunu çağırın.")
            
        return self.model.predict(X)
        
    def find_optimal_dose(self, data_processor, cell_line, target_viability=0.2):
        """
        Belirli bir hücre hattı için optimal dozu bul
        target_viability: hedeflenen canlılık oranı (0.2 = %20 canlılık = %80 ölüm)
        """
        if self.model is None:
            raise ValueError("Model eğitilmedi. train() fonksiyonunu çağırın.")
        
        # Orijinal doz aralığı
        dose_range = np.logspace(np.log10(0.0004), np.log10(0.1024), 1000)
        
        # Hücre hattını kodla
        try:
            cell_line_encoded = data_processor.label_encoder.transform([cell_line])[0]
        except ValueError:
            print(f"Hücre hattı '{cell_line}' bulunamadı!")
            return None, None, None
        
        # Tahmin için veri hazırla
        log_doses = np.log10(dose_range + 1e-10)
        
        # Özellikleri ölçekle (data_processor'daki scaler'ı kullan)
        dose_scaled = data_processor.scaler.transform(
            np.column_stack([dose_range, log_doses])
        )
        
        X_pred = np.column_stack([
            dose_scaled[:, 0],  # ölçeklenmiş doz
            np.full(len(dose_range), cell_line_encoded),  # hücre hattı
            dose_scaled[:, 1]   # ölçeklenmiş log doz
        ])
        
        # Canlılık tahminleri
        predicted_viability = self.predict(X_pred)
        
        # Hedef canlılığa en yakın dozu bul
        optimal_idx = np.argmin(np.abs(predicted_viability - target_viability))
        optimal_dose = dose_range[optimal_idx]
        predicted_viability_at_optimal = predicted_viability[optimal_idx]
        
        # Bootstrap ile güven aralığı hesapla
        bootstrap_doses = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Rastgele örnekleme
            indices = np.random.choice(len(X_pred), len(X_pred), replace=True)
            X_boot = X_pred[indices]
            y_boot = self.predict(X_boot)
            
            # Bu bootstrap örneği için optimal doz
            opt_idx = np.argmin(np.abs(y_boot - target_viability))
            bootstrap_doses.append(dose_range[opt_idx])
        
        # %95 güven aralığı
        ci_lower = np.percentile(bootstrap_doses, 2.5)
        ci_upper = np.percentile(bootstrap_doses, 97.5)
        
        print(f"\n{cell_line} için optimal doz analizi:")
        print(f"- Optimal doz: {optimal_dose:.6f} µM")
        print(f"- Tahmini canlılık: {predicted_viability_at_optimal:.3f}")
        print(f"- %95 GA: [{ci_lower:.6f}, {ci_upper:.6f}] µM")
        
        return optimal_dose, ci_lower, ci_upper
        
    def get_feature_importance(self):
        """Özellik önemini döndür"""
        if self.model is None:
            raise ValueError("Model eğitilmedi. train() fonksiyonunu çağırın.")
            
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return feature_importance_df
        
    def predict_dose_response_curve(self, data_processor, cell_line, n_points=100):
        """Belirli bir hücre hattı için doz-yanıt eğrisi tahmin et"""
        if self.model is None:
            raise ValueError("Model eğitilmedi. train() fonksiyonunu çağırın.")
        
        # Doz aralığı
        dose_range = np.logspace(np.log10(0.0004), np.log10(0.1024), n_points)
        
        # Hücre hattını kodla
        try:
            cell_line_encoded = data_processor.label_encoder.transform([cell_line])[0]
        except ValueError:
            return None, None
        
        # Tahmin için veri hazırla
        log_doses = np.log10(dose_range + 1e-10)
        dose_scaled = data_processor.scaler.transform(
            np.column_stack([dose_range, log_doses])
        )
        
        X_pred = np.column_stack([
            dose_scaled[:, 0],
            np.full(len(dose_range), cell_line_encoded),
            dose_scaled[:, 1]
        ])
        
        # Tahminleri yap
        predicted_viability = self.predict(X_pred)
        
        return dose_range, predicted_viability 