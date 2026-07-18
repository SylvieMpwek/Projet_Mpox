import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_processing import load_and_prepare_data

def train_model():
    # 1. Chargement et préparation automatique des données
    X, y, feature_names = load_and_prepare_data('data/donnees_mpox_RDC_3000.csv')
    
    df_complet = pd.read_csv('data/donnees_mpox_RDC_3000.csv')
    provinces = df_complet['Province'].values
    
    X_arr = X.values
    y_arr = y.values

    # 2. Configuration de la validation croisée stratifiée
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)
    
    scaler = MinMaxScaler()
   
    models = {
        'Linear_Regression': LinearRegression(),
        'Ridge': Ridge(random_state=20),
        'Lasso': Lasso(random_state=20),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=20),
        'Gradient_Boosting': GradientBoostingRegressor(random_state=20)
    }
    
    # 3. Boucle d'évaluation par Validation Croisée
    for name, model in models.items():
        print(f"\n Évaluation de : {name} via {n_splits}-Fold CV...")
        
        fold_r2 = []
        fold_rmse = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_arr, provinces)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            
            # Normalisation [0, 1] sur le pli en cours
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entraînement sur le pli
            model.fit(X_train_scaled, y_train)
            
            # Évaluation du pli
            y_pred = model.predict(X_test_scaled)
            fold_r2.append(r2_score(y_test, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            
        print(f"   -> R² Moyen    : {np.mean(fold_r2):.4f} (+/- {np.std(fold_r2):.4f})")
        print(f"   -> RMSE Moyen  : {np.mean(fold_rmse):.4f} (+/- {np.std(fold_rmse):.4f})")
        
        # 4. Entraînement final pour Streamlit
        print(f"   Entraînement final de {name} sur toutes les données...")
        X_all_scaled = scaler.fit_transform(X_arr)
        model.fit(X_all_scaled, y_arr)
        
        # Sauvegarde individuelle
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/model_{name}.pkl')
        print(f"   Modèle sauvegardé dans 'models/model_{name}.pkl'")
        
    # Sauvegarde du scaler final entraîné
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n Tous les modèles et le scaler ont été sauvegardés avec succès !")

if __name__ == "__main__":
    train_model()