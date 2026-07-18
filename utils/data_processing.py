import pandas as pd
import numpy as np

def load_and_prepare_data(filepath):
    """
    Charge le jeu de données Mpox, convertit la saison en numérique
    et sépare les variables explicatives (X) de la variable cible (y).
    """
    # 1. Chargement du CSV
    df = pd.read_csv(filepath)
    
    # 2. Gestion et conversion sécurisée de la colonne Saison
    if 'Saison' in df.columns and 'Saison_num' not in df.columns:
        # On nettoie les espaces cachés au cas où (ex: "Saison Sèche " -> "Saison Sèche")
        df['Saison_nettoyee'] = df['Saison'].astype(str).str.strip()
        
        # Dictionnaire de correspondance flexible
        mapping_saison = {
            'Saison Sèche': 0, 
            'Saison des Pluies': 1,
            'Seche': 0,
            'Pluie': 1,
            '0': 0,
            '1': 1
        }
        
        # Application du mapping
        df['Saison_num'] = df['Saison_nettoyee'].map(mapping_saison)
        
        # Sécurité : Si des valeurs n'ont pas pu être converties, on force en numérique par défaut
        if df['Saison_num'].isna().any():
            # Si le mapping échoue, on tente de convertir directement en ignorant les erreurs de texte
            df['Saison_num'] = pd.to_numeric(df['Saison'], errors='coerce').fillna(0).astype(int)
        else:
            df['Saison_num'] = df['Saison_num'].astype(int)
            
        # On supprime la colonne temporaire
        df = df.drop(columns=['Saison_nettoyee'])

    # Liste exacte de tes 11 variables explicatives
    feature_names = [
        'Pluviometrie_mm', 'Temperature_C', 'NDVI', 'Humidite_pct', 
        'Densite_Population', 'Couverture_Vaccinale_pct', 'Tests_Realises', 
        'Distance_Centre_Sante_km', 'Reservoirs_Animaux', 'Mobilite_Humaine', 
        'Population_Risque', 'Saison_num'
    ]
    
    # Séparation X (features) et y (cible)
    X = df[feature_names]
    y = df['Cas_Confirmes']
    
    return X, y, feature_names

def validate_inputs(pluviometrie, temperature, ndvi, humidite, densite_pop, 
                    couverture_vac, tests_realises, dist_centre, reservoirs, 
                    mobilite, pop_risque, saison):
    """
    Valide les entrées du formulaire Streamlit pour éviter les valeurs aberrantes.
    """
    if not (10.0 <= temperature <= 45.0):
        return False, "La température doit être comprise entre 10°C et 45°C."
        
    if not (0.0 <= ndvi <= 1.0):
        return False, "L'indice NDVI doit être compris entre 0 et 1."
        
    if not (0.0 <= couverture_vac <= 100.0):
        return False, "La couverture vaccinale doit être comprise entre 0% et 100%."
        
    if not (0.0 <= humidite <= 100.0):
        return False, "L'humidité doit être comprise entre 0% et 100%."
        
    if saison not in [0, 1]:
        return False, "Le code saison doit être 0 (Sèche) ou 1 (Pluies)."
        
    return True, ""

def create_input_dataframe(pluviometrie, temperature, ndvi, humidite, densite_pop, 
                           couverture_vac, tests_realises, dist_centre, reservoirs, 
                           mobilite, pop_risque, saison):
    """
    Convertit les entrées en DataFrame Pandas avec l'ordre exact.
    """
    data = [{
        'Pluviometrie_mm': pluviometrie,
        'Temperature_C': temperature,
        'NDVI': ndvi,
        'Humidite_pct': humidite,
        'Densite_Population': densite_pop,
        'Couverture_Vaccinale_pct': couverture_vac,
        'Tests_Realises': tests_realises,
        'Distance_Centre_Sante_km': dist_centre,
        'Reservoirs_Animaux': reservoirs,
        'Mobilite_Humaine': mobilite,
        'Population_Risque': pop_risque,
        'Saison_num': saison
    }]
    
    return pd.DataFrame(data)

