import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils.data_processing import validate_inputs

# 1. Configuration de la page (Doit être la toute première commande)
st.set_page_config(
    page_title="Prédiction Mpox - RDC",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; color: #2E86C1; text-align: center; font-weight: bold; margin-bottom: 1rem; }
    .prediction-box { background-color: #D4E6F1; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0; border-left: 6px solid #1A5276; }
    .prediction-number { font-size: 3.2rem; font-weight: bold; color: #1A5276; margin: 0; }
    .metric-card { background-color: #F8F9F9; padding: 10px; border-radius: 5px; border: 1px solid #E5E8E8; }
</style>
""", unsafe_allow_html=True)

# 2. Barre latérale (Sidebar) - Définition du modèle
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=90)
    st.markdown("## Configuration du Modèle")
    
    # Choix du modèle - Hors de tout formulaire pour garantir la réactivité de la flèche
    model_choice = st.selectbox(
        "Modèle d'apprentissage :",
        ["Gradient Boosting", "Random Forest", "Régression Linéaire", "Ridge", "Lasso"],
        key="selected_model_dropdown"
    )
    
    model_mapping = {
        "Régression Linéaire": "Linear_Regression",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "Random Forest": "Random_Forest",
        "Gradient Boosting": "Gradient_Boosting"
    }
    selected_model_name = model_mapping[model_choice]

    # Dictionnaire contenant TES vrais résultats obtenus
    model_stats = {
        "Gradient_Boosting": {"r2": "0.8902", "rmse": "23.18", "status": "Champion (Robuste)"},
        "Random_Forest": {"r2": "0.8580", "rmse": "26.38", "status": "Très Élevé"},
        "Linear_Regression": {"r2": "0.8218", "rmse": "29.56", "status": "Stable / Linéaire"},
        "Ridge": {"r2": "0.8216", "rmse": "29.58", "status": "Régularisé (L2)"},
        "Lasso": {"r2": "0.7631", "rmse": "34.09", "status": "Moins Performant"}
    }

    st.markdown("---")
    st.markdown("### Performances Réelles (CV 5-Fold)")
    stats = model_stats[selected_model_name]
    st.metric(label=f"R² Moyen ({model_choice})", value=stats["r2"], delta=stats["status"])
    st.metric(label="Marge d'erreur (RMSE)", value=f"{stats['rmse']} cas")
    st.markdown("---")
    st.caption("🎓 Projet Master 2 (MASG) - RDC")

# 3. Chargement des fichiers de modélisation (.pkl)
@st.cache_resource
def load_selected_model(model_name):
    try:
        model = joblib.load(f'models/model_{model_name}.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error(f"⚠️ Fichier d'aiguillage manquant pour {model_name}. Exécutez d'abord : python train_model.py")
        st.stop()

model, scaler = load_selected_model(selected_model_name)

# Ordre strict des caractéristiques attendues par tes scripts d'entraînement
feature_names = [
    'Pluviometrie_mm', 'Temperature_C', 'NDVI', 'Humidite_pct', 
    'Densite_Population', 'Couverture_Vaccinale_pct', 'Tests_Realises', 
    'Distance_Centre_Sante_km', 'Reservoirs_Animaux', 'Mobilite_Humaine', 
    'Population_Risque', 'Saison_num'
]

# En-tête de la page principale
st.markdown('<p class="main-header">🇨🇩 Plateforme Intégrée de Prédiction Épidémiologique : Mpox en RDC</p>', unsafe_allow_html=True)

# Corps de l'application (Saisie des variables et Rapport de Simulation)
col_form, col_results = st.columns([5, 4])

with col_form:
    # Utilisation d'un formulaire unique pour stabiliser la soumission des données
    with st.form("main_prediction_form"):
        st.markdown("### 📋 Paramètres du Scénario Épidémique")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🌍 Indicateurs Climatiques & Écologiques**")
            pluviometrie = st.number_input("Pluviométrie Mensuelle (mm)", min_value=0.0, value=150.0, step=10.0)
            temperature = st.number_input("Température Moyenne (°C)", min_value=10.0, max_value=45.0, value=26.0, step=0.5)
            ndvi = st.slider("Indice de Végétation (NDVI)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
            humidite = st.slider("Humidité Relative (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            saison = st.selectbox("Saison Actuelle", options=[0, 1], format_func=lambda x: "Saison Sèche" if x == 0 else "Saison des Pluies")
            reservoirs = st.number_input("Index des Réservoirs Animaux", min_value=0, value=10, step=1)
            
        with c2:
            st.markdown("**🏥 Facteurs Démographiques & Sanitaires**")
            densite_pop = st.number_input("Densité de Population (hab/km²)", min_value=0.0, value=200.0, step=10.0)
            population_risque = st.number_input("Population Exposée au Risque", min_value=0, value=10000, step=500)
            mobilite = st.number_input("Index de Mobilité Humaine", min_value=0.0, value=50.0, step=5.0)
            dist_centre = st.number_input("Distance au Centre de Santé (km)", min_value=0.0, value=12.5, step=0.5)
            tests_realises = st.number_input("Volume de Tests Réalisés", min_value=0, value=500, step=50)
            couverture_vac = st.slider("Couverture Vaccinale de la Zone (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
            
        st.markdown("###")
        submitted = st.form_submit_button("⚡ Lancer l'Analyse Prédictive", type="primary")

# Gestion de l'affichage des résultats et importance des variables
with col_results:
    st.markdown("### 📊 Résultats et Diagnostic Automatique")
    
    if submitted:
        # Validation réglementaire des entrées
        is_valid, error_msg = validate_inputs(
            pluviometrie, temperature, ndvi, humidite, densite_pop, 
            couverture_vac, tests_realises, dist_centre, reservoirs, mobilite, population_risque, saison
        )
        
        if not is_valid:
            st.error(f"Donnée non valide : {error_msg}")
        else:
            # Construction ordonnée du vecteur d'entrée
            input_dict = {
                'Pluviometrie_mm': pluviometrie, 'Temperature_C': temperature, 'NDVI': ndvi, 'Humidite_pct': humidite,
                'Densite_Population': densite_pop, 'Couverture_Vaccinale_pct': couverture_vac, 'Tests_Realises': tests_realises,
                'Distance_Centre_Sante_km': dist_centre, 'Reservoirs_Animaux': reservoirs, 'Mobilite_Humaine': mobilite,
                'Population_Risque': population_risque, 'Saison_num': saison
            }
            input_df = pd.DataFrame([input_dict])[feature_names]
            
            # Normalisation et prédiction (utilisation de .values pour éviter l'alerte de colonnes)
            input_scaled = scaler.transform(input_df.values)
            prediction = model.predict(input_scaled)[0]
            prediction_finale = max(0, round(prediction, 1))
            
            # Affichage de la boîte de prédiction principale
            st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size: 1.1rem; color: #1A5276; margin: 0; font-weight: bold;">
                    Estimation des Cas Confirmés ({model_choice})
                </p>
                <p class="prediction-number">{prediction_finale} cas</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🔍 Importance Relative des Variables")
            
            # Extraction des importances ou poids selon le modèle choisi
            importances = None
            coef_type = ""
            
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                coef_type = "Importance Spécifique"
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
                coef_type = "Magnitude du Coefficient (Absolu)"
                
            if importances is not None:
                # Création du graphique d'importance
                imp_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                
                fig = px.bar(
                    imp_df, x='Importance', y='Variable', orientation='h',
                    labels={'Importance': coef_type, 'Variable': 'Variables Clés'},
                    color='Importance', color_continuous_scale='Blugrn', height=340
                )
                fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Bloc d'interprétation contextuelle dynamique
                st.markdown("** Interprétation Statistique :**")
                if "Boosting" in model_choice or "Forest" in model_choice:
                    st.info(f"Le modèle **{model_choice}** capture des interactions non linéaires. Les variables en haut du graphique indiquent les leviers qui provoquent de fortes ruptures ou des effets de seuil sur l'émergence épidémique.")
                else:
                    st.info(f"Les coefficients de la **{model_choice}** traduisent un impact direct et proportionnel. Une hausse des variables à fort coefficient augmente linéairement la prédiction du volume de cas.")
            else:
                st.warning("L'importance des variables n'est pas disponible pour ce type d'estimateur.")
    else:
        st.info("💡 Modifiez les curseurs ou les valeurs numériques à gauche, puis cliquez sur le bouton pour calculer la prédiction et afficher l'analyse d'impact des critères.")

st.markdown("---")
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption(" Application conçue pour l'aide à la décision face aux urgences de santé publique — RDC")
with col_f2:
    st.caption("[📂 Code Source GitHub](https://github.com/SylvieMpwek/Projet_Mpox)")