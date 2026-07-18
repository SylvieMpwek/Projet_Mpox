# Modélisation Épidémiologique et Prédiction de Mpox en RDC 🇨🇩

##  Description
Application de prédiction du nombre de cas confirmés de Mpox (Monkeypox) en République Démocratique du Congo. Ce projet s'appuie sur des modèles de Machine Learning (Régression Linéaire Multiple, Ridge, Lasso, Random Forest et Gradient Boosting) pour estimer la dynamique de transmission en fonction de facteurs environnementaux, climatiques, démographiques et sanitaires.

##  Fonctionnalités
- **Sélection dynamique du modèle** : Choix entre 5 algorithmes d'apprentissage dans l'interface (Régression Linéaire, Ridge, Lasso, Random Forest, Gradient Boosting).
- **Validation croisée robuste** : Évaluation des performances via une stratégie `StratifiedKFold` à 5 plis basée sur les provinces.
- **Prédiction en temps réel** : Interface utilisateur intuitive développée avec Streamlit pour simuler des scénarios épidémiologiques.
- **Sécurisation des saisies** : Contrôle automatique des valeurs aberrantes avant exécution des prédictions.

## Variables utilisées
L'application utilise 11 variables explicatives pour prédire la variable cible (**Cas_Confirmes**) :

| Catégorie | Variable | Description |
| :--- | :--- | :--- |
| **Climatique & Environnementale** | `Pluviometrie_mm` | Précipitations enregistrées (en mm) |
| | `Temperature_C` | Température moyenne (en °C) |
| | `NDVI` | Indice de végétation par télédétection (0 à 1) |
| | `Humidite_pct` | Taux d'humidité relative (en %) |
| | `Saison_num` | Code numérique de la saison (0 : Sèche, 1 : Pluies) |
| **Démographique & Spatiale** | `Densite_Population` | Densité de la population locale (hab/km²) |
| | `Population_Risque` | Taille de la population exposée au risque |
| | `Mobilite_Humaine` | Flux ou indice de mobilité de la population |
| | `Distance_Centre_Sante_km` | Distance moyenne menant au centre de santé (en km) |
| **Sanitaire & Écologique** | `Tests_Realises` | Nombre de tests diagnostiques effectués |
| | `Couverture_Vaccinale_pct` | Taux de couverture vaccinale de la zone (en %) |
| | `Reservoirs_Animaux` | Densité/index de présence des réservoirs (rongeurs, primates) |

## Installation

### Prérequis
- Python 3.8 ou supérieur
- Git

### Étapes d'installation

```bash
# 1. Cloner le dépôt et se placer dans le dossier du projet
git clone [https://github.com/SylvieMpwek/Projet_Mpox.git](https://github.com/SylvieMpwek/Projet_Mpox.git)
cd Projet_Mpox

# 2. Créer et activer un environnement virtuel
python -m venv venv
# Sur Linux/Mac :
source venv/bin/activate  
# Sur Windows (PowerShell) :
.\venv\Scripts\Activate.ps1

# 3. Installer les dépendances requises
pip install -r requirements.txt

# 4. Entraîner les modèles et générer les fichiers (.pkl)
python train_model.py

# 5. Lancer l'application interactive Streamlit
streamlit run app.py