# -*- coding: utf-8 -*-
"""
Main - Hub d'Analyse Dashboard Production
Application principale avec navigation entre les 4 analyseurs
"""
import streamlit as st
from datetime import datetime

# Import du module pour charger le CSS externe
from css_loader import apply_custom_css

# Imports des analyseurs
from analyseur_optimisation_plateaux import show_weekly_analysis
from analyseur_production_mensuelle import show_monthly_production
from analyseur_atterrissage import show_landing_analysis
from analyseur_entrees_sorties import show_entrees_sorties_analysis

# ========== CONFIGURATION DE LA PAGE ==========
st.set_page_config(
    page_title="🏠 Hub d'Analyse - Dashboard Production",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== STYLES CSS EXTERNES ==========
# Chargement des styles depuis le fichier externe styles.css
apply_custom_css("styles.css")

def show_home_page():
    """Page d'accueil - Hub de navigation"""
    
    st.markdown('<h1 class="hub-title">🏠 Hub d\'Analyse - Dashboard Production</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <h3>🎯 Bienvenue dans votre centre d'analyse de production</h3>
        <p style='font-size: 1.2rem; color: #666;'>
            Choisissez l'outil d'analyse adapté à vos besoins
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cartes des différents outils avec alignement amélioré
    st.markdown("""
    <style>
    .button-container {
        margin-bottom: 10px;
    }
    .card-spacing {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Carte cliquable pour l'Optimisation des plateaux
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        weekly_clicked = st.button(
            label="📊 Optimisation des plateaux - Cliquez pour lancer",
            key="weekly_card_btn",
            help="Cliquer pour accéder à l'Optimisation des plateaux",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Affichage de la carte avec les détails
        st.markdown("""
        <div class="hub-card hub-card-primary card-spacing">
            <h2>📊 Optimisation des plateaux</h2>
            <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
            <p><strong>✨ Fonctionnalités :</strong></p>
            <ul style="text-align: left; margin: 1rem 0;">
                <li>📈 Comparaison performance par région</li>
                <li>📊 Évolution mensuelle avec barres</li>
                <li>🏆 Classification automatique des régions</li>
                <li>📋 Insights de performance détaillés</li>
                <li>🎯 Analyse des écarts et moyennes</li>
            </ul>
            <p><strong>📁 Formats supportés :</strong> Excel (.xlsx, .xls)</p>
            <p><strong>🎨 Visualisations :</strong> Barres, Heatmap, Tendances</p>
            <p style="margin-top: 1.5rem; font-weight: bold; color: #fff3cd;">👆 Cliquez sur le bouton ci-dessus pour lancer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action du clic sur la carte
        if weekly_clicked:
            st.session_state.current_page = "weekly_analysis"
            st.rerun()
        
        # Carte cliquable pour l'Analyseur Atterrissage
        st.markdown('<div class="button-container-bottom">', unsafe_allow_html=True)
        landing_clicked = st.button(
            label="🎯 Analyseur Atterrissage - Cliquez pour lancer",
            key="landing_card_btn",
            help="Cliquer pour accéder à l'Analyseur d'Atterrissage",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="hub-card hub-card-quaternary card-bottom-alignment">
            <h2>🎯 Analyseur Atterrissage</h2>
            <h4>📊 Prévisions & Réalisations</h4>
            <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
            <p><strong>✨ Fonctionnalités :</strong></p>
            <ul style="text-align: left; margin: 1rem 0;">
                <li>📊 TX de réalisation par région</li>
                <li>📈 Prévisions d'atterrissage</li>
                <li>📋 Comparaison Mensuel</li>
                <li>🎯 Reste à faire par région</li>
                <li>📊 Barres multiples par période</li>
            </ul>
            <p><strong>📁 Import :</strong> Upload Excel ou chemin personnalisé</p>
            <p><strong>🎨 Visualisations :</strong> Barres comparatives, Analyses</p>
            <p style="margin-top: 1.5rem; font-weight: bold; color: #fff3cd;">👆 Cliquez sur le bouton ci-dessus pour lancer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action du clic sur la carte
        if landing_clicked:
            st.session_state.current_page = "landing_analysis"
            st.rerun()
    
    with col2:
        # Carte cliquable pour la Production Mensuelle
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        monthly_prod_clicked = st.button(
            label="📈 Analyseur Production Mensuelle - Cliquez pour lancer",
            key="monthly_prod_card_btn",
            help="Cliquer pour accéder à l'Analyseur de Production Mensuelle",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Affichage de la carte avec les détails
        st.markdown("""
        <div class="hub-card hub-card-tertiary card-spacing">
            <h2>📈 Production Mensuelle</h2>
            <h4>📊 Analyse PROD HORS PAE</h4>
            <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
            <p><strong>✨ Fonctionnalités :</strong></p>
            <ul style="text-align: left; margin: 1rem 0;">
                <li>📊 Analyse par région et financeur</li>
                <li>📈 Évolution mensuelle avec barres</li>
                <li>📋 Courbe de cumul sur le total</li>
                <li>🎯 Classification des performances</li>
                <li>📋 Insights détaillés par région</li>
            </ul>
            <p><strong>📁 Import :</strong> Upload Excel ou chemin personnalisé</p>
            <p><strong>🎨 Visualisations :</strong> Barres par financeur, Cumul</p>
            <p style="margin-top: 1.5rem; font-weight: bold; color: #fff3cd;">👆 Cliquez sur le bouton ci-dessus pour lancer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action du clic sur la carte
        if monthly_prod_clicked:
            st.session_state.current_page = "monthly_production"
            st.rerun()
        
        # Carte cliquable pour l'Analyseur Entrées-Sorties-Abandons (NOUVELLE CARTE)
        st.markdown('<div class="button-container-bottom">', unsafe_allow_html=True)
        entrees_sorties_clicked = st.button(
            label="🔄 Analyseur Entrées-Sorties-Abandons - Cliquez pour lancer",
            key="entrees_sorties_card_btn",
            help="Cliquer pour accéder à l'Analyseur Entrées-Sorties-Abandons",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="hub-card card-bottom-alignment" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h2>🔄 Entrées-Sorties-Abandons</h2>
            <h4>📊 Analyse des Flux par Financeur</h4>
            <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
            <p><strong>✨ Fonctionnalités :</strong></p>
            <ul style="text-align: left; margin: 1rem 0;">
                <li>📥 Analyse des entrées (SE) par région</li>
                <li>📤 Suivi des sorties (SS) par financeur</li>
                <li>❌ Monitoring des abandons par région</li>
                <li>💪 Calcul automatique du taux de rétention</li>
                <li>📊 Barres groupées par financeur</li>
                <li>🎯 Vue comparative multi-métriques</li>
            </ul>
            <p><strong>📁 Import :</strong> Upload Excel ou chemin personnalisé</p>
            <p><strong>🎨 Visualisations :</strong> Barres groupées, Statistiques détaillées</p>
            <p style="margin-top: 1.5rem; font-weight: bold; color: #fff3cd;">👆 Cliquez sur le bouton ci-dessus pour lancer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action du clic sur la carte
        if entrees_sorties_clicked:
            st.session_state.current_page = "entrees_sorties_analysis"
            st.rerun()

def main():
    """Fonction principale avec navigation entre les pages"""
    
    # Initialisation de la session state pour la navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Sidebar avec navigation
    with st.sidebar:
        st.markdown("# 🏠 Navigation")
        
        # Boutons de navigation
        if st.button("🏠 Accueil", key="nav_home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()
        
        if st.button("📊 Optimisation des plateaux", key="nav_weekly", use_container_width=True):
            st.session_state.current_page = 'weekly_analysis'
            st.rerun()
            
        if st.button("📈 Analyseur Production Mensuelle", key="nav_monthly_prod", use_container_width=True):
            st.session_state.current_page = 'monthly_production'
            st.rerun()
            
        if st.button("🎯 Analyseur Atterrissage", key="nav_landing", use_container_width=True):
            st.session_state.current_page = 'landing_analysis'
            st.rerun()
        
        if st.button("🔄 Analyseur Entrées-Sorties-Abandons", key="nav_entrees_sorties", use_container_width=True):
            st.session_state.current_page = 'entrees_sorties_analysis'
            st.rerun()
        
        st.markdown("---")
        
        # Informations sur la page actuelle
        pages_info = {
            'home': "🏠 Hub d'Analyse",
            'weekly_analysis': "📊 Optimisation des plateaux",
            'monthly_production': "📈 Analyseur Production Mensuelle",
            'landing_analysis': "🎯 Analyseur Atterrissage",
            'entrees_sorties_analysis': "🔄 Entrées-Sorties-Abandons"
        }
        
        current_page_name = pages_info.get(st.session_state.current_page, "Page inconnue")
        st.info(f"**Page actuelle :**\n{current_page_name}")
        
        # Footer avec informations
        st.markdown("---")
        st.markdown("### 📋 Informations")
        st.markdown("""
        **🔧 Version :** v2.3 Hub  
        **📅 Date :** Octobre 2025   
        **📊 Outils :** 4 analyseurs  
        """)
    
    # Affichage de la page selon la sélection
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'weekly_analysis':
        show_weekly_analysis()
    elif st.session_state.current_page == 'monthly_production':
        show_monthly_production()
    elif st.session_state.current_page == 'landing_analysis':
        show_landing_analysis()
    elif st.session_state.current_page == 'entrees_sorties_analysis':
        show_entrees_sorties_analysis()
    else:
        st.error("❌ Page non trouvée")
        st.session_state.current_page = 'home'
        st.rerun()

if __name__ == "__main__":
    main()