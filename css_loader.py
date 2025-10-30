# -*- coding: utf-8 -*-
import os

def load_css_file(css_file_path):
    """
    Charge le contenu d'un fichier CSS et retourne le HTML pour l'intégrer dans Streamlit
    """
    try:
        # Construire le chemin absolu du fichier CSS
        if not os.path.isabs(css_file_path):
            # Si le chemin est relatif, le construire par rapport au fichier actuel
            current_dir = os.path.dirname(os.path.abspath(__file__))
            css_file_path = os.path.join(current_dir, css_file_path)
        
        # Lire le contenu du fichier CSS
        with open(css_file_path, 'r', encoding='utf-8') as css_file:
            css_content = css_file.read()
        
        # Retourner le HTML avec les styles
        return f"""
<style>
{css_content}
</style>
"""
    except FileNotFoundError:
        print(f"⚠️ Fichier CSS non trouvé : {css_file_path}")
        return "<style>/* CSS file not found */</style>"
    except Exception as e:
        print(f"❌ Erreur lors du chargement du CSS : {str(e)}")
        return "<style>/* CSS loading error */</style>"

def apply_custom_css(css_file_path="styles.css"):
    """
    Fonction principale pour appliquer les styles CSS personnalisés
    """
    import streamlit as st
    
    css_html = load_css_file(css_file_path)
    st.markdown(css_html, unsafe_allow_html=True)