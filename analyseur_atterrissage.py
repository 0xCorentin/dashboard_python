# -*- coding: utf-8 -*-
"""
Analyseur Atterrissage
Module dédié à l'analyse d'atterrissage avec TX de réalisation et prévisions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Import des fonctions utilitaires
from utils import load_landing_data

def create_landing_visualization(df):
    """Crée la visualisation d'atterrissage avec barres multiples par période"""
    
    st.markdown("### 🎯 Analyse d'Atterrissage par Région")
    
    # Filtrer les données pour enlever les totaux
    df_filtered = df[~df['REGION'].str.contains('total|Total|TOTAL|ENSEMBLE', case=False, na=False)].copy()
    df_filtered = df_filtered.dropna(subset=['REGION'])
    
    # Options de configuration
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            df_filtered['REGION'].unique().tolist(),
            default=df_filtered['REGION'].unique().tolist(),
            key="landing_regions"
        )
    
    with col_config2:
        sort_order = st.selectbox(
            "📊 Ordre d'affichage:",
            ["TX Septembre décroissant", "TX Décembre décroissant", "Alphabétique", "Reste à faire décroissant"],
            key="landing_sort"
        )
    
    # Filtrer selon les sélections
    df_viz = df_filtered[df_filtered['REGION'].isin(regions_to_show)].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Tri selon la sélection
    if sort_order == "TX Septembre décroissant":
        df_viz = df_viz.sort_values('TX DE REALISATION A FIN SEPTEMBRE', ascending=False)
    elif sort_order == "TX Décembre décroissant":
        df_viz = df_viz.sort_values('TX DE REALISATION /AU BUDGET A FIN DECEMBRE', ascending=False)
    elif sort_order == "Alphabétique":
        df_viz = df_viz.sort_values('REGION', ascending=True)
    elif sort_order == "Reste à faire décroissant":
        df_viz = df_viz.sort_values('RESTE A FAIRE / NOUVELLES ENTREES', ascending=False)
    
    # Créer le graphique avec barres multiples
    fig = go.Figure()
    
    regions = df_viz['REGION'].tolist()
    
    # Barre 1: TX de réalisation à fin septembre (en %)
    tx_septembre = (df_viz['TX DE REALISATION A FIN SEPTEMBRE'] * 100).tolist()
    
    # Barre 2: TX de réalisation à fin décembre (en %)
    tx_decembre = (df_viz['TX DE REALISATION /AU BUDGET A FIN DECEMBRE'] * 100).tolist()
    
    # Barre 3: Reste à faire - garder les valeurs brutes
    reste_a_faire = df_viz['RESTE A FAIRE / NOUVELLES ENTREES'].tolist()
    
    # Ajouter les barres TX avec axe Y principal (pourcentages)
    fig.add_trace(go.Bar(
        x=regions,
        y=tx_septembre,
        name='TX Réalisation Septembre (%)',
        marker_color='#3498db',  # Bleu
        text=[f"{val:.1f}%" for val in tx_septembre],
        textposition='outside',
        yaxis='y',
        offsetgroup=0  # Premier groupe
    ))
    
    fig.add_trace(go.Bar(
        x=regions,
        y=tx_decembre,
        name='TX Réalisation Décembre (%)',
        marker_color='#e74c3c',  # Rouge
        text=[f"{val:.1f}%" for val in tx_decembre],
        textposition='outside',
        yaxis='y',
        offsetgroup=1  # Deuxième groupe
    ))
    
    # Reste à faire avec axe Y secondaire (valeurs brutes)
    fig.add_trace(go.Bar(
        x=regions,
        y=reste_a_faire,
        name='Reste à Faire (valeurs)',
        marker_color='#f39c12',  # Orange
        text=[f"{val:,.0f}" for val in reste_a_faire],
        textposition='outside',
        yaxis='y2',  # Axe secondaire pour les valeurs brutes
        offsetgroup=2  # Groupe séparé pour le positionnement
    ))
    
    # Configuration du graphique avec double axe Y
    fig.update_layout(
        title="🎯 Analyse d'Atterrissage : TX Réalisation et Reste à Faire par Région",
        xaxis_title="Régions",
        yaxis=dict(
            title="Taux de Réalisation (%)",
            tickformat=".1f",
            ticksuffix="%",
            side='left'
        ),
        yaxis2=dict(
            title="Reste à Faire (valeurs)",
            overlaying='y',
            side='right',
            tickformat=',.0f'
        ),
        height=700,
        barmode='group',  # Barres groupées côte à côte
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques récapitulatives
    create_landing_statistics(df_viz)

def create_landing_statistics(df):
    """Crée les statistiques récapitulatives pour l'atterrissage"""
    
    st.markdown("### 📊 Statistiques d'Atterrissage")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    # Moyennes
    avg_tx_sept = df['TX DE REALISATION A FIN SEPTEMBRE'].mean() * 100
    avg_tx_dec = df['TX DE REALISATION /AU BUDGET A FIN DECEMBRE'].mean() * 100
    
    # Calculer le reste à faire moyen en valeurs brutes
    avg_reste_brut = df['RESTE A FAIRE / NOUVELLES ENTREES'].mean()
    total_reste_brut = df['RESTE A FAIRE / NOUVELLES ENTREES'].sum()
    
    # Meilleure région septembre
    best_sept_idx = df['TX DE REALISATION A FIN SEPTEMBRE'].idxmax()
    best_sept_region = df.loc[best_sept_idx, 'REGION']
    best_sept_val = df.loc[best_sept_idx, 'TX DE REALISATION A FIN SEPTEMBRE'] * 100
    
    with col_stats1:
        st.metric(
            "📊 TX Moyen Septembre",
            f"{avg_tx_sept:.1f}%",
            help="Taux de réalisation moyen à fin septembre"
        )
    
    with col_stats2:
        st.metric(
            "📈 TX Moyen Décembre",
            f"{avg_tx_dec:.1f}%",
            f"+{avg_tx_dec - avg_tx_sept:.1f}% vs Sept"
        )
    
    with col_stats3:
        st.metric(
            "🏆 Meilleure Région Sept",
            best_sept_region[:15] + "..." if len(best_sept_region) > 15 else best_sept_region,
            f"{best_sept_val:.1f}%"
        )
    
    with col_stats4:
        st.metric(
            "📋 Reste à Faire Moyen",
            f"{avg_reste_brut:,.0f}",
            help="Valeur moyenne du reste à faire (valeurs brutes)"
        )
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région"):
        # Préparer les données pour l'affichage
        display_data = df.copy()
        display_data['TX Sept (%)'] = display_data['TX DE REALISATION A FIN SEPTEMBRE'] * 100
        display_data['TX Déc (%)'] = display_data['TX DE REALISATION /AU BUDGET A FIN DECEMBRE'] * 100
        display_data['Écart Sep-Déc'] = display_data['TX Déc (%)'] - display_data['TX Sept (%)']
        
        # Ajouter le reste à faire en valeurs brutes
        display_data['Reste à Faire (valeurs)'] = display_data['RESTE A FAIRE / NOUVELLES ENTREES']
        
        # Sélectionner les colonnes à afficher
        columns_to_show = [
            'REGION', 'TX Sept (%)', 'TX Déc (%)', 'Écart Sep-Déc', 'Reste à Faire (valeurs)'
        ]
        
        display_data_filtered = display_data[columns_to_show].copy()
        display_data_filtered = display_data_filtered.sort_values('TX Sept (%)', ascending=False)
        
        # Configuration des colonnes
        column_config = {
            'REGION': 'Région',
            'TX Sept (%)': st.column_config.NumberColumn(
                'TX Sept (%)',
                format="%.1f%%"
            ),
            'TX Déc (%)': st.column_config.NumberColumn(
                'TX Déc (%)',
                format="%.1f%%"
            ),
            'Écart Sep-Déc': st.column_config.NumberColumn(
                'Écart Sep-Déc',
                format="%.1f%%",
                help="Différence entre TX Décembre et TX Septembre"
            ),
            'Reste à Faire (valeurs)': st.column_config.NumberColumn(
                'Reste à Faire (valeurs)',
                format="%.0f",
                help="Valeurs brutes du reste à faire"
            )
        }
        
        st.dataframe(
            display_data_filtered,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )

def show_landing_analysis():
    """Page de l'analyseur d'atterrissage"""
    
    # Titre principal
    st.markdown('<h1 class="main-title">🎯 Analyseur Atterrissage - Prévisions & Réalisations</h1>', 
                unsafe_allow_html=True)
    
    # Section d'import de fichier
    st.markdown("## 📁 Import de Fichier")
    st.markdown("*Choisissez le fichier Excel d'atterrissage à analyser*")
    
    # Options d'import
    import_col1, import_col2 = st.columns(2)
    
    with import_col1:
        import_method = st.radio(
            "🔧 Méthode d'import:",
            ["Upload d'un nouveau fichier", "Chemin personnalisé"],
            key="landing_import_method"
        )
    
    uploaded_file = None
    custom_path = None
    file_info = ""
    
    if import_method == "Upload d'un nouveau fichier":
        with import_col2:
            uploaded_file = st.file_uploader(
                "📤 Choisir un fichier Excel:",
                type=['xlsx', 'xls'],
                help="Sélectionnez un fichier Excel (.xlsx ou .xls) avec les données d'atterrissage",
                key="landing_file_uploader"
            )
            if uploaded_file is not None:
                file_info = f"📁 Fichier: {uploaded_file.name}"
    
    elif import_method == "Chemin personnalisé":
        with import_col2:
            custom_path = st.text_input(
                "📂 Chemin du fichier:",
                placeholder="Ex: C:/Dashboard/atterrissage.xlsx",
                help="Entrez le chemin complet vers votre fichier Excel d'atterrissage",
                key="landing_custom_path"
            )
            if custom_path:
                file_info = f"📁 Fichier: {custom_path}"
            else:
                file_info = "📁 Veuillez spécifier un chemin de fichier"
    
    # Affichage des informations du fichier
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
        <h3>🎯 Analyse d'Atterrissage : TX Réalisation et Prévisions par Région</h3>
        <p><strong>{file_info}</strong> | <strong>Dernière mise à jour:</strong> {datetime.now().strftime("%d/%m/%Y à %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données selon la méthode choisie
    if import_method == "Upload d'un nouveau fichier" and uploaded_file is None:
        st.warning("⚠️ Veuillez sélectionner un fichier Excel à analyser")
        return
    elif import_method == "Chemin personnalisé" and not custom_path:
        st.warning("⚠️ Veuillez entrer le chemin vers votre fichier Excel")
        return
    
    # Chargement des données
    if import_method == "Upload d'un nouveau fichier":
        df, error = load_landing_data(uploaded_file=uploaded_file)
    else:  # Chemin personnalisé
        df, error = load_landing_data(file_path=custom_path)
    
    if error:
        st.error(error)
        
        # Messages d'aide selon la méthode d'import
        if import_method == "Upload d'un nouveau fichier":
            st.info("💡 Vérifiez que votre fichier Excel n'est pas corrompu et contient des données d'atterrissage")
        else:
            st.info("💡 Vérifiez le chemin du fichier et qu'il existe bien sur votre système")
        
        st.markdown("### 📋 Format de Fichier Attendu")
        st.markdown("""
        **Le fichier Excel doit contenir :**
        - Une colonne 'Régions' ou 'REGION' avec les noms des régions
        - Une colonne 'TX DE REALISATION A FIN SEPTEMBRE' avec les taux (0-1)
        - Une colonne 'TX DE REALISATION /AU BUDGET A FIN DECEMBRE' avec les taux (0-1)
        - Une colonne 'RESTE A FAIRE / NOUVELLES ENTREES' avec les valeurs numériques
        - **Format des taux** : Décimaux entre 0 et 1 (ex: 0.85 pour 85%)
        """)
        return
    
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible dans le fichier")
        return
    
    # Validation du format des données
    st.success(f"✅ Fichier chargé avec succès : {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Vérifications de compatibilité
    warnings = []
    if 'REGION' not in df.columns:
        warnings.append("❌ Colonne 'REGION' ou 'Régions' non trouvée - impossible de créer les analyses")
    
    required_cols = [
        'TX DE REALISATION A FIN SEPTEMBRE',
        'TX DE REALISATION /AU BUDGET A FIN DECEMBRE',
        'RESTE A FAIRE / NOUVELLES ENTREES'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            warnings.append(f"❌ Colonne '{col}' non trouvée - impossible de créer les analyses")
        
    if warnings:
        for warning in warnings:
            st.error(warning)
        
        st.info("💡 Votre fichier doit contenir toutes les colonnes requises pour l'analyse d'atterrissage")
        
        # Afficher les colonnes disponibles pour aider l'utilisateur
        with st.expander("📋 Colonnes Disponibles dans votre Fichier"):
            st.write("Colonnes trouvées:")
            for col in df.columns:
                st.write(f"• {col}")
        return
    
    # Sidebar avec informations sur les données
    with st.sidebar:
        st.header("📋 Informations du Dataset")
        
        regions_count = df['REGION'].dropna().nunique()
        
        st.metric("🏢 Nombre de Régions", regions_count)
        st.metric("📊 Lignes de Données", len(df))
        
        # Statistiques rapides
        avg_tx_sept = df['TX DE REALISATION A FIN SEPTEMBRE'].mean() * 100
        avg_tx_dec = df['TX DE REALISATION /AU BUDGET A FIN DECEMBRE'].mean() * 100
        
        st.subheader("📊 Aperçu Rapide")
        st.metric("TX Moyen Sept", f"{avg_tx_sept:.1f}%")
        st.metric("TX Moyen Déc", f"{avg_tx_dec:.1f}%")
        st.metric("Évolution", f"+{avg_tx_dec - avg_tx_sept:.1f}%")
    
    # Génération de l'analyse
    st.markdown("---")
    try:
        create_landing_visualization(df)
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des graphiques: {str(e)}")
        
        # Debug info
        with st.expander("🔧 Informations de Debug"):
            st.write("Colonnes disponibles:", df.columns.tolist())
            st.write("Forme du DataFrame:", df.shape)
            st.write("Types de données:", df.dtypes.to_dict())
            st.write("Échantillon de données:", df.head())