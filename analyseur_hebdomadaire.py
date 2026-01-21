# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import du module pour charger le CSS externe
from css_loader import apply_custom_css

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

# ========== FONCTIONS UTILITAIRES ==========

def compute_totals(df, regions, months, metrics, region_col='Region'):
    """Calcule les totaux par région et par métrique sur les mois donnés."""
    results = []
    for region in regions:
        region_data = df[df[region_col] == region]
        if not region_data.empty:
            region_row = region_data.iloc[0]
            region_totals = {'Région': region}
            for metric in metrics:
                total_metric = 0
                for month in months:
                    col_name = f'{month}_{metric}'
                    if col_name in region_row.index:
                        total_metric += region_row[col_name]
                region_totals[metric] = int(total_metric)
            results.append(region_totals)
    return results

@st.cache_data(ttl=600)  # Cache 10 minutes
def load_weekly_data(file_path=None, uploaded_file=None):
    """Charge et nettoie les données hebdomadaires depuis un fichier ou upload"""
    try:
        if uploaded_file is not None:
            # Fichier uploadé via Streamlit
            df = pd.read_excel(uploaded_file)
        else:
            # Fichier spécifique fourni via chemin
            df = pd.read_excel(file_path)
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Supprimer les lignes "Total général" ou similaires
        df = df[~df['REGION'].str.contains('total|Total|TOTAL', case=False, na=False)]
        
        return df, None
    except FileNotFoundError as e:
        if uploaded_file is not None:
            return None, "❌ Erreur lors de la lecture du fichier uploadé"
        else:
            return None, f"❌ Fichier '{file_path}' non trouvé"
    except Exception as e:
        return None, f"❌ Erreur lors du chargement: {str(e)}"

def aggregate_data_smart(df, cat_col, val_col, method):
    """Agrégation intelligente des données pour les comparaisons"""
    try:
        if method == "mean":
            result = df.groupby(cat_col)[val_col].mean().reset_index()
        elif method == "sum":
            result = df.groupby(cat_col)[val_col].sum().reset_index()
        elif method == "count":
            result = df.groupby(cat_col)[val_col].count().reset_index()
        elif method == "max":
            result = df.groupby(cat_col)[val_col].max().reset_index()
        elif method == "min":
            result = df.groupby(cat_col)[val_col].min().reset_index()
        else:
            result = df.groupby(cat_col)[val_col].mean().reset_index()
        
        # Tri par valeur décroissante
        result = result.sort_values(val_col, ascending=False)
        
        return result
    except Exception as e:
        st.error(f"Erreur d'agrégation: {e}")
        return pd.DataFrame()

def calculate_performance_insights(df, value_col):
    """Calcule des insights de performance"""
    insights = {}
    
    if len(df) > 0:
        insights['top_performer'] = df.iloc[0]['REGION'] if 'REGION' in df.columns else "N/A"
        insights['top_value'] = df.iloc[0][value_col] if len(df) > 0 else 0
        insights['bottom_performer'] = df.iloc[-1]['REGION'] if 'REGION' in df.columns else "N/A" 
        insights['bottom_value'] = df.iloc[-1][value_col] if len(df) > 0 else 0
        insights['average'] = df[value_col].mean()
        insights['total_regions'] = len(df)
        insights['above_average'] = len(df[df[value_col] > insights['average']])
        insights['performance_gap'] = insights['top_value'] - insights['bottom_value']
        
        # Classification des régions
        q1 = df[value_col].quantile(0.25)
        q2 = df[value_col].quantile(0.50)
        q3 = df[value_col].quantile(0.75)
        
        insights['excellent'] = len(df[df[value_col] >= q3])
        insights['good'] = len(df[(df[value_col] >= q2) & (df[value_col] < q3)])
        insights['fair'] = len(df[(df[value_col] >= q1) & (df[value_col] < q2)])
        insights['poor'] = len(df[df[value_col] < q1])
    
    return insights

def create_comparison_visualization(df, category_col, value_col, agg_method):
    """Crée la visualisation de comparaison optimisée pour l'analyse hebdomadaire"""
    
    # Agrégation des données
    agg_data = aggregate_data_smart(df, category_col, value_col, agg_method)
    
    if agg_data.empty:
        st.warning("⚠️ Aucune donnée à afficher après agrégation")
        return
    
    # Mapping des méthodes pour affichage
    title_mapping = {
        "sum": "Total",
        "mean": "Moyenne", 
        "count": "Nombre",
        "max": "Maximum",
        "min": "Minimum"
    }
    
    # Options de focus et filtrage
    st.markdown("### 🎯 Options d'Affichage")
    
    col_focus1, col_focus2, col_focus3 = st.columns(3)
    
    with col_focus1:
        focus_type = st.selectbox(
            "📍 Type de vue:",
            ["Toutes les régions", "Top 10", "Top 5", "Régions spécifiques"],
            key="focus_type_weekly"
        )
    
    # Appliquer le filtrage selon le type choisi
    if focus_type == "Top 10":
        agg_data_display = agg_data.head(10)
        focus_info = f"📊 Affichage du Top 10 sur {len(agg_data)} régions"
    elif focus_type == "Top 5":
        agg_data_display = agg_data.head(5)
        focus_info = f"📊 Affichage du Top 5 sur {len(agg_data)} régions"
    elif focus_type == "Régions spécifiques":
        with col_focus2:
            available_regions = agg_data[category_col].tolist()
            selected_regions = st.multiselect(
                "🏷️ Sélectionner les régions:",
                available_regions,
                default=available_regions[:5],
                key="selected_regions_weekly"
            )
        
        if selected_regions:
            agg_data_display = agg_data[agg_data[category_col].isin(selected_regions)]
            focus_info = f"📊 Affichage de {len(selected_regions)} régions sélectionnées"
        else:
            agg_data_display = agg_data.head(5)
            focus_info = "⚠️ Aucune région sélectionnée, affichage du Top 5"
    else:
        agg_data_display = agg_data
        focus_info = f"📊 Affichage de toutes les {len(agg_data)} régions"
    
    with col_focus3:
        sort_order = st.selectbox(
            "📊 Ordre:",
            ["Performance décroissante", "Performance croissante", "Alphabétique"],
            key="sort_weekly"
        )
    
    # Appliquer l'ordre de tri
    if sort_order == "Performance croissante":
        agg_data_display = agg_data_display.sort_values(value_col, ascending=True)
    elif sort_order == "Alphabétique":
        agg_data_display = agg_data_display.sort_values(category_col, ascending=True)
    
    # Affichage des informations
    st.info(focus_info)
    
    # Création du graphique principal
    title = f"{title_mapping[agg_method]} de {value_col} par {category_col}"
    
    fig = px.bar(
        agg_data_display, 
        x=category_col, 
        y=value_col,
        title=title,
        color=value_col,
        color_continuous_scale="Viridis",
        text=value_col
    )
    
    # Personnalisation du graphique
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
        height=600,
        showlegend=False,
        xaxis_title="Régions",
        yaxis_title=value_col
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights de performance
    insights = calculate_performance_insights(agg_data_display, value_col)
    
    # Affichage des métriques clés
    st.markdown("### 📈 Insights de Performance")
    
    col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
    
    with col_insight1:
        st.metric(
            "🥇 Meilleure Région",
            insights['top_performer'],
            f"{insights['top_value']:.2f}"
        )
    
    with col_insight2:
        st.metric(
            "📊 Moyenne Générale",
            f"{insights['average']:.2f}",
            f"{insights['above_average']}/{insights['total_regions']} au-dessus"
        )
    
    with col_insight3:
        st.metric(
            "📉 Région à Améliorer",
            insights['bottom_performer'],
            f"{insights['bottom_value']:.2f}"
        )
    
    with col_insight4:
        st.metric(
            "📏 Écart Performance",
            f"{insights['performance_gap']:.2f}",
            f"{((insights['performance_gap']/insights['average'])*100):.1f}% vs moyenne"
        )
    
    # Classification des performances
    st.markdown("### 🎯 Classification des Régions")
    
    class_col1, class_col2, class_col3, class_col4 = st.columns(4)
    
    with class_col1:
        st.markdown(f"""
        <div class="alert-success">
            <h4>🟢 Excellence</h4>
            <h2>{insights['excellent']}</h2>
            <small>régions (Q4)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with class_col2:
        st.markdown(f"""
        <div style="background-color: #e7f3ff; border: 1px solid #b3d9ff; color: #0056b3; padding: 1rem; border-radius: 8px; text-align: center;">
            <h4>🔵 Bonne</h4>
            <h2>{insights['good']}</h2>
            <small>régions (Q3)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with class_col3:
        st.markdown(f"""
        <div class="alert-warning">
            <h4>🟡 Moyenne</h4>
            <h2>{insights['fair']}</h2>
            <small>régions (Q2)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with class_col4:
        st.markdown(f"""
        <div class="alert-danger">
            <h4>🔴 À Améliorer</h4>
            <h2>{insights['poor']}</h2>
            <small>régions (Q1)</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées"):
        # Ajout de colonnes calculées
        display_data = agg_data_display.copy()
        display_data['Écart à la Moyenne'] = display_data[value_col] - insights['average']
        display_data['% vs Moyenne'] = ((display_data[value_col] / insights['average']) - 1) * 100
        
        # Classification
        q1 = agg_data[value_col].quantile(0.25)
        q2 = agg_data[value_col].quantile(0.50) 
        q3 = agg_data[value_col].quantile(0.75)
        
        def classify_performance(value):
            if value >= q3:
                return "🟢 Excellence"
            elif value >= q2:
                return "🔵 Bonne"
            elif value >= q1:
                return "🟡 Moyenne"
            else:
                return "🔴 À Améliorer"
        
        display_data['Classification'] = display_data[value_col].apply(classify_performance)
        
        # Configuration des colonnes selon le type de données
        if 'TX' in value_col.upper():
            # Pour les colonnes TX (pourcentages), utiliser ProgressColumn avec format %
            value_config = st.column_config.ProgressColumn(
                value_col,
                help=f"Taux de capacité - {value_col}",
                min_value=0,
                max_value=1,  # Les TX sont en décimal (0-1)
                format="%.1%%"
            )
            ecart_format = "%.4f"
        else:
            # Pour les colonnes REALISE (valeurs absolues), utiliser NumberColumn
            value_config = st.column_config.NumberColumn(
                value_col,
                help=f"Valeur réalisée - {value_col}",
                format="%.0f"
            )
            ecart_format = "%.0f"
        
        st.dataframe(
            display_data,
            use_container_width=True,
            column_config={
                value_col: value_config,
                'Écart à la Moyenne': st.column_config.NumberColumn(
                    'Écart à la Moyenne',
                    help="Différence par rapport à la moyenne générale",
                    format=ecart_format
                ),
                '% vs Moyenne': st.column_config.NumberColumn(
                    '% vs Moyenne',
                    help="Pourcentage par rapport à la moyenne",
                    format="%.1f%%"
                )
            }
        )

def load_landing_data(file_path=None, uploaded_file=None):
    """Charge les données d'atterrissage depuis un fichier ou upload"""
    try:
        if uploaded_file is not None:
            # Fichier uploadé via Streamlit
            df = pd.read_excel(uploaded_file)
        else:
            # Fichier spécifique fourni via chemin
            df = pd.read_excel(file_path)
        
        # Nettoyage des données
        df.columns = df.columns.str.strip()
        
        # Renommer la colonne Régions pour uniformité
        if 'Régions' in df.columns:
            df = df.rename(columns={'Régions': 'REGION'})
        
        return df, None
    except FileNotFoundError as e:
        if uploaded_file is not None:
            return None, "❌ Erreur lors de la lecture du fichier uploadé"
        else:
            return None, f"❌ Fichier '{file_path}' non trouvé"
    except Exception as e:
        return None, f"❌ Erreur lors du chargement: {str(e)}"

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

def load_monthly_production_data(file_path=None, uploaded_file=None):
    """Charge les données de production mensuelle depuis un fichier ou upload"""
    try:
        if uploaded_file is not None:
            # Fichier uploadé via Streamlit
            df = pd.read_excel(uploaded_file)
        else:
            # Fichier spécifique fourni via chemin
            df = pd.read_excel(file_path)
        
        # Nettoyage des données
        df.columns = df.columns.str.strip()
        
        # Convertir les colonnes de production en numériques
        production_cols = [col for col in df.columns if 'PRODUCTION HORS PAE' in col]
        for col in production_cols:
            # Remplacer les tirets par 0
            df[col] = df[col].replace('-', '0')
            # Convertir en numérique
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convertir TOTAL en numérique
        if 'TOTAL' in df.columns:
            df['TOTAL'] = df['TOTAL'].replace('-', '0')
            df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')
        
        return df, None
    except FileNotFoundError as e:
        if uploaded_file is not None:
            return None, "❌ Erreur lors de la lecture du fichier uploadé"
        else:
            return None, f"❌ Fichier '{file_path}' non trouvé"
    except Exception as e:
        return None, f"❌ Erreur lors du chargement: {str(e)}"

def load_entrees_sorties_data(file_path=None, uploaded_file=None, sheet_name="Feuil1"):
    """Charge les données d'entrées-sorties-abandons depuis un fichier Excel uploadé ou un chemin"""
    try:
        if uploaded_file is not None:
            # Fichier uploadé via Streamlit
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            # Fichier spécifique fourni via chemin
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Nettoyage des données
        df.columns = df.columns.str.strip()
        
        # Renommer les colonnes pour uniformité
        column_mapping = {
            'Région': 'REGION',
            'Financeurs': 'FINANCEURS',
            'Nb. de stagiaires entrés': 'Entrées',
            'Nb. de stagaires sortis': 'Sorties',  # Note: il y a une faute de frappe dans l'original
            'DONT ABANDONS': 'Abandons',
            'Ecart \n(entrées-sorties)': 'Ecart'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Reconstruire la structure hiérarchique région/financeur
        df_clean = []
        current_region = None
        
        for index, row in df.iterrows():
            # Identifier les régions principales (non NaN et non Total)
            if pd.notna(row['REGION']) and not str(row['REGION']).startswith('Total'):
                current_region = row['REGION']
            
            # Ajouter les lignes avec financeurs
            if pd.notna(row['FINANCEURS']) and current_region:
                df_clean.append({
                    'REGION': current_region,
                    'FINANCEURS': row['FINANCEURS'],
                    'Entrées': row['Entrées'] if pd.notna(row['Entrées']) else 0,
                    'Sorties': row['Sorties'] if pd.notna(row['Sorties']) else 0,
                    'Abandons': row['Abandons'] if pd.notna(row['Abandons']) else 0,
                    'Ecart': row['Ecart'] if pd.notna(row['Ecart']) else 0
                })
        
        df_final = pd.DataFrame(df_clean)
        
        # Convertir les colonnes numériques
        numeric_cols = ['Entrées', 'Sorties', 'Abandons', 'Ecart']
        for col in numeric_cols:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
        
        return df_final, None
    except FileNotFoundError as e:
        if uploaded_file is not None:
            return None, "❌ Erreur lors de la lecture du fichier uploadé"
        else:
            return None, f"❌ Fichier '{file_path}' non trouvé"
    except Exception as e:
        return None, f"❌ Erreur lors du chargement: {str(e)}"

def load_monthly_data(file_path=None, uploaded_file=None):
    """Charge les données mensuelles depuis la Feuil2 d'un fichier Excel"""
    try:
        # Lire sans header pour traiter la structure complexe
        if uploaded_file is not None:
            df_raw = pd.read_excel(uploaded_file, sheet_name="Feuil2", header=None)
        elif file_path is not None:
            df_raw = pd.read_excel(file_path, sheet_name="Feuil2", header=None)
        else:
            raise ValueError("Aucun fichier fourni (file_path ou uploaded_file requis)")
        
        # Identifier les mois et leurs positions
        month_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                       'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        
        # Trouver les positions des mois dans la première ligne
        first_row = df_raw.iloc[0].tolist()
        month_positions = []
        
        for i, val in enumerate(first_row):
            if pd.notna(val):
                val_str = str(val).strip()
                for month in month_names:
                    if month.lower() in val_str.lower():
                        month_positions.append((i, month))
                        break
        
        # Traiter les données (à partir de la ligne 2, ligne 1 contient les sous-headers)
        reconstructed_data = []
        
        for row_idx in range(2, len(df_raw)):
            row_data = df_raw.iloc[row_idx]
            region = row_data[0]
            
            if pd.isna(region) or str(region).strip() == '':
                continue
                
            region_data: dict = {'Region': str(region).strip()}
            
            # Pour chaque mois, extraire les 4 valeurs
            for pos, month in month_positions:
                # Accéder aux valeurs avec iloc pour éviter les problèmes de Series
                try:
                    entrees_val = row_data.iloc[pos] if pos < len(row_data) else 0
                    sorties_val = row_data.iloc[pos + 1] if pos + 1 < len(row_data) else 0
                    abandons_val = row_data.iloc[pos + 2] if pos + 2 < len(row_data) else 0
                    ecart_val = row_data.iloc[pos + 3] if pos + 3 < len(row_data) else 0
                    
                    entrees = entrees_val if pd.notna(entrees_val) else 0
                    sorties = sorties_val if pd.notna(sorties_val) else 0
                    abandons = abandons_val if pd.notna(abandons_val) else 0
                    ecart = ecart_val if pd.notna(ecart_val) else 0
                    
                except (IndexError, KeyError):
                    entrees = sorties = abandons = ecart = 0
                
                # Convertir en numérique de manière sécurisée
                try:
                    entrees = float(entrees) if entrees != 0 and not pd.isna(entrees) else 0.0
                    sorties = float(sorties) if sorties != 0 and not pd.isna(sorties) else 0.0
                    abandons = float(abandons) if abandons != 0 and not pd.isna(abandons) else 0.0
                    ecart = float(ecart) if ecart != 0 and not pd.isna(ecart) else 0.0
                except (ValueError, TypeError):
                    entrees = sorties = abandons = ecart = 0.0
                
                region_data[f'{month}_Entrées'] = float(entrees)
                region_data[f'{month}_Sorties'] = float(sorties)
                region_data[f'{month}_Abandons'] = float(abandons)
                region_data[f'{month}_Écart'] = float(ecart)
            
            reconstructed_data.append(region_data)
        
        return pd.DataFrame(reconstructed_data)
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données mensuelles : {str(e)}")
        return pd.DataFrame()

def load_feuil3_data(file_path=None, uploaded_file=None):
    """Charge les données de la Feuil3 (analyse par financeur et région)"""
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, sheet_name="Feuil3")
        elif file_path is not None:
            df = pd.read_excel(file_path, sheet_name="Feuil3")
        else:
            raise ValueError("Aucun fichier fourni (file_path ou uploaded_file requis)")
        
        # Nettoyage des colonnes
        df.columns = df.columns.str.strip()
        
        # Renommer les colonnes si nécessaire pour uniformité
        column_mapping = {
            'Région': 'REGION',
            'Régions': 'REGION',
            'Financeurs': 'FINANCEURS',
            'Financeur': 'FINANCEURS'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Identifier les colonnes numériques (toutes sauf REGION et FINANCEURS)
        metric_cols = [col for col in df.columns if col not in ['REGION', 'FINANCEURS']]
        
        # Reconstruire la structure hiérarchique région/financeur (comme pour Feuil1)
        df_clean = []
        current_region = None
        
        for index, row in df.iterrows():
            # Identifier les régions principales (non NaN et non Total)
            if pd.notna(row['REGION']) and not str(row['REGION']).startswith('Total'):
                # Vérifier si c'est une ligne de région (pas de financeur ou financeur vide)
                if pd.isna(row.get('FINANCEURS')) or str(row.get('FINANCEURS')).strip() == '':
                    current_region = row['REGION']
                    continue
                else:
                    # Si la ligne a à la fois REGION et FINANCEURS remplis, c'est une ligne de données
                    current_region = row['REGION']
            
            # Ajouter les lignes avec financeurs
            if pd.notna(row.get('FINANCEURS')) and current_region and str(row.get('FINANCEURS')).strip() != '':
                row_data = {
                    'REGION': current_region,
                    'FINANCEURS': str(row['FINANCEURS']).strip()
                }
                
                # Ajouter toutes les colonnes métriques
                for col in metric_cols:
                    if col in row.index:
                        value = row[col]
                        row_data[col] = pd.to_numeric(value, errors='coerce') if pd.notna(value) else 0
                    else:
                        row_data[col] = 0
                
                df_clean.append(row_data)
        
        if not df_clean:
            # Si aucune donnée n'a été extraite avec la structure hiérarchique,
            # essayer de charger directement (structure plate)
            df = df[df['REGION'].notna()].copy()
            df = df[df['FINANCEURS'].notna()].copy()
            df = df[~df['REGION'].str.contains('Total', case=False, na=False)]
            
            # Convertir les colonnes numériques
            for col in metric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df, None
        
        df_final = pd.DataFrame(df_clean)
        
        return df_final, None
        
    except Exception as e:
        return None, f"❌ Erreur lors du chargement de la Feuil3: {str(e)}"

def create_feuil3_visualization(df):
    """Crée la visualisation pour la Feuil3 : Régions en abscisse, barres par financeur"""
    
    st.markdown("### 📊 Analyse par Région et Financeur (Feuil3)")
    
    if df is None or df.empty:
        st.warning("⚠️ Aucune donnée disponible dans la Feuil3")
        return
    
    # Vérifier les colonnes requises
    if 'REGION' not in df.columns or 'FINANCEURS' not in df.columns:
        st.error("❌ Colonnes 'REGION' et 'FINANCEURS' requises dans la Feuil3")
        return
    
    # Identifier les colonnes de métriques (toutes sauf REGION et FINANCEURS)
    metric_cols = [col for col in df.columns if col not in ['REGION', 'FINANCEURS']]
    
    if not metric_cols:
        st.error("❌ Aucune colonne de métrique trouvée dans la Feuil3")
        with st.expander("🔍 Informations de Debug"):
            st.write("**Colonnes détectées:**", df.columns.tolist())
            st.write("**Premières lignes:**")
            st.dataframe(df.head())
        return
    
    # Afficher les informations de debug
    with st.expander("🔍 Informations de Debug - Structure des Données"):
        st.write(f"**Nombre de lignes:** {len(df)}")
        st.write(f"**Nombre de régions uniques:** {df['REGION'].nunique()}")
        st.write(f"**Nombre de financeurs uniques:** {df['FINANCEURS'].nunique()}")
        st.write(f"**Colonnes métriques disponibles:** {metric_cols}")
        st.write("**Aperçu des données:**")
        st.dataframe(df.head(10))
    
    # Exclure "Total général" de la liste des régions
    all_regions = [r for r in df['REGION'].unique().tolist() if 'Total' not in str(r)]
    all_financeurs = df['FINANCEURS'].unique().tolist()
    
    # Initialiser les valeurs par défaut dans session_state si elles n'existent pas
    # OU si les financeurs disponibles ont changé (pour forcer la mise à jour)
    if 'feuil3_regions' not in st.session_state or 'feuil3_available_regions' not in st.session_state:
        st.session_state.feuil3_regions = all_regions
        st.session_state.feuil3_available_regions = all_regions
    elif set(st.session_state.feuil3_available_regions) != set(all_regions):
        # Les régions disponibles ont changé, réinitialiser
        st.session_state.feuil3_regions = all_regions
        st.session_state.feuil3_available_regions = all_regions
    
    if 'feuil3_financeurs' not in st.session_state or 'feuil3_available_financeurs' not in st.session_state:
        st.session_state.feuil3_financeurs = all_financeurs
        st.session_state.feuil3_available_financeurs = all_financeurs
    elif set(st.session_state.feuil3_available_financeurs) != set(all_financeurs):
        # Les financeurs disponibles ont changé, réinitialiser pour afficher TOUS les financeurs
        st.session_state.feuil3_financeurs = all_financeurs
        st.session_state.feuil3_available_financeurs = all_financeurs
    
    # Boutons de contrôle rapide - AVANT les widgets
    st.markdown("### 🎛️ Contrôles Rapides")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("🚀 Top 10 Régions", help="Afficher les 10 meilleures régions", key="top10_regions_f3"):
            # Choisir la première métrique disponible pour le calcul
            metric_for_calc = metric_cols[0]
            region_totals = df.groupby('REGION')[metric_for_calc].sum()
            top_regions = region_totals.nlargest(10).index.tolist()
            st.session_state.feuil3_regions = [r for r in top_regions if 'Total' not in str(r)]
            if 'feuil3_regions_select' in st.session_state:
                del st.session_state.feuil3_regions_select
            st.rerun()
    
    with col_btn2:
        if st.button("✅ Toutes les Régions", help="Sélectionner toutes les régions", key="all_regions_f3"):
            st.session_state.feuil3_regions = all_regions
            if 'feuil3_regions_select' in st.session_state:
                del st.session_state.feuil3_regions_select
            st.rerun()
    
    with col_btn3:
        if st.button("✅ Tous les Financeurs", help="Sélectionner tous les financeurs", key="all_financeurs_f3"):
            st.session_state.feuil3_financeurs = all_financeurs
            if 'feuil3_financeurs_select' in st.session_state:
                del st.session_state.feuil3_financeurs_select
            st.rerun()
    
    with col_btn4:
        if st.button("🔄 Réinitialiser", help="Réinitialiser tous les filtres", key="reset_filters_f3"):
            st.session_state.feuil3_regions = all_regions
            st.session_state.feuil3_financeurs = all_financeurs
            if 'feuil3_regions_select' in st.session_state:
                del st.session_state.feuil3_regions_select
            if 'feuil3_financeurs_select' in st.session_state:
                del st.session_state.feuil3_financeurs_select
            st.rerun()
    
    # Options de configuration - APRÈS les boutons
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            all_regions,
            default=all_regions if 'feuil3_regions' not in st.session_state else st.session_state.feuil3_regions,
            key="feuil3_regions_select"
        )
    
    with col_config2:
        financeurs_to_show = st.multiselect(
            "💰 Financeurs à afficher:",
            all_financeurs,
            default=all_financeurs if 'feuil3_financeurs' not in st.session_state else st.session_state.feuil3_financeurs,
            key="feuil3_financeurs_select"
        )
    
    with col_config3:
        metric_choice = st.selectbox(
            "📊 Métrique à analyser:",
            metric_cols,
            index=0,
            key="feuil3_metric"
        )
    
    # Détecter automatiquement si des colonnes année précédente et année en cours existent
    cols_previous = [col for col in metric_cols if str(previous_year) in str(col) and 'Ecart' not in str(col)]
    cols_current = [col for col in metric_cols if str(current_year) in str(col) and 'Ecart' not in str(col)]
    cols_ecart = [col for col in metric_cols if 'Ecart' in str(col) or 'écart' in str(col).lower()]
    
    can_compare_years = len(cols_previous) > 0 and len(cols_current) > 0
    
    # Option de comparaison d'années
    st.markdown("### 🔄 Options de Comparaison")
    col_comp1, col_comp2 = st.columns(2)
    
    comparison_metric = None  # Initialiser la variable
    
    with col_comp1:
        if can_compare_years:
            comparison_mode = st.selectbox(
                "📊 Mode d'analyse:",
                ["Vue simple", f"Comparaison {previous_year} vs {current_year}"],
                key="feuil3_comparison_mode"
            )
        else:
            comparison_mode = "Vue simple"
            st.info(f"ℹ️ Mode comparaison indisponible (colonnes {previous_year}/{current_year} manquantes)")
    
    with col_comp2:
        if comparison_mode == f"Comparaison {previous_year} vs {current_year}" and can_compare_years:
            # Déterminer les métriques de base (sans l'année)
            base_metrics = set()
            for col in cols_previous:
                # Enlever l'année précédente et nettoyer le nom
                base_name = col.replace(str(previous_year), '').replace(f'_{previous_year}', '').strip()
                # Enlever les caractères de ponctuation en début/fin
                base_name = base_name.strip(' -_.,')
                if base_name:
                    base_metrics.add(base_name)
            
            if base_metrics:
                comparison_metric = st.selectbox(
                    "📈 Métrique à comparer:",
                    sorted(list(base_metrics)),
                    key="feuil3_comparison_metric"
                )
            else:
                comparison_metric = None
                st.warning("⚠️ Impossible de déterminer les métriques à comparer")
        elif comparison_mode == "Vue simple":
            st.info(f"💡 Sélectionnez 'Comparaison {previous_year} vs {current_year}' pour comparer les années")
    
    if not regions_to_show or not financeurs_to_show:
        st.warning("⚠️ Veuillez sélectionner au moins une région et un financeur")
        return
    
    # Filtrer les données
    df_filtered = df[
        (df['REGION'].isin(regions_to_show)) & 
        (df['FINANCEURS'].isin(financeurs_to_show))
    ].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # ========== MODE COMPARAISON ANNUEL ==========
    if comparison_mode == f"Comparaison {previous_year} vs {current_year}" and can_compare_years and comparison_metric:
        st.markdown(f"### 📊 Comparaison {previous_year} vs {current_year}")
        
        # Construire les noms de colonnes pour l'année précédente et l'année en cours
        # Chercher les colonnes correspondantes à la métrique choisie
        possible_cols_previous = []
        possible_cols_current = []
        
        # Essayer plusieurs variantes de recherche
        search_patterns = [
            comparison_metric,  # Nom exact
            f"{comparison_metric} {previous_year}",  # Avec année précédente
            comparison_metric.replace('.', '').replace(',', ''),  # Sans ponctuation
        ]
        
        for pattern in search_patterns:
            if not possible_cols_previous:
                possible_cols_previous = [c for c in cols_previous if pattern.lower() in c.lower()]
            if not possible_cols_current:
                possible_cols_current = [c for c in cols_current if pattern.lower() in c.lower()]
            
            if possible_cols_previous and possible_cols_current:
                break
        
        # Si on n'a toujours pas trouvé, chercher juste avec une partie du nom
        if not possible_cols_previous or not possible_cols_current:
            # Prendre le premier mot significatif
            first_word = comparison_metric.split()[0] if ' ' in comparison_metric else comparison_metric
            possible_cols_previous = [c for c in cols_previous if first_word.lower() in c.lower()]
            possible_cols_current = [c for c in cols_current if first_word.lower() in c.lower()]
        
        # Vérifier qu'on a bien trouvé des colonnes
        if not possible_cols_previous or not possible_cols_current:
            st.error(f"❌ Impossible de trouver les colonnes pour '{comparison_metric}'")
            
            with st.expander("🔍 Informations de Debug - Colonnes Détectées"):
                st.write(f"**Métrique recherchée:** {comparison_metric}")
                st.write(f"**Colonnes {previous_year} disponibles:** {', '.join(cols_previous) if cols_previous else 'Aucune'}")
                st.write(f"**Colonnes {current_year} disponibles:** {', '.join(cols_current) if cols_current else 'Aucune'}")
                st.write(f"**Colonnes d'écart disponibles:** {', '.join(cols_ecart) if cols_ecart else 'Aucune'}")
                st.write(f"**Toutes les colonnes métriques:** {', '.join(metric_cols)}")
            
            return
        
        col_previous = possible_cols_previous[0]
        col_current = possible_cols_current[0]
        
        # Chercher la colonne d'écart si elle existe
        col_ecart = None
        if cols_ecart:
            ecart_matches = [c for c in cols_ecart if str(current_year) in c and str(previous_year) in c]
            if ecart_matches:
                col_ecart = ecart_matches[0]
        
        st.info(f"📊 Comparaison: **{col_previous}** vs **{col_current}**" + 
                (f" | Écart: **{col_ecart}**" if col_ecart else ""))
        
        # Calculer l'ordre des régions par total année en cours
        region_totals_current = df_filtered.groupby('REGION')[col_current].sum().sort_values(ascending=False)
        region_order = region_totals_current.index.tolist()
        
        # Créer le graphique de comparaison
        fig_compare = go.Figure()
        
        # Barres pour l'année précédente (toutes régions)
        regions_previous = []
        values_previous = []
        for region in region_order:
            region_data = df_filtered[df_filtered['REGION'] == region]
            if not region_data.empty and col_previous in region_data.columns:
                regions_previous.append(region)
                values_previous.append(region_data[col_previous].sum())
        
        fig_compare.add_trace(go.Bar(
            x=regions_previous,
            y=values_previous,
            name=str(previous_year),
            marker_color='#FF6B6B',
            text=[f"{val:,.0f}" for val in values_previous],
            textposition='outside',
            offsetgroup=0
        ))
        
        # Barres pour l'année en cours (toutes régions)
        regions_current = []
        values_current = []
        for region in region_order:
            region_data = df_filtered[df_filtered['REGION'] == region]
            if not region_data.empty and col_current in region_data.columns:
                regions_current.append(region)
                values_current.append(region_data[col_current].sum())
        
        fig_compare.add_trace(go.Bar(
            x=regions_current,
            y=values_current,
            name=str(current_year),
            marker_color='#4ECDC4',
            text=[f"{val:,.0f}" for val in values_current],
            textposition='outside',
            offsetgroup=1
        ))
        
        # Configuration du graphique
        fig_compare.update_layout(
            title=f"📊 Comparaison {comparison_metric} : {previous_year} vs {current_year}",
            xaxis_title="Régions",
            yaxis_title=comparison_metric,
            height=700,
            barmode='group',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # KPI de Comparaison
        st.markdown(f"### 📈 KPI de Comparaison {previous_year} vs {current_year}")
        
        total_previous = df_filtered[col_previous].sum() if col_previous in df_filtered.columns else 0
        total_current = df_filtered[col_current].sum() if col_current in df_filtered.columns else 0
        evolution = total_current - total_previous
        evolution_pct = (evolution / total_previous * 100) if total_previous > 0 else 0
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric(
                f"📊 Total {previous_year}",
                f"{total_previous:,.0f}",
                help=f"Total {comparison_metric} en {previous_year}"
            )
        
        with kpi_col2:
            st.metric(
                f"📊 Total {current_year}",
                f"{total_current:,.0f}",
                delta=f"{evolution:+,.0f}",
                help=f"Total {comparison_metric} en {current_year}"
            )
        
        with kpi_col3:
            delta_color = "normal" if evolution >= 0 else "inverse"
            st.metric(
                "📈 Évolution",
                f"{evolution_pct:+.1f}%",
                delta=f"{evolution:+,.0f}",
                delta_color=delta_color,
                help="Évolution en pourcentage et valeur absolue"
            )
        
        with kpi_col4:
            avg_current = total_current / len(regions_to_show) if len(regions_to_show) > 0 else 0
            avg_previous = total_previous / len(regions_to_show) if len(regions_to_show) > 0 else 0
            avg_evolution_pct = ((avg_current - avg_previous) / avg_previous * 100) if avg_previous > 0 else 0
            st.metric(
                f"📊 Moy. par Région {current_year}",
                f"{avg_current:,.0f}",
                delta=f"{avg_evolution_pct:+.1f}%",
                help=f"Moyenne par région en {current_year}"
            )
        
        # Analyse par région
        st.markdown("### 🏆 Analyse Détaillée par Région")
        
        comparison_data = []
        for region in region_order:
            region_data = df_filtered[df_filtered['REGION'] == region]
            if not region_data.empty:
                val_previous = region_data[col_previous].sum() if col_previous in region_data.columns else 0
                val_current = region_data[col_current].sum() if col_current in region_data.columns else 0
                
                # Utiliser la colonne d'écart si elle existe, sinon calculer
                if col_ecart and col_ecart in region_data.columns:
                    diff = region_data[col_ecart].sum()
                else:
                    diff = val_current - val_previous
                
                diff_pct = (diff / val_previous * 100) if val_previous > 0 else 0
                
                performance = "🟢 Croissance" if diff > 0 else "🔴 Décroissance" if diff < 0 else "🟡 Stable"
                
                row_data = {
                    'Région': region,
                    str(previous_year): val_previous,
                    str(current_year): val_current,
                    'Différence': diff,
                    'Évolution (%)': diff_pct,
                    'Performance': performance
                }
                
                # Ajouter la colonne d'écart si elle existe
                if col_ecart and col_ecart in region_data.columns:
                    row_data['Écart (fichier)'] = region_data[col_ecart].sum()
                
                comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Configuration des colonnes
        column_config = {
            'Région': st.column_config.TextColumn('Région', width='medium'),
            str(previous_year): st.column_config.NumberColumn(str(previous_year), format="%.0f"),
            str(current_year): st.column_config.NumberColumn(str(current_year), format="%.0f"),
            'Différence': st.column_config.NumberColumn('Différence', format="%+.0f"),
            'Évolution (%)': st.column_config.NumberColumn('Évolution (%)', format="%.1f%%"),
            'Performance': st.column_config.TextColumn('Performance')
        }
        
        if 'Écart (fichier)' in comparison_df.columns:
            column_config['Écart (fichier)'] = st.column_config.NumberColumn(
                'Écart (fichier)',
                format="%+.0f",
                help="Écart tel que défini dans le fichier source"
            )
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        # Statistiques de croissance
        st.markdown("### 📊 Statistiques de Performance")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        regions_croissance = len([r for r in comparison_data if r['Différence'] > 0])
        regions_decroissance = len([r for r in comparison_data if r['Différence'] < 0])
        regions_stable = len([r for r in comparison_data if r['Différence'] == 0])
        
        with stat_col1:
            st.metric(
                "🟢 Régions en Croissance",
                regions_croissance,
                f"{(regions_croissance/len(comparison_data)*100):.0f}%"
            )
        
        with stat_col2:
            st.metric(
                "🔴 Régions en Décroissance",
                regions_decroissance,
                f"{(regions_decroissance/len(comparison_data)*100):.0f}%"
            )
        
        # Meilleure progression
        best_progress = max(comparison_data, key=lambda x: x['Évolution (%)']) if comparison_data else None
        if best_progress:
            with stat_col3:
                st.metric(
                    "🚀 Meilleure Progression",
                    best_progress['Région'][:15],
                    f"+{best_progress['Évolution (%)']:.1f}%"
                )
        
        # Plus forte baisse
        worst_progress = min(comparison_data, key=lambda x: x['Évolution (%)']) if comparison_data else None
        if worst_progress and worst_progress['Évolution (%)'] < 0:
            with stat_col4:
                st.metric(
                    "⚠️ Plus Forte Baisse",
                    worst_progress['Région'][:15],
                    f"{worst_progress['Évolution (%)']:.1f}%"
                )
        
        return  # Sortir de la fonction après le mode comparaison
    
    # ========== MODE VUE SIMPLE (Original) ==========
    st.markdown("### 📊 Analyse Simple")
    
    # Créer le graphique avec barres groupées par financeur
    fig = go.Figure()
    
    # Palette de couleurs
    colors = px.colors.qualitative.Set3
    
    # Calculer l'ordre des régions (par total décroissant)
    region_totals = df_filtered.groupby('REGION')[metric_choice].sum().sort_values(ascending=False)
    region_order = region_totals.index.tolist()
    
    # Créer une barre pour chaque financeur
    for i, financeur in enumerate(financeurs_to_show):
        df_financeur = df_filtered[df_filtered['FINANCEURS'] == financeur]
        
        # Réorganiser selon l'ordre des régions
        regions_ordered = []
        values_ordered = []
        
        for region in region_order:
            region_data = df_financeur[df_financeur['REGION'] == region]
            if not region_data.empty:
                regions_ordered.append(region)
                values_ordered.append(region_data[metric_choice].iloc[0])
        
        if regions_ordered:
            fig.add_trace(go.Bar(
                x=regions_ordered,
                y=values_ordered,
                name=financeur,
                marker_color=colors[i % len(colors)],
                text=[f"{val:,.0f}" for val in values_ordered],
                textposition='outside',
                hovertemplate=f'<b>{financeur}</b><br>%{{x}}: %{{y:,.0f}}<extra></extra>'
            ))
    
    # Configuration du graphique
    fig.update_layout(
        title=f"📊 {metric_choice} par Région et Financeur",
        xaxis_title="Régions",
        yaxis_title=metric_choice,
        height=700,
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Section Totaux
    st.markdown("---")
    st.markdown(f"### 📊 Totaux - {metric_choice}")
    
    totals_data = []
    
    for region in regions_to_show:
        region_data = df_filtered[df_filtered['REGION'] == region]
        if not region_data.empty:
            total_region = region_data[metric_choice].sum()
            
            # Détails par financeur
            financeur_details = {}
            for financeur in financeurs_to_show:
                fin_data = region_data[region_data['FINANCEURS'] == financeur]
                if not fin_data.empty:
                    financeur_details[financeur] = fin_data[metric_choice].iloc[0]
                else:
                    financeur_details[financeur] = 0
            
            totals_data.append({
                'Région': region,
                'Total': total_region,
                **financeur_details
            })
    
    # Ajouter le Total Général
    total_general = df_filtered[metric_choice].sum()
    general_details = {'Région': '🏆 Total Général', 'Total': total_general}
    
    for financeur in financeurs_to_show:
        fin_total = df_filtered[df_filtered['FINANCEURS'] == financeur][metric_choice].sum()
        general_details[financeur] = fin_total
    
    totals_data.append(general_details)
    
    # DataFrame des totaux
    totals_df = pd.DataFrame(totals_data)
    
    # Trier par Total décroissant (Total Général en bas)
    total_general_row = totals_df[totals_df['Région'] == '🏆 Total Général']
    other_rows = totals_df[totals_df['Région'] != '🏆 Total Général'].sort_values('Total', ascending=False)
    totals_df = pd.concat([other_rows, total_general_row], ignore_index=True)
    
    # Configuration des colonnes
    column_config = {
        'Région': st.column_config.TextColumn('Région', width='medium'),
        'Total': st.column_config.NumberColumn(
            f'Total {metric_choice}',
            format="%.0f",
            help="Total tous financeurs"
        )
    }
    
    for financeur in financeurs_to_show:
        column_config[financeur] = st.column_config.NumberColumn(
            financeur,
            format="%.0f"
        )
    
    st.dataframe(
        totals_df,
        use_container_width=True,
        column_config=column_config,
        hide_index=True,
        height=min(600, (len(totals_df) + 1) * 35 + 38)
    )
    
    # Statistiques
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric(
            "🏆 Total Général",
            f"{total_general:,.0f}",
            help=f"Somme totale de {metric_choice}"
        )
    
    # Meilleure région
    best_region_data = totals_df[totals_df['Région'] != '🏆 Total Général'].iloc[0] if len(totals_df) > 1 else None
    if best_region_data is not None:
        with col_stat2:
            st.metric(
                "🥇 Meilleure Région",
                best_region_data['Région'][:20],
                f"{best_region_data['Total']:,.0f}"
            )
    
    # Meilleur financeur
    best_financeur = None
    best_financeur_value = 0
    for financeur in financeurs_to_show:
        financeur_total = df_filtered[df_filtered['FINANCEURS'] == financeur][metric_choice].sum()
        if financeur_total > best_financeur_value:
            best_financeur_value = financeur_total
            best_financeur = financeur
    
    with col_stat3:
        if best_financeur:
            st.metric(
                "🥇 Meilleur Financeur",
                best_financeur[:20],
                f"{best_financeur_value:,.0f}"
            )
    
    # Moyenne par région
    avg_by_region = totals_df[totals_df['Région'] != '🏆 Total Général']['Total'].mean()
    with col_stat4:
        st.metric(
            "📊 Moyenne par Région",
            f"{avg_by_region:,.0f}",
            help="Moyenne sur toutes les régions sélectionnées"
        )

def create_cumulative_curve(df, production_cols):
    """Crée une visualisation de production par région avec cumul"""
    
    st.markdown("### 📊 Production par Région avec Cumul")
    
    # Filtrer les données pour enlever les totaux
    df_filtered = df[~df['REGION'].str.contains('total|Total|TOTAL', case=False, na=False)].copy()
    df_filtered = df_filtered.dropna(subset=['REGION'])
    
    # Calculer la production totale par région
    region_totals = []
    region_names = []
    
    # Regrouper par région et sommer la production
    for region in df_filtered['REGION'].unique():
        region_data = df_filtered[df_filtered['REGION'] == region]
        
        # Calculer le total de production pour cette région (somme de tous les mois)
        total_production = 0
        for col in production_cols:
            if col in region_data.columns:
                total_production += region_data[col].sum()
        
        region_totals.append(total_production)
        region_names.append(region)
    
    # Trier par ordre décroissant de production
    region_data_sorted = sorted(zip(region_names, region_totals), key=lambda x: x[1], reverse=True)
    region_names = [x[0] for x in region_data_sorted]
    region_totals = [x[1] for x in region_data_sorted]
    
    # Calculer le cumul
    cumulative_totals = []
    cumul = 0
    for total in region_totals:
        cumul += total
        cumulative_totals.append(cumul)
    
    # Créer le graphique de courbe
    fig_cumul = go.Figure()
    
    # Ajouter la courbe de cumul
    fig_cumul.add_trace(go.Scatter(
        x=region_names,
        y=cumulative_totals,
        mode='lines+markers+text',
        name='Cumul Production',
        line=dict(color='#ff6b6b', width=4),
        marker=dict(size=10, color='#ff6b6b'),
        text=[f"{val:,.0f}" for val in cumulative_totals],
        textposition='top center'
    ))
    
    # Ajouter les barres par région
    fig_cumul.add_trace(go.Bar(
        x=region_names,
        y=region_totals,
        name='Production par Région',
        marker_color='rgba(52, 152, 219, 0.7)',
        yaxis='y2'
    ))
    
    # Configuration avec double axe Y
    fig_cumul.update_layout(
        title="📊 Production HORS PAE par Région avec Cumul",
        xaxis_title="Régions",
        yaxis=dict(
            title=dict(text="Cumul Production", font=dict(color="#ff6b6b")),
            tickfont=dict(color="#ff6b6b"),
            tickformat=",.0f"
        ),
        yaxis2=dict(
            title=dict(text="Production par Région", font=dict(color="#3498db")),
            tickfont=dict(color="#3498db"),
            anchor="x",
            overlaying="y",
            side="right",
            tickformat=",.0f"
        ),
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_tickangle=-45  # Rotation des noms de régions
    )
    
    st.plotly_chart(fig_cumul, use_container_width=True)
    
    # NOTE: Cette fonction a été remplacée - voir ligne suivante
    
def create_cumulative_curve_new(df, production_cols):
    """Crée une visualisation de production par région avec barres par financeur"""
    
    st.markdown("### 📊 Production par Région et Financeur")
    
    # Reconstruire la hiérarchie région/financeur
    def reconstruct_data(df):
        df_reconstructed = []
        current_region = None
        
        for index, row in df.iterrows():
            if pd.notna(row['REGION']) and 'Total' not in str(row['REGION']):
                current_region = row['REGION']
            
            if pd.notna(row['FINANCEURS']) and current_region and row['FINANCEURS'] != 'Pas de financeur':
                # Calculer la production totale pour ce financeur/région
                total_production = 0
                for col in production_cols:
                    if col in df.columns and pd.notna(row[col]):
                        total_production += row[col]
                
                df_reconstructed.append({
                    'REGION': current_region,
                    'FINANCEURS': row['FINANCEURS'],
                    'TOTAL': total_production if total_production > 0 else (row['TOTAL '] if 'TOTAL ' in row and pd.notna(row['TOTAL ']) else 0)
                })
        
        return pd.DataFrame(df_reconstructed)
    
    # Reconstruire les données
    df_viz = reconstruct_data(df)
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée de financeur disponible")
        return
    
    # Options de configuration
    col_config1, col_config2, col_config3, col_config4 = st.columns(4)
    
    with col_config1:
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            df_viz['REGION'].unique().tolist(),
            default=df_viz['REGION'].unique().tolist(),  # Afficher toutes les régions par défaut
            key="cumul_regions"
        )
    
    with col_config2:
        financeurs_to_show = st.multiselect(
            "💰 Financeurs à afficher:",
            df_viz['FINANCEURS'].unique().tolist(),
            default=df_viz['FINANCEURS'].unique().tolist(),
            key="cumul_financeurs"
        )
    
    with col_config3:
        sort_order = st.selectbox(
            "📊 Ordre d'affichage:",
            ["Décroissant (Plus grand → Plus petit)", "Croissant (Plus petit → Plus grand)", "Alphabétique (A → Z)"],
            index=0,  # Décroissant par défaut
            key="region_sort_order"
        )
    
    with col_config4:
        financeurs_with_labels = st.multiselect(
            "🔢 Afficher % pour:",
            df_viz['FINANCEURS'].unique().tolist(),
            default=df_viz['FINANCEURS'].unique().tolist(),  # Tous les financeurs par défaut
            key="financeurs_labels",
            help="Sélectionnez les financeurs pour lesquels afficher les pourcentages sur les barres"
        )
    
    # Boutons de contrôle rapide des filtres
    st.markdown("### 🎛️ Contrôles Rapides")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("✅ Toutes les Régions", help="Sélectionner toutes les régions"):
            st.session_state.cumul_regions = df_viz['REGION'].unique().tolist()
            st.rerun()
    
    with col_btn2:
        if st.button("❌ Aucune Région", help="Désélectionner toutes les régions"):
            st.session_state.cumul_regions = []
            st.rerun()
    
    with col_btn3:
        if st.button("✅ Tous les Financeurs", help="Sélectionner tous les financeurs"):
            st.session_state.cumul_financeurs = df_viz['FINANCEURS'].unique().tolist()
            st.rerun()
    
    with col_btn4:
        if st.button("❌ Aucun Financeur", help="Désélectionner tous les financeurs"):
            st.session_state.cumul_financeurs = []
            st.rerun()
    
    # Filtrer selon les sélections
    df_filtered = df_viz[
        (df_viz['REGION'].isin(regions_to_show)) & 
        (df_viz['FINANCEURS'].isin(financeurs_to_show))
    ].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return

    # Fonction pour extraire les totaux par région avec tri
    def get_region_totals_from_data(df, sort_order):
        """Extrait les totaux de chaque région depuis les lignes 'Total [Region]'"""
        total_lines = df[df['REGION'].str.contains('Total', na=False) & ~df['REGION'].str.contains('Total général', na=False)]
        
        region_totals_dict = {}
        for index, row in total_lines.iterrows():
            region_name = row['REGION'].replace('Total ', '')
            total_value = row['TOTAL'] if pd.notna(row['TOTAL']) else 0
            region_totals_dict[region_name] = total_value
        
        # Filtrer seulement les régions sélectionnées par l'utilisateur
        filtered_totals = {region: total for region, total in region_totals_dict.items() 
                          if region in regions_to_show}
        
        # Appliquer le tri selon l'option choisie
        if sort_order == "Alphabétique (A → Z)":
            return sorted(filtered_totals.items(), key=lambda x: x[0])  # Tri alphabétique par nom de région
        elif sort_order == "Croissant (Plus petit → Plus grand)":
            return sorted(filtered_totals.items(), key=lambda x: x[1])  # Tri croissant par valeur
        else:  # "Décroissant (Plus grand → Plus petit)"
            return sorted(filtered_totals.items(), key=lambda x: x[1], reverse=True)  # Tri décroissant par valeur

    # Obtenir les totaux réels par région avec l'ordre choisi
    region_totals_sorted = get_region_totals_from_data(df, sort_order)
    region_names_ordered = [item[0] for item in region_totals_sorted]
    region_totals_values = [item[1] for item in region_totals_sorted]
    
    # Créer un dictionnaire pour un accès rapide aux totaux par région
    region_totals_dict = dict(region_totals_sorted)
    
    # Créer le graphique avec barres groupées par financeur et région
    fig = go.Figure()
    
    # Palette de couleurs pour les financeurs
    colors = px.colors.qualitative.Set3
    financeurs_unique = df_filtered['FINANCEURS'].unique()
    
    # Créer une barre pour chaque financeur en respectant l'ordre des régions de la courbe
    for i, financeur in enumerate(financeurs_unique):
        df_financeur = df_filtered[df_filtered['FINANCEURS'] == financeur]
        
        # Réorganiser les données selon l'ordre des régions de la courbe
        regions_ordered = []
        percentages_ordered = []
        
        for region in region_names_ordered:
            region_data = df_financeur[df_financeur['REGION'] == region]
            if not region_data.empty:
                regions_ordered.append(region)
                # Calculer le pourcentage par rapport au total de la région
                financeur_value = region_data['TOTAL'].iloc[0]
                region_total = region_totals_dict.get(region, 1)  # Éviter division par zéro
                percentage = (financeur_value / region_total * 100) if region_total > 0 else 0
                percentages_ordered.append(percentage)
        
        # Déterminer si on affiche les pourcentages pour ce financeur
        show_text = financeur in financeurs_with_labels
        
        if regions_ordered:  # Seulement si on a des données pour ce financeur
            fig.add_trace(go.Bar(
                x=regions_ordered,  # Régions en X dans l'ordre choisi
                y=percentages_ordered,   # Pourcentage en Y
                name=financeur,
                marker_color=colors[i % len(colors)],
                text=[f"<b>{val:.1f}%</b>" for val in percentages_ordered] if show_text else None,  # Afficher les % seulement si sélectionné
                textposition='outside' if show_text else None,
                textfont=dict(size=20) if show_text else None,  # Taille de police augmentée
                offsetgroup=i  # Groupes distincts pour chaque financeur
            ))
    
    # Ajouter la courbe avec les totaux individuels de chaque région (pas de cumul)
    # Afficher les valeurs totales réelles pour chaque région
    
    fig.add_trace(go.Scatter(
        x=region_names_ordered,
        y=region_totals_values,
        mode='lines+markers+text',
        name='Total par Région (100%)',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=10, color='#ff6b6b', symbol='circle'),
        text=[f"{val:,.0f}" for val in region_totals_values],
        textposition='top center',
        hovertemplate='<b>%{x}</b><br>Total: %{y:,.0f}<extra></extra>',
        yaxis='y2'  # Utiliser l'axe secondaire pour les valeurs absolues
    ))
    
    # Configuration du graphique avec double axe Y
    fig.update_layout(
        title="📊 Production HORS PAE par Région et Financeur (% de contribution par région)",
        xaxis_title="Régions",
        yaxis=dict(
            title="Part de Production (%)",
            tickformat=".0f",
            ticksuffix="%",
            range=[0, 120]  # Augmenté pour laisser plus d'espace aux labels plus grands
        ),
        yaxis2=dict(
            title="Total par Région (100%)",
            overlaying='y',
            side='right',
            tickformat=',.0f',
            showgrid=False
        ),
        height=700,
        barmode='group',  # Barres groupées par financeur
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_tickangle=-45,  # Rotation des noms de régions
        uniformtext=dict(mode='show', minsize=20)  # Force l'affichage du texte avec une taille minimale
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques par financeur et cumul
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    # Calculer les totaux par financeur
    financeur_totals = df_filtered.groupby('FINANCEURS')['TOTAL'].sum().sort_values(ascending=False)
    
    with col_stats1:
        st.metric(
            "📊 Total Général",
            f"{df_filtered['TOTAL'].sum():,.0f}",
            help="Total de production tous financeurs confondus"
        )
    
    with col_stats2:
        best_financeur = financeur_totals.index[0]
        st.metric(
            "🏆 Meilleur Financeur",
            best_financeur[:15] + "..." if len(best_financeur) > 15 else best_financeur,
            f"{financeur_totals.iloc[0]:,.0f}"
        )
    
    with col_stats3:
        avg_by_financeur = financeur_totals.mean()
        st.metric(
            "📊 Moyenne par Financeur",
            f"{avg_by_financeur:,.0f}",
            help="Production moyenne par financeur"
        )
    
    with col_stats4:
        best_region = region_names_ordered[0] if region_names_ordered else "N/A"
        best_region_total = region_totals_values[0] if region_totals_values else 0
        st.metric(
            "🎯 Meilleure Région (Cumul)",
            best_region[:15] + "..." if len(best_region) > 15 else best_region,
            f"{best_region_total:,.0f}",
            help="Région avec le plus gros total (tous financeurs)"
        )
    # NOTE: Cette fonction a été remplacée - voir ligne suivante

def create_entrees_sorties_visualization(df):
    """Crée la visualisation des entrées-sorties-abandons par région et financeur"""
    
    st.markdown("### 📊 Analyse Entrées-Sorties-Abandons par Région et Financeur")
    
    # Les données sont déjà agrégées depuis le fichier Excel
    df_agg = df.copy()
    
    # Options de configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            df_agg['REGION'].unique().tolist(),
            default=df_agg['REGION'].unique().tolist(),  # Toutes les régions par défaut
            key="entrees_sorties_regions"
        )
    
    with col_config2:
        financeurs_to_show = st.multiselect(
            "💰 Financeurs à afficher:",
            df_agg['FINANCEURS'].unique().tolist(),
            default=df_agg['FINANCEURS'].unique().tolist(),
            key="entrees_sorties_financeurs"
        )
    
    with col_config3:
        metric_choice = st.selectbox(
            "📊 Métrique à analyser:",
            ["Entrées", "Sorties", "Abandons", "Ecart", "Toutes les métriques"],
            index=0,
            key="entrees_sorties_metric"
        )
    
    # Filtrer selon les sélections
    df_filtered = df_agg[
        (df_agg['REGION'].isin(regions_to_show)) & 
        (df_agg['FINANCEURS'].isin(financeurs_to_show))
    ].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Boutons de contrôle rapide
    st.markdown("### 🎛️ Contrôles Rapides")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("🚀 Top 5 Régions", help="Afficher les 5 meilleures régions", key="top5_regions"):
            # Calculer les totaux par région et prendre le top 5
            region_totals = df_agg.groupby('REGION')[metric_choice if metric_choice != "Toutes les métriques" else "Entrées"].sum()
            top_regions = region_totals.nlargest(5).index.tolist()
            st.session_state.entrees_sorties_regions = top_regions
            st.rerun()
    
    with col_btn2:
        if st.button("✅ Toutes les Régions", help="Sélectionner toutes les régions", key="all_regions_es"):
            st.session_state.entrees_sorties_regions = df_agg['REGION'].unique().tolist()
            st.rerun()
    
    with col_btn3:
        if st.button("✅ Tous les Financeurs", help="Sélectionner tous les financeurs", key="all_financeurs_es"):
            st.session_state.entrees_sorties_financeurs = df_agg['FINANCEURS'].unique().tolist()
            st.rerun()
    
    with col_btn4:
        if st.button("🔄 Réinitialiser", help="Réinitialiser tous les filtres", key="reset_filters_es"):
            st.session_state.entrees_sorties_regions = df_agg['REGION'].unique().tolist()[:10]
            st.session_state.entrees_sorties_financeurs = df_agg['FINANCEURS'].unique().tolist()
            st.rerun()
    
    # Tri par ordre décroissant de la métrique choisie
    if metric_choice != "Toutes les métriques":
        # Calculer les totaux par région pour le tri
        region_totals = df_filtered.groupby('REGION')[metric_choice].sum().sort_values(ascending=False)
        region_order = region_totals.index.tolist()
    else:
        # Utiliser le total des Entrées pour le tri
        region_totals = df_filtered.groupby('REGION')['Entrées'].sum().sort_values(ascending=False)
        region_order = region_totals.index.tolist()
    
    if metric_choice == "Toutes les métriques":
        # Créer un graphique avec 4 sous-graphiques
        from plotly.subplots import make_subplots
        
        metrics = ['Entrées', 'Sorties', 'Abandons', 'Ecart']
        metric_titles = ['Entrées de Stagiaires', 'Sorties de Stagiaires', 'Abandons', 'Écart (Entrées-Sorties)']
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=metric_titles,
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics):
            for j, financeur in enumerate(df_filtered['FINANCEURS'].unique()):
                df_financeur = df_filtered[df_filtered['FINANCEURS'] == financeur]
                
                # Réorganiser selon l'ordre des régions
                regions_ordered = []
                values_ordered = []
                
                for region in region_order:
                    region_data = df_financeur[df_financeur['REGION'] == region]
                    if not region_data.empty:
                        regions_ordered.append(region)
                        values_ordered.append(region_data[metric].iloc[0])
                
                if regions_ordered:
                    fig.add_trace(
                        go.Bar(
                            x=regions_ordered,
                            y=values_ordered,
                            name=financeur,
                            marker_color=colors[j % len(colors)],
                            showlegend=(i == 0),  # Légende seulement sur le premier graphique
                            text=[f"{val:,.0f}" for val in values_ordered],
                            textposition='outside'
                        ),
                        row=i+1, col=1
                    )
        
        fig.update_layout(
            title="📊 Analyse Complète des Flux de Stagiaires par Région et Financeur",
            height=1400,
            barmode='group',
            hovermode='x unified'
        )
        
        # Mettre à jour les axes X
        for i in range(4):
            fig.update_xaxes(tickangle=-45, row=i+1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Graphique simple pour une métrique
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, financeur in enumerate(df_filtered['FINANCEURS'].unique()):
            df_financeur = df_filtered[df_filtered['FINANCEURS'] == financeur]
            
            # Réorganiser selon l'ordre des régions
            regions_ordered = []
            values_ordered = []
            
            for region in region_order:
                region_data = df_financeur[df_financeur['REGION'] == region]
                if not region_data.empty:
                    regions_ordered.append(region)
                    values_ordered.append(region_data[metric_choice].iloc[0])
            
            if regions_ordered:
                fig.add_trace(go.Bar(
                    x=regions_ordered,
                    y=values_ordered,
                    name=financeur,
                    marker_color=colors[i % len(colors)],
                    text=[f"{val:,.0f}" for val in values_ordered],
                    textposition='outside'
                ))
        
        # Titre en fonction de la métrique
        metric_names = {
            "Entrées": "Entrées de Stagiaires",
            "Sorties": "Sorties de Stagiaires", 
            "Abandons": "Abandons",
            "Ecart": "Écart (Entrées-Sorties)"
        }
        
        fig.update_layout(
            title=f"📊 {metric_names[metric_choice]} par Région et Financeur",
            xaxis_title="Régions",
            yaxis_title=metric_names[metric_choice],
            height=700,
            barmode='group',
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
    create_entrees_sorties_statistics(df_filtered, metric_choice)

def create_entrees_sorties_statistics(df, metric_choice):
    """Crée les statistiques récapitulatives pour entrées-sorties-abandons"""
    
    st.markdown("### 📊 Statistiques Récapitulatives")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    if metric_choice == "Toutes les métriques":
        # Statistiques globales
        total_entrees = df['Entrées'].sum()
        total_sorties = df['Sorties'].sum()
        total_abandons = df['Abandons'].sum()
        total_ecart = df['Ecart'].sum()
        
        with col_stats1:
            st.metric(
                "📥 Total Entrées",
                f"{total_entrees:,.0f}",
                help="Total des entrées de stagiaires tous financeurs confondus"
            )
        
        with col_stats2:
            st.metric(
                "📤 Total Sorties",
                f"{total_sorties:,.0f}",
                help="Total des sorties de stagiaires tous financeurs confondus"
            )
        
        with col_stats3:
            st.metric(
                "❌ Total Abandons",
                f"{total_abandons:,.0f}",
                help="Total des abandons tous financeurs confondus"
            )
        
        with col_stats4:
            # Calcul du taux de rétention (entrées - abandons) / entrées
            if total_entrees > 0:
                retention_rate = ((total_entrees - total_abandons) / total_entrees) * 100
                st.metric(
                    "💪 Taux de Rétention",
                    f"{retention_rate:.1f}%",
                    help="(Entrées - Abandons) / Entrées * 100"
                )
            else:
                st.metric("💪 Taux de Rétention", "N/A")
        
        # Métriques supplémentaires en seconde ligne
        st.markdown("### 📈 Métriques Complémentaires")
        col_extra1, col_extra2, col_extra3, col_extra4 = st.columns(4)
        
        with col_extra1:
            st.metric(
                "🔄 Écart Total",
                f"{total_ecart:,.0f}",
                help="Écart total (Entrées - Sorties)"
            )
        
        with col_extra2:
            # Taux de sortie
            if total_entrees > 0:
                taux_sortie = (total_sorties / total_entrees) * 100
                st.metric(
                    "📊 Taux de Sortie",
                    f"{taux_sortie:.1f}%",
                    help="Sorties / Entrées * 100"
                )
            else:
                st.metric("📊 Taux de Sortie", "N/A")
        
        with col_extra3:
            # Taux d'abandon
            if total_entrees > 0:
                taux_abandon = (total_abandons / total_entrees) * 100
                st.metric(
                    "⚠️ Taux d'Abandon",
                    f"{taux_abandon:.1f}%",
                    help="Abandons / Entrées * 100"
                )
            else:
                st.metric("⚠️ Taux d'Abandon", "N/A")
        
        with col_extra4:
            # Efficacité (écart positif)
            if total_entrees > 0:
                efficacite = (total_ecart / total_entrees) * 100
                delta_color = "normal" if efficacite >= 0 else "inverse"
                st.metric(
                    "🎯 Efficacité",
                    f"{efficacite:.1f}%",
                    f"{'Positif' if efficacite >= 0 else 'Négatif'}",
                    delta_color=delta_color,
                    help="Écart / Entrées * 100"
                )
            else:
                st.metric("🎯 Efficacité", "N/A")
    
    else:
        # Statistiques pour une métrique spécifique
        total_metric = df[metric_choice].sum()
        avg_by_region = df.groupby('REGION')[metric_choice].sum().mean()
        
        # Meilleure région
        region_totals = df.groupby('REGION')[metric_choice].sum().sort_values(ascending=False)
        best_region = region_totals.index[0] if len(region_totals) > 0 else "N/A"
        best_value = region_totals.iloc[0] if len(region_totals) > 0 else 0
        
        # Meilleur financeur
        financeur_totals = df.groupby('FINANCEURS')[metric_choice].sum().sort_values(ascending=False)
        best_financeur = financeur_totals.index[0] if len(financeur_totals) > 0 else "N/A"
        
        with col_stats1:
            metric_display_names = {
                "Entrées": "Entrées Totales",
                "Sorties": "Sorties Totales",
                "Abandons": "Abandons Totaux", 
                "Ecart": "Écart Total"
            }
            st.metric(
                f"📊 {metric_display_names.get(metric_choice, metric_choice)}",
                f"{total_metric:,.0f}",
                help=f"Total tous financeurs et régions confondus"
            )
        
        with col_stats2:
            st.metric(
                "🏆 Meilleure Région",
                best_region[:15] + "..." if len(best_region) > 15 else best_region,
                f"{best_value:,.0f}"
            )
        
        with col_stats3:
            st.metric(
                "🥇 Meilleur Financeur",
                best_financeur[:15] + "..." if len(best_financeur) > 15 else best_financeur,
                f"{financeur_totals.iloc[0]:,.0f}" if len(financeur_totals) > 0 else "0"
            )
        
        with col_stats4:
            st.metric(
                "📊 Moyenne par Région",
                f"{avg_by_region:,.0f}",
                help="Valeur moyenne par région"
            )
    
    # Tableau détaillé par région et financeur
    with st.expander("📋 Données Détaillées par Région et Financeur"):
        # Préparer les données pour l'affichage
        display_data = df.copy()
        
        # Calculer des métriques additionnelles si toutes les métriques
        if metric_choice == "Toutes les métriques":
            display_data['Taux Abandon (%)'] = (display_data['Abandons'] / 
                                              (display_data['Entrées'] + 0.001) * 100).round(1)
            display_data['Taux Sortie (%)'] = (display_data['Sorties'] / 
                                             (display_data['Entrées'] + 0.001) * 100).round(1)
            display_data['Flux Net'] = display_data['Entrées'] - display_data['Sorties']
        
        # Trier par région puis par financeur
        display_data = display_data.sort_values(['REGION', 'FINANCEURS'])
        
        # Configuration des colonnes
        column_config = {
            'REGION': 'Région',
            'FINANCEURS': 'Financeur',
            'Entrées': st.column_config.NumberColumn(
                'Entrées',
                format="%.0f"
            ),
            'Sorties': st.column_config.NumberColumn(
                'Sorties',
                format="%.0f"
            ),
            'Abandons': st.column_config.NumberColumn(
                'Abandons',
                format="%.0f"
            ),
            'Ecart': st.column_config.NumberColumn(
                'Écart (E-S)',
                format="%.0f"
            )
        }
        
        if metric_choice == "Toutes les métriques":
            column_config.update({
                'Taux Abandon (%)': st.column_config.NumberColumn(
                    'Taux Abandon (%)',
                    format="%.1f%%"
                ),
                'Taux Sortie (%)': st.column_config.NumberColumn(
                    'Taux Sortie (%)',
                    format="%.1f%%"
                ),
                'Flux Net': st.column_config.NumberColumn(
                    'Flux Net (E-S)',
                    format="%.0f"
                )
            })
        
        st.dataframe(
            display_data,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )

def create_monthly_evolution_chart(df):
    """Crée un graphique d'évolution mensuelle avec barres par mois, axe X = Région, axe Y = Pourcentage"""
    
    st.markdown("### 📈 Évolution Mensuelle par Région")
    st.markdown("*Abscisse: Région, Ordonnée: Pourcentage, Barres par Mois*")
    
    # Identifier les colonnes de pourcentage mensuelles (TX DE CAPACITE)
    tx_cols = [col for col in df.columns if 'TX DE CAPACITE' in col and any(mois in col for mois in 
               ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN', 'JUILLET', 'AOÛT', 'SEPTEMBRE'])]
    
    if len(tx_cols) == 0:
        st.warning("⚠️ Aucune colonne de taux de capacité mensuelle trouvée")
        return
    
    # Nettoyer les noms des mois pour l'affichage
    month_mapping = {
        'JANVIER': 'Janvier', 'FEVRIER': 'Février', 'MARS': 'Mars', 'AVRIL': 'Avril',
        'MAI': 'Mai', 'JUIN': 'Juin', 'JUILLET': 'Juillet', 'AOÛT': 'Août', 'SEPTEMBRE': 'Septembre'
    }
    
    # Options de configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        chart_type = st.selectbox(
            "📊 Type de visualisation:",
            ["Barres groupées par mois", "Barres empilées", "Heatmap mensuelle"],
            key="monthly_chart_type"
        )
    
    with col_config2:
        # Boutons de gestion des filtres
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            if st.button("🚫 Enlever Tout", key="clear_all_regions", help="Désélectionner toutes les régions"):
                st.session_state.monthly_regions = []
                st.rerun()
        
        with filter_col2:
            if st.button("✅ Tout Sélectionner", key="select_all_regions", help="Sélectionner toutes les régions"):
                st.session_state.monthly_regions = df['REGION'].tolist()
                st.rerun()
        
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            df['REGION'].tolist(),
            default=df['REGION'].tolist() if 'monthly_regions' not in st.session_state else st.session_state.monthly_regions,
            key="monthly_regions"
        )
    
    with col_config3:
        show_average_line = st.checkbox(
            "📈 Afficher ligne de moyenne",
            value=True,
            key="show_avg_line"
        )
    
    # Filtrer les données selon les régions sélectionnées
    df_filtered = df[df['REGION'].isin(regions_to_show)].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune région sélectionnée")
        return
    
    # Préparation des données pour le graphique - NOUVEAU FORMAT
    if chart_type == "Barres groupées par mois":
        
        # Créer le graphique principal avec Région en X et Pourcentage en Y
        fig = go.Figure()
        
        # Palette de couleurs pour les mois
        colors = px.colors.qualitative.Set3
        
        # Créer les noms de mois complets
        month_names = []
        for col in tx_cols:
            for full_month, display_month in month_mapping.items():
                if full_month in col:
                    month_names.append(display_month)
                    break
        
        # Ajouter les barres pour chaque mois
        for i, month_col in enumerate(tx_cols):
            month_name = month_names[i] if i < len(month_names) else f"Mois {i+1}"
            values = [row[month_col] * 100 for _, row in df_filtered.iterrows()]  # Convertir en pourcentage
            regions = df_filtered['REGION'].tolist()
            
            fig.add_trace(go.Bar(
                x=regions,  # Régions en X
                y=values,   # Pourcentages en Y
                name=month_name,  # Nom du mois pour la légende
                marker_color=colors[i % len(colors)],
                offsetgroup=i,
                text=[f"{val:.1f}%" for val in values],
                textposition='outside'
            ))
        
        # Calculer et ajouter la ligne de moyenne si demandée
        if show_average_line:
            avg_values = []
            regions = df_filtered['REGION'].tolist()
            
            for _, row in df_filtered.iterrows():
                region_avg = np.mean([row[col] for col in tx_cols]) * 100
                avg_values.append(region_avg)
            
            fig.add_trace(go.Scatter(
                x=regions,
                y=avg_values,
                mode='lines+markers',
                name='🎯 Moyenne par Région',
                line=dict(color='red', width=4, dash='solid'),
                marker=dict(size=8, color='red', symbol='diamond')
            ))
        
        # Configuration du graphique
        fig.update_layout(
            title=f"📊 Évolution Mensuelle TX Capacité par Région - {len(regions_to_show)} Régions",
            xaxis_title="Régions",
            yaxis_title="Taux de Capacité (%)",
            height=700,
            barmode='group',  # Barres groupées par mois
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_tickangle=-45  # Rotation des noms de régions
        )
        
        # Personnaliser les axes
        fig.update_layout(yaxis=dict(ticksuffix="%", tickformat=".1f"))
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Barres empilées":
        
        # Préparer les données pour barres empilées - NOUVEAU FORMAT
        fig = go.Figure()
        
        # Créer les noms de mois complets
        month_names = []
        for col in tx_cols:
            for full_month, display_month in month_mapping.items():
                if full_month in col:
                    month_names.append(display_month)
                    break
        
        colors = px.colors.qualitative.Pastel
        regions = df_filtered['REGION'].tolist()
        
        # Ajouter une barre pour chaque mois
        for i, month_col in enumerate(tx_cols):
            month_name = month_names[i] if i < len(month_names) else f"Mois {i+1}"
            values = [row[month_col] * 100 for _, row in df_filtered.iterrows()]
            
            fig.add_trace(go.Bar(
                x=regions,  # Régions en X
                y=values,   # Pourcentages en Y
                name=month_name,  # Nom du mois
                marker_color=colors[i % len(colors)],
                text=[f"{val:.1f}%" for val in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="📊 Évolution Mensuelle - Vue Empilée par Région",
            xaxis_title="Régions",
            yaxis_title="Taux de Capacité (%)",
            height=700,
            barmode='stack',
            xaxis_tickangle=-45,
            yaxis=dict(ticksuffix="%", tickformat=".1f"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Heatmap mensuelle
        
        # Préparer les données pour la heatmap - Format: Mois en Y, Régions en X
        heatmap_data = []
        month_names = []
        
        for col in tx_cols:
            for full_month, display_month in month_mapping.items():
                if full_month in col:
                    month_names.append(display_month)
                    values = [row[col] * 100 for _, row in df_filtered.iterrows()]
                    heatmap_data.append(values)
                    break
        
        # Créer la heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=df_filtered['REGION'],  # Régions en X
            y=month_names,           # Mois en Y
            colorscale='Viridis',
            hoverongaps=False,
            text=[[f"{val:.1f}%" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Taux de Capacité (%)")
        ))
        
        fig.update_layout(
            title="🔥 Heatmap des Performances Mensuelles",
            xaxis_title="Régions",
            yaxis_title="Mois",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques récapitulatives
    st.markdown("### 📊 Statistiques Récapitulatives")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    # Calculer les statistiques
    all_monthly_values = []
    for col in tx_cols:
        all_monthly_values.extend(df_filtered[col].tolist())
    
    avg_global = np.mean(all_monthly_values) * 100
    max_global = np.max(all_monthly_values) * 100
    min_global = np.min(all_monthly_values) * 100
    
    # Meilleure région en moyenne
    region_averages = []
    for _, row in df_filtered.iterrows():
        region_avg = np.mean([row[col] for col in tx_cols]) * 100
        region_averages.append((row['REGION'], region_avg))
    
    best_region = max(region_averages, key=lambda x: x[1])
    worst_region = min(region_averages, key=lambda x: x[1])
    
    with col_stats1:
        st.metric(
            "🎯 Moyenne Générale",
            f"{avg_global:.1f}%",
            help="Moyenne de tous les taux sur toutes les régions et mois"
        )
    
    with col_stats2:
        st.metric(
            "🏆 Meilleure Région",
            best_region[0][:15] + "..." if len(best_region[0]) > 15 else best_region[0],
            f"{best_region[1]:.1f}%"
        )
    
    with col_stats3:
        st.metric(
            "📈 Performance Max",
            f"{max_global:.1f}%",
            f"+{max_global - avg_global:.1f}% vs moyenne"
        )
    
    with col_stats4:
        st.metric(
            "📉 Performance Min",
            f"{min_global:.1f}%",
            f"{min_global - avg_global:.1f}% vs moyenne"
        )
    
    # Classification des Régions pour l'évolution mensuelle
    st.markdown("### 🏆 Classification des Régions - Performance Mensuelle")
    
    # Calculer les moyennes par région pour la classification
    region_performances = []
    for _, row in df_filtered.iterrows():
        region = row['REGION']
        monthly_values = [row[col] * 100 for col in tx_cols]
        avg_performance = np.mean(monthly_values)
        region_performances.append({
            'region': region,
            'average': avg_performance,
            'max_month': np.max(monthly_values),
            'min_month': np.min(monthly_values),
            'consistency': np.std(monthly_values)
        })
    
    # Tri par performance moyenne décroissante
    region_performances.sort(key=lambda x: x['average'], reverse=True)
    
    # Calcul des quartiles pour la classification
    averages = [rp['average'] for rp in region_performances]
    q1 = np.percentile(averages, 25)
    q2 = np.percentile(averages, 50)  # Médiane
    q3 = np.percentile(averages, 75)
    
    # Classification des régions
    excellent_regions = [rp for rp in region_performances if rp['average'] >= q3]
    good_regions = [rp for rp in region_performances if q2 <= rp['average'] < q3]
    fair_regions = [rp for rp in region_performances if q1 <= rp['average'] < q2]
    poor_regions = [rp for rp in region_performances if rp['average'] < q1]
    
    # Affichage de la classification en colonnes
    class_col1, class_col2, class_col3, class_col4 = st.columns(4)
    
    with class_col1:
        st.markdown(f"""
        <div class="alert-success">
            <h4>🟢 Excellence (Q4)</h4>
            <h2>{len(excellent_regions)}</h2>
            <small>régions ≥ {q3:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
        
        if excellent_regions:
            with st.expander(f"🏆 Top Performers ({len(excellent_regions)} régions)"):
                for i, rp in enumerate(excellent_regions[:5], 1):  # Top 5
                    consistency_icon = "🟢" if rp['consistency'] < 5 else "🟡" if rp['consistency'] < 10 else "🔴"
                    st.write(f"**{i}.** {rp['region'][:20]} - {rp['average']:.1f}% {consistency_icon}")
                if len(excellent_regions) > 5:
                    st.write(f"... et {len(excellent_regions) - 5} autres")
    
    with class_col2:
        st.markdown(f"""
        <div style="background-color: #e7f3ff; border: 1px solid #b3d9ff; color: #0056b3; padding: 1rem; border-radius: 8px; text-align: center;">
            <h4>🔵 Bonne (Q3)</h4>
            <h2>{len(good_regions)}</h2>
            <small>régions {q2:.1f}% - {q3:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
        
        if good_regions:
            with st.expander(f"👍 Bonnes Performances ({len(good_regions)} régions)"):
                for i, rp in enumerate(good_regions[:5], 1):
                    consistency_icon = "🟢" if rp['consistency'] < 5 else "🟡" if rp['consistency'] < 10 else "🔴"
                    st.write(f"**{i}.** {rp['region'][:20]} - {rp['average']:.1f}% {consistency_icon}")
                if len(good_regions) > 5:
                    st.write(f"... et {len(good_regions) - 5} autres")
    
    with class_col3:
        st.markdown(f"""
        <div class="alert-warning">
            <h4>🟡 Moyenne (Q2)</h4>
            <h2>{len(fair_regions)}</h2>
            <small>régions {q1:.1f}% - {q2:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
        
        if fair_regions:
            with st.expander(f"📊 Performances Moyennes ({len(fair_regions)} régions)"):
                for i, rp in enumerate(fair_regions[:5], 1):
                    consistency_icon = "🟢" if rp['consistency'] < 5 else "🟡" if rp['consistency'] < 10 else "🔴"
                    st.write(f"**{i}.** {rp['region'][:20]} - {rp['average']:.1f}% {consistency_icon}")
                if len(fair_regions) > 5:
                    st.write(f"... et {len(fair_regions) - 5} autres")
    
    with class_col4:
        st.markdown(f"""
        <div class="alert-danger">
            <h4>🔴 À Améliorer (Q1)</h4>
            <h2>{len(poor_regions)}</h2>
            <small>régions < {q1:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
        
        if poor_regions:
            with st.expander(f"⚠️ À Améliorer ({len(poor_regions)} régions)"):
                for i, rp in enumerate(poor_regions[:5], 1):
                    consistency_icon = "🟢" if rp['consistency'] < 5 else "🟡" if rp['consistency'] < 10 else "🔴"
                    st.write(f"**{i}.** {rp['region'][:20]} - {rp['average']:.1f}% {consistency_icon}")
                if len(poor_regions) > 5:
                    st.write(f"... et {len(poor_regions) - 5} autres")
    
    # Insights de classification
    st.markdown("#### 🔍 Insights de Classification")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.info(f"""
        **🎯 Distribution des Performances:**
        - Excellence: {(len(excellent_regions)/len(region_performances)*100):.1f}%
        - Bonne: {(len(good_regions)/len(region_performances)*100):.1f}%
        - Moyenne: {(len(fair_regions)/len(region_performances)*100):.1f}%
        - À améliorer: {(len(poor_regions)/len(region_performances)*100):.1f}%
        """)
    
    with insight_col2:
        # Région la plus stable
        most_stable = min(region_performances, key=lambda x: x['consistency'])
        st.success(f"""
        **🎯 Région la Plus Stable:**
        {most_stable['region'][:25]}
        Écart-type: {most_stable['consistency']:.1f}%
        Moyenne: {most_stable['average']:.1f}%
        """)
    
    with insight_col3:
        # Écart de performance
        performance_gap = averages[0] - averages[-1]
        st.warning(f"""
        **📏 Écart de Performance:**
        {performance_gap:.1f} points de %
        Meilleure: {averages[0]:.1f}%
        Plus faible: {averages[-1]:.1f}%
        """)
    
    # Tableau détaillé des moyennes par région
    with st.expander("📋 Moyennes Détaillées par Région"):
        summary_data = []
        
        for _, row in df_filtered.iterrows():
            region = row['REGION']
            monthly_values = [row[col] * 100 for col in tx_cols]
            
            summary_data.append({
                'Région': region,
                'Moyenne': np.mean(monthly_values),
                'Maximum': np.max(monthly_values),
                'Minimum': np.min(monthly_values),
                'Écart-type': np.std(monthly_values),
                'Régularité': "🟢 Stable" if np.std(monthly_values) < 5 else "🟡 Variable" if np.std(monthly_values) < 10 else "🔴 Irrégulière"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Moyenne', ascending=False)
        
        # Configuration des formats selon le type de données dans le résumé
        column_config = {}
        for col in summary_df.columns:
            if col in ['Moyenne', 'Maximum', 'Minimum']:
                # Utiliser le nom de la colonne sélectionnée pour déterminer le format
                selected_col = st.session_state.get('selected_column', '')
                if 'TX' in selected_col.upper():
                    column_config[col] = st.column_config.NumberColumn(
                        f"{col} (%)",
                        format="%.1f%%"
                    )
                else:
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.0f"
                    )
            elif col == 'Écart-type':
                selected_col = st.session_state.get('selected_column', '')
                format_str = "%.4f" if 'TX' in selected_col.upper() else "%.0f"
                column_config[col] = st.column_config.NumberColumn(
                    'Écart-type',
                    format=format_str
                )
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config=column_config
        )

def create_monthly_analysis_visualization(df):
    """Crée la visualisation mensuelle des entrées-sorties-abandons"""
    
    st.markdown("### 📅 Analyse Mensuelle des Flux de Stagiaires")
    
    if df.empty:
        st.warning("⚠️ Aucune donnée mensuelle disponible")
        return
    
    # Extraire les mois disponibles
    months = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
              'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    
    available_months = []
    for month in months:
        if f'{month}_Entrées' in df.columns:
            available_months.append(month)
    
    if not available_months:
        st.warning("⚠️ Aucune donnée mensuelle trouvée dans la structure")
        return
    
    # Configuration de l'analyse
    col_config1, col_config2, col_config3 = st.columns(3)
    
    # Exclure "Total général" de la liste des régions disponibles
    all_regions = [r for r in df['Region'].unique().tolist() if 'Total général' not in str(r)]
    
    with col_config1:
        regions_to_show = st.multiselect(
            "🏷️ Régions à analyser:",
            all_regions,
            default=all_regions,  # Toutes les régions par défaut (sans Total général)
            key="monthly_regions"
        )
    
    with col_config2:
        months_to_show = st.multiselect(
            "📅 Mois à analyser:",
            available_months,
            default=available_months,
            key="months_selection"
        )
    
    with col_config3:
        analysis_type = st.selectbox(
            "📊 Type d'analyse:",
            ["Entrées", "Sorties", "Abandons", "Écart", "Vue d'ensemble"],
            index=0,
            key="monthly_analysis_type"
        )
    
    if not regions_to_show or not months_to_show:
        st.warning("⚠️ Veuillez sélectionner au moins une région et un mois")
        return
    
    # Séparer "Total général" des autres régions
    df_total_general = df[df['Region'].str.contains('Total général', case=False, na=False)].copy()
    df_regions = df[~df['Region'].str.contains('Total général', case=False, na=False)].copy()
    
    # Filtrer les données (sans le Total général)
    df_filtered = df_regions[df_regions['Region'].isin(regions_to_show)].copy()
    
    if analysis_type == "Vue d'ensemble":
        # Créer 4 sous-graphiques pour chaque métrique
        from plotly.subplots import make_subplots
        
        metrics = ['Entrées', 'Sorties', 'Abandons', 'Écart']
        metric_titles = ['📥 Entrées mensuelles', '📤 Sorties mensuelles', 
                        '❌ Abandons mensuels', '📊 Écart mensuel (Entrées-Sorties)']
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=metric_titles,
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Palette de couleurs personnalisée pour éviter les similitudes
        custom_colors = [
            '#FF6B6B',  # Janvier - Rouge vif
            '#4ECDC4',  # Février - Turquoise
            '#45B7D1',  # Mars - Bleu
            '#96CEB4',  # Avril - Vert menthe
            '#FFEAA7',  # Mai - Jaune
            '#DDA0DD',  # Juin - Violet clair
            '#98D8C8',  # Juillet - Vert aqua
            '#F7DC6F',  # Août - Jaune orangé
            '#BB8FCE',  # Septembre - Violet foncé (différent de Janvier)
            '#F8C471',  # Octobre - Orange
            '#85C1E9',  # Novembre - Bleu clair
            '#F1948A'   # Décembre - Rose
        ]
        
        for metric_idx, metric in enumerate(metrics):
            # Pour chaque mois, créer une barre groupée
            for month_idx, month in enumerate(months_to_show):
                regions_for_month = []
                values_for_month = []
                
                for region in regions_to_show:
                    region_data = df_filtered[df_filtered['Region'] == region].iloc[0]
                    col_name = f'{month}_{metric}'
                    
                    if col_name in region_data.index:
                        value = region_data[col_name]
                    else:
                        value = 0
                    
                    regions_for_month.append(region)
                    values_for_month.append(value)
                
                fig.add_trace(
                    go.Bar(
                        x=regions_for_month,
                        y=values_for_month,
                        name=month,
                        marker_color=custom_colors[month_idx % len(custom_colors)],
                        showlegend=(metric_idx == 0),  # Légende seulement sur le premier graphique
                        text=[f"{val:,.0f}" for val in values_for_month],
                        textposition='outside',
                        hovertemplate=f'<b>{month}</b><br>%{{x}}: %{{y:,.0f}}<extra></extra>'
                    ),
                    row=metric_idx+1, col=1
                )
        
        fig.update_layout(
            title="📅 Vue d'ensemble mensuelle des flux de stagiaires par région",
            height=1200,
            barmode='group',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Mettre à jour les axes
        for i in range(4):
            fig.update_xaxes(tickangle=-45, row=i+1, col=1, title_text="Régions")
            fig.update_yaxes(title_text="Nombre", row=i+1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Graphique en barres avec régions en abscisse et mois groupés
        fig = go.Figure()
        # Palette de couleurs personnalisée pour éviter les similitudes
        custom_colors = [
            '#FF6B6B',  # Janvier - Rouge vif
            '#4ECDC4',  # Février - Turquoise
            '#45B7D1',  # Mars - Bleu
            '#96CEB4',  # Avril - Vert menthe
            '#FFEAA7',  # Mai - Jaune
            '#DDA0DD',  # Juin - Violet clair
            '#98D8C8',  # Juillet - Vert aqua
            '#F7DC6F',  # Août - Jaune orangé
            '#BB8FCE',  # Septembre - Violet foncé (différent de Janvier)
            '#F8C471',  # Octobre - Orange
            '#85C1E9',  # Novembre - Bleu clair
            '#F1948A'   # Décembre - Rose
        ]
        
        # Réorganiser les données : régions en X, barres groupées par mois
        for month_idx, month in enumerate(months_to_show):
            regions_for_month = []
            values_for_month = []
            
            for region in regions_to_show:
                region_data = df_filtered[df_filtered['Region'] == region].iloc[0]
                col_name = f'{month}_{analysis_type}'
                
                if col_name in region_data.index:
                    value = region_data[col_name]
                else:
                    value = 0
                
                regions_for_month.append(region)
                values_for_month.append(value)
            
            fig.add_trace(go.Bar(
                x=regions_for_month,
                y=values_for_month,
                name=month,
                marker_color=custom_colors[month_idx % len(custom_colors)],
                text=[f"{val:,.0f}" for val in values_for_month],
                textposition='outside',
                hovertemplate=f'<b>{month}</b><br>%{{x}}: %{{y:,.0f}}<extra></extra>'
            ))
        
        metric_emojis = {
            "Entrées": "📥", 
            "Sorties": "📤", 
            "Abandons": "❌", 
            "Écart": "📊"
        }
        
        # ========== AJOUTER COURBE MOYENNE POUR LES ENTRÉES ==========
        monthly_averages = []
        months_labels = []
        
        if analysis_type == "Entrées":
            # Calculer les moyennes par mois
            for month in months_to_show:
                col_name = f'{month}_{analysis_type}'
                # Calculer la moyenne de toutes les régions pour ce mois
                month_values = []
                for region in regions_to_show:
                    region_data = df_filtered[df_filtered['Region'] == region].iloc[0]
                    if col_name in region_data.index:
                        value = region_data[col_name]
                        if pd.notna(value) and value != 0:  # Exclure les valeurs nulles et zéros
                            month_values.append(value)
                
                if month_values:  # Si on a des valeurs pour ce mois
                    average = sum(month_values) / len(month_values)
                    monthly_averages.append(average)
                    months_labels.append(month)
            
            if monthly_averages and len(regions_to_show) > 0:
                # Créer une courbe séparée au-dessus du graphique principal
                # Utiliser des positions numériques pour avoir une vraie courbe
                x_numeric = list(range(len(months_labels)))
                
                # Ajouter la courbe moyenne avec des positions numériques
                fig.add_trace(go.Scatter(
                    x=x_numeric,
                    y=monthly_averages,
                    mode='lines+markers+text',
                    name='📈 Moyenne par Mois',
                    line=dict(color='#e74c3c', width=4),
                    marker=dict(size=12, color='#e74c3c', symbol='circle'),
                    text=[f"{avg:.0f}" for avg in monthly_averages],
                    textposition='top center',
                    hovertemplate='<b>%{customdata}</b><br>Moyenne: %{y:.1f} entrées<extra></extra>',
                    customdata=months_labels,
                    yaxis='y2',  # Utiliser un axe secondaire pour la courbe
                    xaxis='x2'   # Utiliser un axe X secondaire pour la courbe
                ))
                
                # Configurer l'axe X secondaire pour la courbe
                fig.update_layout(
                    xaxis2=dict(
                        overlaying='x',
                        side='top',
                        tickmode='array',
                        tickvals=x_numeric,
                        ticktext=months_labels,
                        showgrid=False,
                        title=dict(text="Mois (Courbe Moyenne)", font=dict(color="#e74c3c"))
                    )
                )
        
        fig.update_layout(
            title=f"{metric_emojis.get(analysis_type, '📊')} {analysis_type} par Région et Mois" + 
                  (" avec Courbe Moyenne" if analysis_type == "Entrées" else ""),
            xaxis_title="Régions",
            yaxis_title=f"Nombre de {analysis_type.lower()}",
            height=700,
            barmode='group',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis_tickangle=-45
        )
        
        # Configuration de l'axe secondaire pour la courbe moyenne (si applicable)
        if analysis_type == "Entrées":
            fig.update_layout(
                yaxis2=dict(
                    title=dict(text="Moyenne par Région", font=dict(color="#e74c3c")),
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    tickfont=dict(color="#e74c3c")
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ========== AFFICHAGE DES STATISTIQUES DE LA COURBE MOYENNE ==========
        if analysis_type == "Entrées" and monthly_averages and months_labels:
            st.markdown("---")
            st.markdown("### 📈 Statistiques de la Courbe Moyenne des Entrées")
            
            col_avg1, col_avg2, col_avg3, col_avg4 = st.columns(4)
            
            # Moyenne générale
            general_average = sum(monthly_averages) / len(monthly_averages)
            with col_avg1:
                st.metric(
                    "📊 Moyenne Générale",
                    f"{general_average:.1f}",
                    help="Moyenne de toutes les moyennes mensuelles"
                )
            
            # Meilleur mois
            if monthly_averages and months_labels:
                best_month_idx = monthly_averages.index(max(monthly_averages))
                best_month = months_labels[best_month_idx]
                best_value = monthly_averages[best_month_idx]
                with col_avg2:
                    st.metric(
                        "🏆 Meilleur Mois",
                        best_month,
                        f"{best_value:.1f}"
                    )
                
                # Mois le plus faible
                worst_month_idx = monthly_averages.index(min(monthly_averages))
                worst_month = months_labels[worst_month_idx]
                worst_value = monthly_averages[worst_month_idx]
                with col_avg3:
                    st.metric(
                        "📉 Mois le Plus Faible",
                        worst_month,
                        f"{worst_value:.1f}"
                    )
                
                # Écart entre le meilleur et le pire
                ecart = best_value - worst_value
                with col_avg4:
                    st.metric(
                        "📏 Écart Max-Min",
                        f"{ecart:.1f}",
                        help="Différence entre le meilleur et le pire mois"
                    )
    
    # ========== SECTION TOTAUX ==========
    st.markdown("---")
    st.markdown("### 📊 Totaux par Région et Mois")
    
    # Calculer les totaux par région pour la métrique sélectionnée
    if analysis_type != "Vue d'ensemble":
        totals_data = []
        
        # Calculer pour chaque région filtrée
        for region in regions_to_show:
            region_data = df_filtered[df_filtered['Region'] == region]
            if not region_data.empty:
                region_row = region_data.iloc[0]
                
                # Total pour cette région sur tous les mois sélectionnés
                total_region = 0
                monthly_details = {}
                
                for month in months_to_show:
                    col_name = f'{month}_{analysis_type}'
                    if col_name in region_row.index:
                        value = region_row[col_name]
                        total_region += value
                        monthly_details[month] = value
                    else:
                        monthly_details[month] = 0
                
                totals_data.append({
                    'Région': region,
                    'Total': total_region,
                    **monthly_details
                })
        
        # Ajouter le Total Général s'il existe
        if not df_total_general.empty:
            total_general_row = df_total_general.iloc[0]
            total_general = 0
            monthly_details_general = {}
            
            for month in months_to_show:
                col_name = f'{month}_{analysis_type}'
                if col_name in total_general_row.index:
                    value = total_general_row[col_name]
                    total_general += value
                    monthly_details_general[month] = value
                else:
                    monthly_details_general[month] = 0
            
            totals_data.append({
                'Région': '🏆 Total Général',
                'Total': total_general,
                **monthly_details_general
            })
        
        # Créer un DataFrame pour l'affichage
        totals_df = pd.DataFrame(totals_data)
        
        # Trier par Total décroissant (mais garder Total Général en bas)
        total_general_row = totals_df[totals_df['Région'] == '🏆 Total Général']
        other_rows = totals_df[totals_df['Région'] != '🏆 Total Général'].sort_values('Total', ascending=False)
        totals_df = pd.concat([other_rows, total_general_row], ignore_index=True)
        
        # Affichage avec mise en forme
        st.markdown(f"#### 📊 Totaux pour : **{analysis_type}**")
        
        # Configuration des colonnes
        column_config = {
            'Région': st.column_config.TextColumn('Région', width='medium'),
            'Total': st.column_config.NumberColumn(
                f'Total {analysis_type}',
                format="%.0f",
                help=f"Total sur {len(months_to_show)} mois"
            )
        }
        
        # Ajouter les colonnes des mois
        for month in months_to_show:
            column_config[month] = st.column_config.NumberColumn(
                month,
                format="%.0f"
            )
        
        # Styliser le dataframe
        st.dataframe(
            totals_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True,
            height=min(600, (len(totals_df) + 1) * 35 + 38)  # Hauteur dynamique
        )
        
        # Statistiques rapides
        col_total1, col_total2, col_total3 = st.columns(3)
        
        if not df_total_general.empty:
            total_general_value = totals_df[totals_df['Région'] == '🏆 Total Général']['Total'].iloc[0]
            with col_total1:
                st.metric(
                    "🏆 Total Général",
                    f"{total_general_value:,.0f}",
                    help=f"Somme totale de tous les {analysis_type.lower()}"
                )
        
        # Meilleure région
        best_region_data = totals_df[totals_df['Région'] != '🏆 Total Général'].iloc[0] if len(totals_df) > 1 else None
        if best_region_data is not None:
            with col_total2:
                st.metric(
                    "🥇 Meilleure Région",
                    best_region_data['Région'][:20],
                    f"{best_region_data['Total']:,.0f}"
                )
        
        # Moyenne par région
        avg_by_region = totals_df[totals_df['Région'] != '🏆 Total Général']['Total'].mean()
        with col_total3:
            st.metric(
                "📊 Moyenne par Région",
                f"{avg_by_region:,.0f}",
                help="Moyenne sur toutes les régions sélectionnées"
            )
    
    else:
        # Vue d'ensemble - afficher les totaux pour toutes les métriques
        st.markdown("#### 📊 Totaux pour toutes les métriques")
        
        metrics = ['Entrées', 'Sorties', 'Abandons', 'Écart']
        all_totals_data = []
        
        for region in regions_to_show:
            region_data = df_filtered[df_filtered['Region'] == region]
            if not region_data.empty:
                region_row = region_data.iloc[0]
                region_totals = {'Région': region}
                
                for metric in metrics:
                    total_metric = 0
                    for month in months_to_show:
                        col_name = f'{month}_{metric}'
                        if col_name in region_row.index:
                            total_metric += region_row[col_name]
                    region_totals[metric] = total_metric
                
                all_totals_data.append(region_totals)
        
        # Ajouter le Total Général
        if not df_total_general.empty:
            total_general_row = df_total_general.iloc[0]
            general_totals = {'Région': '🏆 Total Général'}
            
            for metric in metrics:
                total_metric = 0
                for month in months_to_show:
                    col_name = f'{month}_{metric}'
                    if col_name in total_general_row.index:
                        total_metric += total_general_row[col_name]
                    general_totals[metric] = str(int(total_metric))
            
            all_totals_data.append(general_totals)
        
        # Créer le DataFrame
        all_totals_df = pd.DataFrame(all_totals_data)
        
        # Trier par Entrées décroissantes (Total Général en bas)
        total_general_row_vue = all_totals_df[all_totals_df['Région'] == '🏆 Total Général']
        other_rows_vue = all_totals_df[all_totals_df['Région'] != '🏆 Total Général'].sort_values('Entrées', ascending=False)
        all_totals_df = pd.concat([other_rows_vue, total_general_row_vue], ignore_index=True)
        
        # Configuration des colonnes
        column_config_vue = {
            'Région': st.column_config.TextColumn('Région', width='medium'),
            'Entrées': st.column_config.NumberColumn('📥 Entrées', format="%.0f"),
            'Sorties': st.column_config.NumberColumn('📤 Sorties', format="%.0f"),
            'Abandons': st.column_config.NumberColumn('❌ Abandons', format="%.0f"),
            'Écart': st.column_config.NumberColumn('📊 Écart', format="%.0f")
        }
        
        st.dataframe(
            all_totals_df,
            use_container_width=True,
            column_config=column_config_vue,
            hide_index=True,
            height=min(600, (len(all_totals_df) + 1) * 35 + 38)
        )
    
    # Statistiques mensuelles
    create_monthly_statistics(df_filtered, months_to_show, analysis_type)

def create_monthly_statistics(df, months_to_show, analysis_type):
    """Crée les statistiques pour l'analyse mensuelle"""
    
    st.markdown("### 📊 Statistiques Mensuelles")
    
    if analysis_type != "Vue d'ensemble":
        # Calculer les totaux par mois
        monthly_totals = {}
        for month in months_to_show:
            col_name = f'{month}_{analysis_type}'
            if col_name in df.columns:
                monthly_totals[month] = df[col_name].sum()
            else:
                monthly_totals[month] = 0
        
        # Identifier le meilleur et le pire mois
        best_month = "N/A"
        worst_month = "N/A"
        total_sum = 0
        average_monthly = 0
        
        if monthly_totals:
            best_month = max(monthly_totals.keys(), key=lambda k: monthly_totals[k])
            worst_month = min(monthly_totals.keys(), key=lambda k: monthly_totals[k])
            total_sum = sum(monthly_totals.values())
            average_monthly = total_sum / len(monthly_totals)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric(
                f"🥇 Meilleur mois ({analysis_type})",
                best_month if monthly_totals else "N/A",
                f"{monthly_totals.get(best_month, 0):,.0f}" if monthly_totals else "0"
            )
        
        with col_stat2:
            st.metric(
                f"📉 Mois le plus faible",
                worst_month if monthly_totals else "N/A",
                f"{monthly_totals.get(worst_month, 0):,.0f}" if monthly_totals else "0"
            )
        
        with col_stat3:
            st.metric(
                f"📊 Total période",
                f"{total_sum:,.0f}",
                f"({len(months_to_show)} mois)"
            )
        
        with col_stat4:
            st.metric(
                "📈 Moyenne mensuelle",
                f"{average_monthly:,.0f}",
                help=f"Moyenne sur {len(months_to_show)} mois"
            )
    
    else:
        # Statistiques globales pour vue d'ensemble
        total_entries = sum(df[f'{month}_Entrées'].sum() for month in months_to_show if f'{month}_Entrées' in df.columns)
        total_exits = sum(df[f'{month}_Sorties'].sum() for month in months_to_show if f'{month}_Sorties' in df.columns)
        total_dropouts = sum(df[f'{month}_Abandons'].sum() for month in months_to_show if f'{month}_Abandons' in df.columns)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("📥 Total Entrées", f"{total_entries:,.0f}")
        
        with col_stat2:
            st.metric("📤 Total Sorties", f"{total_exits:,.0f}")
        
        with col_stat3:
            st.metric("❌ Total Abandons", f"{total_dropouts:,.0f}")
        
        with col_stat4:
            if total_entries > 0:
                retention_rate = ((total_entries - total_dropouts) / total_entries) * 100
                st.metric("💪 Taux de Rétention", f"{retention_rate:.1f}%")
            else:
                st.metric("💪 Taux de Rétention", "N/A")

def show_entrees_sorties_analysis():
    """Page de l'analyseur entrées-sorties-abandons"""
    
    # Titre principal
    st.markdown('<h1 class="main-title">📊 Analyseur Flux de Stagiaires</h1>', 
                unsafe_allow_html=True)
    
    # Section d'import de fichier
    st.markdown("## 📁 Import de Fichier")
    st.markdown("*Choisissez le fichier Excel à analyser*")
    
    # Options d'import
    import_col1, import_col2 = st.columns(2)
    
    with import_col1:
        import_method = st.radio(
            "🔧 Méthode d'import:",
            ["Upload d'un nouveau fichier", "Chemin personnalisé"],
            key="entrees_sorties_import_method"
        )
    
    uploaded_file = None
    custom_path = None
    file_info = ""
    
    if import_method == "Upload d'un nouveau fichier":
        with import_col2:
            uploaded_file = st.file_uploader(
                "📤 Choisir un fichier Excel:",
                type=['xlsx', 'xls', 'xlsb'],
                help="Sélectionnez un fichier Excel avec les données de flux de stagiaires",
                key="entrees_sorties_file_uploader"
            )
            if uploaded_file is not None:
                # Afficher uniquement le nom du fichier (suppression de la mention des feuilles)
                file_info = f"📁 Fichier: {uploaded_file.name}"
    
    else:  # Chemin personnalisé
        with import_col2:
            custom_path = st.text_input(
                "📂 Chemin du fichier:",
                placeholder="Ex: C:/Dashboard/mon_fichier.xlsx",
                help="Entrez le chemin complet vers votre fichier Excel",
                key="entrees_sorties_custom_path"
            )
            if custom_path:
                # Afficher uniquement le chemin du fichier (suppression de la mention des feuilles)
                file_info = f"📁 Fichier: {custom_path}"
            else:
                file_info = "📁 Veuillez spécifier un chemin de fichier"
    
    # Affichage des informations du fichier
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
        <h3>📊 Analyse des Flux de Stagiaires par Région et Financeur</h3>
        <p><strong>{file_info}</strong> | <strong>Dernière mise à jour:</strong> {datetime.now().strftime("%d/%m/%Y à %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérification des prérequis selon la méthode
    if import_method == "Upload d'un nouveau fichier" and uploaded_file is None:
        st.warning("⚠️ Veuillez sélectionner un fichier Excel à analyser")
        return
    elif import_method == "Chemin personnalisé" and not custom_path:
        st.warning("⚠️ Veuillez entrer le chemin vers votre fichier Excel")
        return
    
    # Chargement des données selon la méthode choisie
    # Pour l'analyse par financeurs, on utilise toujours Feuil1
    if import_method == "Upload d'un nouveau fichier":
        df, error = load_entrees_sorties_data(uploaded_file=uploaded_file, sheet_name="Feuil1")
    else:  # Chemin personnalisé
        df, error = load_entrees_sorties_data(file_path=custom_path, sheet_name="Feuil1")
    
    if error:
        st.error(error)
        
        # Messages d'aide selon la méthode d'import
        if import_method == "Upload d'un nouveau fichier":
            st.info("💡 Vérifiez que votre fichier Excel n'est pas corrompu et contient les données de flux de stagiaires")
        else:
            st.info("💡 Vérifiez le chemin du fichier et qu'il existe bien sur votre système")
        
        st.markdown("### 📋 Format de Fichier Attendu")
        st.markdown("""
        **Le fichier Excel doit contenir :**
        - Une colonne 'Région' avec les noms des régions
        - Une colonne 'Financeurs' avec les types de financeurs
        - Une colonne 'Nb. de stagiaires entrés' avec le nombre d'entrées
        - Une colonne 'Nb. de stagaires sortis' avec le nombre de sorties
        - Une colonne 'DONT ABANDONS' avec le nombre d'abandons
        - Une colonne 'Ecart (entrées-sorties)' avec l'écart calculé
        - **Format des données** : Valeurs numériques entières
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
        warnings.append("❌ Colonne 'REGION' (ou 'Région') non trouvée - impossible de créer les analyses")
    
    if 'FINANCEURS' not in df.columns:
        warnings.append("❌ Colonne 'FINANCEURS' (ou 'Financeurs') non trouvée - impossible de créer les analyses")
    
    if warnings:
        for warning in warnings:
            st.error(warning)
        
        st.info("💡 Votre fichier doit contenir toutes les colonnes requises pour l'analyse de flux de stagiaires")
        
        # Afficher les colonnes disponibles pour aider l'utilisateur
        with st.expander("📋 Colonnes Disponibles dans votre Fichier"):
            st.write("Colonnes trouvées:")
            for col in df.columns:
                st.write(f"• {col}")
        return
    
    # Génération de l'analyse avec onglets
    st.markdown("---")
    
    # Onglets d'analyse
    tab1, tab2, tab3 = st.tabs(["📊 Analyse par Financeurs", "📅 Analyse Mensuelle", "📈 Comparatif Annuel"])
    
    with tab1:
        st.markdown("### 💰 Analyse des flux par région et financeur")
        try:
            create_entrees_sorties_visualization(df)
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération des graphiques par financeurs: {str(e)}")
            
            # Debug info
            with st.expander("🔧 Informations de Debug"):
                st.write("Colonnes disponibles:", df.columns.tolist())
                st.write("Forme du DataFrame:", df.shape)
                st.write("Types de données:", df.dtypes.to_dict())
                st.write("Échantillon de données:", df.head())
    
    with tab2:
        st.markdown("### 📅 Analyse temporelle des flux de stagiaires")
        try:
            # Charger les données mensuelles de la Feuil2 depuis le même fichier
            if import_method == "Upload d'un nouveau fichier":
                df_monthly = load_monthly_data(uploaded_file=uploaded_file)
            else:
                df_monthly = load_monthly_data(file_path=custom_path)
            
            if df_monthly.empty:
                st.warning("⚠️ Aucune donnée mensuelle trouvée dans la Feuil2 du fichier")
                st.info("💡 Cette analyse nécessite la Feuil2 avec des données structurées par mois")
            else:
                create_monthly_analysis_visualization(df_monthly)
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de l'analyse mensuelle: {str(e)}")
            
            # Debug info
            with st.expander("🔧 Informations de Debug - Analyse Mensuelle"):
                st.write("Erreur détaillée:", str(e))
                st.markdown("""
                **Format attendu pour l'analyse mensuelle (Feuil2) :**
                - Première ligne : Noms des mois (Janvier, Février, etc.)
                - Deuxième ligne : Sous-catégories (Entrées, Sorties, Abandons, Écart)
                - Lignes suivantes : Données par région
                """)
                
                # Essayer de charger et afficher la structure
                try:
                    if import_method == "Upload d'un nouveau fichier" and uploaded_file is not None:
                        df_raw = pd.read_excel(uploaded_file, sheet_name="Feuil2", header=None)
                    elif custom_path:
                        df_raw = pd.read_excel(custom_path, sheet_name="Feuil2", header=None)
                    else:
                        st.write("Aucun fichier disponible pour le debug")
                        df_raw = None
                    
                    if df_raw is not None:
                        st.write("Structure de la Feuil2:")
                        st.write("Forme:", df_raw.shape)
                        st.write("Premières lignes:", df_raw.head())
                except Exception as debug_error:
                    st.write("Impossible de charger la Feuil2:", str(debug_error))
    
    with tab3:
        st.markdown("### 📈 Analyse par Région et Financeur (Feuil3)")
        try:
            # Charger les données de la Feuil3
            if import_method == "Upload d'un nouveau fichier":
                df_feuil3, error_f3 = load_feuil3_data(uploaded_file=uploaded_file)
            else:
                df_feuil3, error_f3 = load_feuil3_data(file_path=custom_path)
            
            if error_f3:
                st.error(error_f3)
                st.info("💡 Cette analyse nécessite la Feuil3 du fichier Excel avec les données par région et financeur")
            elif df_feuil3 is None or df_feuil3.empty:
                st.warning("⚠️ Aucune donnée trouvée dans la Feuil3")
                st.info("💡 Vérifiez que votre fichier Excel contient une Feuil3 avec les colonnes REGION, FINANCEURS et les métriques associées")
            else:
                create_feuil3_visualization(df_feuil3)
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de l'analyse Feuil3: {str(e)}")
            
            # Debug info
            with st.expander("🔧 Informations de Debug - Analyse Feuil3"):
                st.write("Erreur détaillée:", str(e))
                st.markdown("""
                **Format attendu pour la Feuil3 :**
                - Colonne 'REGION' ou 'Région' avec les noms des régions
                - Colonne 'FINANCEURS' ou 'Financeur' avec les types de financeurs
                - Colonnes numériques avec les métriques à analyser
                - Éviter les lignes "Total général" (elles seront filtrées automatiquement)
                """)
                
                # Essayer de charger et afficher la structure
                try:
                    if import_method == "Upload d'un nouveau fichier" and uploaded_file is not None:
                        df_raw_f3 = pd.read_excel(uploaded_file, sheet_name="Feuil3")
                    elif custom_path:
                        df_raw_f3 = pd.read_excel(custom_path, sheet_name="Feuil3")
                    else:
                        st.write("Aucun fichier disponible pour le debug")
                        df_raw_f3 = None
                    
                    if df_raw_f3 is not None:
                        st.write("Structure de la Feuil3:")
                        st.write("Forme:", df_raw_f3.shape)
                        st.write("Colonnes:", df_raw_f3.columns.tolist())
                        st.write("Premières lignes:", df_raw_f3.head())
                except Exception as debug_error:
                    st.write("Impossible de charger la Feuil3:", str(debug_error))

# ========== PAGES DE L'APPLICATION ==========

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

def show_monthly_production():
    """Page de l'analyseur de production mensuelle"""
    
    # Titre principal
    st.markdown('<h1 class="main-title">📈 Analyseur Production Mensuelle - HORS PAE</h1>', 
                unsafe_allow_html=True)
    
    # Section d'import de fichier
    st.markdown("## 📁 Import de Fichier")
    st.markdown("*Choisissez le fichier Excel de production mensuelle à analyser*")
    
    # Options d'import
    import_col1, import_col2 = st.columns(2)
    
    with import_col1:
        import_method = st.radio(
            "🔧 Méthode d'import:",
            ["Upload d'un nouveau fichier", "Chemin personnalisé"],
            key="prod_import_method"
        )
    
    uploaded_file = None
    custom_path = None
    file_info = ""
    
    if import_method == "Upload d'un nouveau fichier":
        with import_col2:
            uploaded_file = st.file_uploader(
                "📤 Choisir un fichier Excel:",
                type=['xlsx', 'xls'],
                help="Sélectionnez un fichier Excel (.xlsx ou .xls) avec les données de production mensuelle",
                key="prod_file_uploader"
            )
            if uploaded_file is not None:
                file_info = f"📁 Fichier: {uploaded_file.name}"
    
    elif import_method == "Chemin personnalisé":
        with import_col2:
            custom_path = st.text_input(
                "📂 Chemin du fichier:",
                placeholder="Ex: C:/Dashboard/production_mensuelle.xlsx",
                help="Entrez le chemin complet vers votre fichier Excel de production",
                key="prod_custom_path"
            )
            if custom_path:
                file_info = f"📁 Fichier: {custom_path}"
            else:
                file_info = "📁 Veuillez spécifier un chemin de fichier"
    
    # Affichage des informations du fichier
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
        <h3>📊 Analyse de Production HORS PAE par Région et Financeur</h3>
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
        df, error = load_monthly_production_data(uploaded_file=uploaded_file)
    else:  # Chemin personnalisé
        df, error = load_monthly_production_data(file_path=custom_path)
    
    if error:
        st.error(error)
        
        # Messages d'aide selon la méthode d'import
        if import_method == "Upload d'un nouveau fichier":
            st.info("💡 Vérifiez que votre fichier Excel n'est pas corrompu et contient des données de production")
        else:
            st.info("💡 Vérifiez le chemin du fichier et qu'il existe bien sur votre système")
        
        st.markdown("### 📋 Format de Fichier Attendu")
        st.markdown("""
        **Le fichier Excel doit contenir :**
        - Une colonne 'REGION' avec les noms des régions
        - Une colonne 'FINANCEURS' avec les types de financeurs (B2C-CPF, B2C-CPFT, etc.)
        - Des colonnes 'PRODUCTION HORS PAE [MOIS]' pour chaque mois (JANVIER, FEVRIER, etc.)
        - Une colonne 'TOTAL' avec les totaux par ligne
        - **Format des données** : Valeurs numériques, "-" pour les valeurs manquantes
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
        warnings.append("❌ Colonne 'REGION' non trouvée - impossible de créer les analyses")
    
    if 'FINANCEURS' not in df.columns:
        warnings.append("❌ Colonne 'FINANCEURS' non trouvée - impossible de créer les analyses")
    
    production_cols = [col for col in df.columns if 'PRODUCTION HORS PAE' in col]
    if len(production_cols) == 0:
        warnings.append("❌ Aucune colonne 'PRODUCTION HORS PAE' trouvée - impossible de créer les analyses")
    
    if 'TOTAL' not in df.columns:
        warnings.append("⚠️ Colonne 'TOTAL' non trouvée - certaines analyses pourraient ne pas fonctionner")
        
    if warnings:
        for warning in warnings:
            if "❌" in warning:
                st.error(warning)
            else:
                st.warning(warning)
        
        if any("❌" in warning for warning in warnings):
            st.info("💡 Votre fichier doit contenir les colonnes 'REGION', 'FINANCEURS' et au moins une colonne 'PRODUCTION HORS PAE [MOIS]'")
            
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
        financeurs_count = df['FINANCEURS'].dropna().nunique()
        
        st.metric("🏢 Nombre de Régions", regions_count)
        st.metric("💰 Nombre de Financeurs", financeurs_count)
        st.metric("📊 Lignes de Données", len(df))
        
        # Informations sur les financeurs
        st.subheader("💰 Financeurs Disponibles")
        financeurs = df['FINANCEURS'].dropna().unique()
        for financeur in financeurs:
            st.write(f"• {financeur}")
            
        # Informations sur les colonnes de production
        st.subheader("📊 Mois Disponibles")
        for col in production_cols:
            month_name = col.replace('PRODUCTION HORS PAE ', '')
            st.write(f"• {month_name}")
            
        st.subheader("📋 Colonnes Détectées")
        st.metric("🗂️ Total Colonnes", len(df.columns))
        st.metric("📈 Mois Production", len(production_cols))
    
    # Génération de l'analyse
    st.markdown("---")
    
    # Identifier les colonnes de production mensuelle
    production_cols = [col for col in df.columns if 'PRODUCTION HORS PAE' in col]
    
    # Afficher seulement la nouvelle visualisation par financeur
    if production_cols:
        create_cumulative_curve_new(df, production_cols)
    else:
        st.warning("⚠️ Aucune colonne de production mensuelle trouvée")
        
        # Debug info
        with st.expander("🔧 Informations de Debug"):
            st.write("Colonnes disponibles:", df.columns.tolist())
            st.write("Forme du DataFrame:", df.shape)
            st.write("Types de données:", df.dtypes.to_dict())
            st.write("Échantillon de données:", df.head())

def show_weekly_analysis():
    """Page de l'Optimisation des plateaux (code existant)"""
    
    # Titre principal
    st.markdown('<h1 class="main-title">📊 Optimisation des plateaux - Production Régionale</h1>', 
                unsafe_allow_html=True)
    
    # Section d'import de fichier
    st.markdown("## 📁 Import de Fichier")
    st.markdown("*Choisissez le fichier Excel à analyser*")
    
    # Options d'import
    import_col1, import_col2 = st.columns(2)
    
    with import_col1:
        import_method = st.radio(
            "🔧 Méthode d'import:",
            ["Upload d'un nouveau fichier", "Chemin personnalisé"],
            key="import_method"
        )
    
    uploaded_file = None
    custom_path = None
    file_info = ""
    
    if import_method == "Upload d'un nouveau fichier":
        with import_col2:
            uploaded_file = st.file_uploader(
                "📤 Choisir un fichier Excel:",
                type=['xlsx', 'xls'],
                help="Sélectionnez un fichier Excel (.xlsx ou .xls)"
            )
            if uploaded_file is not None:
                file_info = f"📁 Fichier: {uploaded_file.name}"
    
    elif import_method == "Chemin personnalisé":
        with import_col2:
            custom_path = st.text_input(
                "📂 Chemin du fichier:",
                placeholder="Ex: C:/Dashboard/mon_fichier.xlsx",
                help="Entrez le chemin complet vers votre fichier Excel"
            )
            if custom_path:
                file_info = f"📁 Fichier: {custom_path}"
            else:
                file_info = "📁 Veuillez spécifier un chemin de fichier"
    
    # Affichage des informations du fichier
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
        <h3>🎯 Suivi Hebdomadaire des Performances de Production par Région</h3>
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
        df, error = load_weekly_data(uploaded_file=uploaded_file)
    else:  # Chemin personnalisé
        df, error = load_weekly_data(file_path=custom_path)
    
    if error:
        st.error(error)
        
        # Messages d'aide selon la méthode d'import
        if import_method == "Upload d'un nouveau fichier":
            st.info("💡 Vérifiez que votre fichier Excel n'est pas corrompu et contient des données")
        else:
            st.info("💡 Vérifiez le chemin du fichier et qu'il existe bien sur votre système")
        
        st.markdown("### 📋 Format de Fichier Attendu")
        st.markdown("""
        **Le fichier Excel doit contenir :**
        - Une colonne 'REGION' avec les noms des régions
        - Des colonnes numériques pour l'analyse
        - Optionnel : Colonnes 'TX DE CAPACITE DE PRODUCTION [MOIS]' pour l'évolution mensuelle
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
        warnings.append("⚠️ Colonne 'REGION' non trouvée - certaines fonctionnalités pourraient ne pas fonctionner")
    
    tx_cols = [col for col in df.columns if 'TX DE CAPACITE' in col]
    if len(tx_cols) == 0:
        warnings.append("⚠️ Aucune colonne 'TX DE CAPACITE' trouvée - l'évolution mensuelle ne sera pas disponible")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        warnings.append("❌ Aucune colonne numérique trouvée - impossible de créer des analyses")
        
    if warnings:
        for warning in warnings:
            if "❌" in warning:
                st.error(warning)
            else:
                st.warning(warning)
        
        if "❌" in str(warnings):
            st.info("💡 Votre fichier doit contenir au moins une colonne numérique pour pouvoir être analysé")
            return
    
    # Sidebar avec informations sur les données
    with st.sidebar:
        st.header("📋 Informations du Dataset")
        st.metric("📊 Nombre de Régions", len(df))
        st.metric("📈 Colonnes Disponibles", len(df.columns))
        
        # Légende des classifications
        st.subheader("🎯 Légende Classifications")
        st.markdown("""
        - 🟢 **Excellence** (Q4): Top 25%
        - 🔵 **Bonne** (Q3): 50-75%
        - 🟡 **Moyenne** (Q2): 25-50%
        - 🔴 **À Améliorer** (Q1): Bottom 25%
        """)
    
    # Interface de sélection
    st.markdown("## ⚙️ Configuration de l'Analyse")
    
    # Sélection des colonnes pour l'analyse
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ['REGION']  # On force sur REGION comme catégorie principale
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        category_col = st.selectbox(
            "🏷️ Catégorie (Axe X):",
            categorical_cols,
            key="weekly_category",
            help="Colonne utilisée pour regrouper les données"
        )
    
    with col_config2:
        value_col = st.selectbox(
            "📊 Valeur à Analyser (Axe Y):",
            numeric_cols,
            key="weekly_value",
            help="Métrique à analyser et comparer"
        )
    
    with col_config3:
        agg_method_display = st.selectbox(
            "🔢 Méthode d'Agrégation:",
            ["Moyenne", "Total", "Maximum", "Minimum", "Nombre"],
            key="weekly_agg",
            help="Comment agréger les données si plusieurs valeurs par catégorie"
        )
    
    # Mapping français vers anglais pour les calculs
    agg_mapping = {
        "Total": "sum",
        "Moyenne": "mean", 
        "Nombre": "count",
        "Maximum": "max",
        "Minimum": "min"
    }
    agg_method = agg_mapping[agg_method_display]
    
    # Validation des sélections
    if not category_col or not value_col:
        st.warning("⚠️ Veuillez sélectionner une catégorie et une valeur à analyser")
        return
    
    # Génération de l'analyse
    st.markdown("---")
    st.markdown("## 📊 Analyse Comparative")
    
    try:
        create_comparison_visualization(df, category_col, value_col, agg_method)
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du graphique: {str(e)}")
        
        # Debug info pour développement
        with st.expander("🔧 Informations de Debug"):
            st.write("Colonnes disponibles:", df.columns.tolist())
            st.write("Types de données:", df.dtypes.to_dict())
            st.write("Forme du DataFrame:", df.shape)
    
    # Nouvelle section: Évolution mensuelle
    st.markdown("---")
    st.markdown("## 📈 Analyse d'Évolution Mensuelle")
    
    try:
        create_monthly_evolution_chart(df)
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du graphique d'évolution: {str(e)}")
        with st.expander("🔧 Debug Évolution Mensuelle"):
            tx_cols = [col for col in df.columns if 'TX DE CAPACITE' in col]
            st.write("Colonnes TX trouvées:", tx_cols)
            st.write("Données sample:", df.head())

# ========== NAVIGATION PRINCIPALE ==========

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