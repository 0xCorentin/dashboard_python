# -*- coding: utf-8 -*-
"""
Analyseur Optimisation des Plateaux
Module dédié à l'analyse des performances de production par région
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Import des fonctions utilitaires
from utils import (
    load_weekly_data, aggregate_data_smart, calculate_performance_insights
)

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

def show_weekly_analysis():
    """Page de l'Optimisation des plateaux"""
    
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