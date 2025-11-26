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

def create_financeurs_visualization(df):
    """Crée la visualisation par financeurs et régions"""
    
    st.markdown("### 💰 Analyse par Régions et Financeurs")
    
    # Détecter dynamiquement les colonnes disponibles
    hts_col = None
    budget_col = None
    
    # Chercher les colonnes HTS et Budget (insensible à la casse et aux variations)
    for col in df.columns:
        col_upper = col.upper()
        if 'HTS' in col_upper and 'REALIS' in col_upper and hts_col is None:
            hts_col = col
        if 'BUDGET' in col_upper and budget_col is None:
            budget_col = col
    
    if hts_col is None:
        st.error("❌ Aucune colonne HTS REALISEES trouvée dans les données")
        with st.expander("📋 Colonnes disponibles"):
            st.write(list(df.columns))
        return
    
    # Identifier les financeurs
    financeurs_list = ['B2C - CPF', 'B2C - CPFT', "Marché de l'Alternance", 
                       'Marché des Entreprises', 'Marché Public', 'Pas de financeur']
    
    # Restructurer les données : associer chaque financeur à sa région
    data_restructured = []
    current_region = None
    
    for idx, row in df.iterrows():
        region_name = row['Régions']
        
        # Si c'est une région (pas un financeur)
        if region_name not in financeurs_list:
            # Vérifier que ce n'est pas un total
            if not pd.isna(region_name) and not any(x in str(region_name).lower() for x in ['total', 'ensemble', 'dispositif national']):
                current_region = region_name
        # Si c'est un financeur et qu'on a une région courante
        elif current_region is not None and region_name in financeurs_list:
            data_entry = {
                'Region': current_region,
                'Financeur': region_name,
                'HTS_Realisees': row[hts_col]
            }
            if budget_col:
                data_entry['Budget'] = row[budget_col]
            data_restructured.append(data_entry)
    
    # Créer un DataFrame restructuré
    df_restructured = pd.DataFrame(data_restructured)
    
    if df_restructured.empty:
        st.warning("⚠️ Aucune donnée à afficher")
        return
    
    # Remplacer les valeurs '-' par 0
    df_restructured['HTS_Realisees'] = pd.to_numeric(df_restructured['HTS_Realisees'], errors='coerce').fillna(0)
    if 'Budget' in df_restructured.columns:
        df_restructured['Budget'] = pd.to_numeric(df_restructured['Budget'], errors='coerce').fillna(0)
    
    # Options de configuration
    num_cols = 3 if 'Budget' in df_restructured.columns else 2
    cols_config = st.columns(num_cols)
    
    with cols_config[0]:
        regions_disponibles = sorted(df_restructured['Region'].unique().tolist())
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            regions_disponibles,
            default=regions_disponibles,
            key="financeurs_regions"
        )
    
    if 'Budget' in df_restructured.columns:
        with cols_config[1]:
            metric_choice = st.selectbox(
                "📊 Métrique à afficher:",
                ["HTS Réalisées", "Budget", "Les deux"],
                key="metric_choice"
            )
        
        with cols_config[2]:
            sort_order = st.selectbox(
                "📈 Ordre d'affichage:",
                ["HTS Réalisées décroissant", "Budget décroissant", "Alphabétique"],
                key="financeurs_sort"
            )
    else:
        metric_choice = "HTS Réalisées"
        with cols_config[1]:
            sort_order = st.selectbox(
                "📈 Ordre d'affichage:",
                ["HTS Réalisées décroissant", "Alphabétique"],
                key="financeurs_sort"
            )
    
    # Filtrer selon les régions sélectionnées
    df_viz = df_restructured[df_restructured['Region'].isin(regions_to_show)].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Calculer le total par région pour le tri
    agg_dict = {'HTS_Realisees': 'sum'}
    if 'Budget' in df_viz.columns:
        agg_dict['Budget'] = 'sum'
    
    region_totals = df_viz.groupby('Region').agg(agg_dict).reset_index()
    
    # Tri selon la sélection
    if sort_order == "HTS Réalisées décroissant":
        region_order = region_totals.sort_values('HTS_Realisees', ascending=False)['Region'].tolist()
    elif sort_order == "Budget décroissant" and 'Budget' in region_totals.columns:
        region_order = region_totals.sort_values('Budget', ascending=False)['Region'].tolist()
    else:  # Alphabétique
        region_order = sorted(regions_to_show)
    
    # Créer le graphique
    fig = go.Figure()
    
    # Couleurs pour les financeurs
    financeur_colors = {
        'B2C - CPF': '#3498db',
        'B2C - CPFT': '#2ecc71',
        "Marché de l'Alternance": '#9b59b6',
        'Marché des Entreprises': '#e74c3c',
        'Marché Public': '#f39c12',
        'Pas de financeur': '#95a5a6'
    }
    
    # Ajouter les barres pour chaque financeur
    if metric_choice in ["HTS Réalisées", "Les deux"]:
        for financeur in financeurs_list:
            df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
            y_values = []
            for region in region_order:
                try:
                    if region in df_financeur.index.tolist():
                        val = df_financeur.loc[region, 'HTS_Realisees']
                        if pd.notna(val):
                            y_values.append(pd.to_numeric(val, errors='coerce'))
                        else:
                            y_values.append(0)
                    else:
                        y_values.append(0)
                except:
                    y_values.append(0)
            
            fig.add_trace(go.Bar(
                name=f'{financeur} (HTS)',
                x=region_order,
                y=y_values,
                marker_color=financeur_colors.get(financeur, '#34495e'),
                text=[f"{v:,.0f}" if v > 0 else "" for v in y_values],
                textposition='inside',
                legendgroup=financeur,
                showlegend=True
            ))
    
    if 'Budget' in df_viz.columns and metric_choice in ["Budget", "Les deux"]:
        for financeur in financeurs_list:
            df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
            y_values = []
            for region in region_order:
                try:
                    if region in df_financeur.index.tolist():
                        val = df_financeur.loc[region, 'Budget']
                        if pd.notna(val):
                            y_values.append(pd.to_numeric(val, errors='coerce'))
                        else:
                            y_values.append(0)
                    else:
                        y_values.append(0)
                except:
                    y_values.append(0)
            
            # Si on affiche les deux, ajouter un pattern pour différencier
            if metric_choice == "Les deux":
                name_suffix = ' (Budget)'
            else:
                name_suffix = ''
            
            fig.add_trace(go.Bar(
                name=f'{financeur}{name_suffix}',
                x=region_order,
                y=y_values,
                marker_color=financeur_colors.get(financeur, '#34495e'),
                text=[f"{v:,.0f}" if v > 0 else "" for v in y_values],
                textposition='inside',
                legendgroup=financeur if metric_choice == "Budget" else f'{financeur}_budget',
                showlegend=True,
                opacity=0.7 if metric_choice == "Les deux" else 1.0
            ))
    
    # Configuration du graphique
    title = "💰 "
    if metric_choice == "HTS Réalisées":
        title += "HTS Réalisées par Région et Financeur"
    elif metric_choice == "Budget":
        title += "Budget par Région et Financeur"
    else:
        title += "HTS Réalisées vs Budget par Région et Financeur"
    
    fig.update_layout(
        title=title,
        xaxis_title="Régions",
        yaxis_title="Heures",
        height=700,
        barmode='group',  # Barres groupées côte à côte pour chaque financeur
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis_tickangle=-45,
        yaxis=dict(tickformat=',')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques récapitulatives
    create_financeurs_statistics(df_restructured, regions_to_show)

def create_financeurs_visualization_decembre(df):
    """Crée la visualisation par financeurs et régions pour décembre"""
    
    st.markdown("### 📅 Analyse par Période - HTS, Suites & Budget")
    
    # Détecter dynamiquement les colonnes disponibles (recherche flexible)
    total_hts_col = None
    budget_col = None
    reste_col = None
    tx_col = None
    suites_col = None
    
    for col in df.columns:
        col_upper = col.upper()
        if 'TOTAL' in col_upper and 'HTS' in col_upper and total_hts_col is None:
            total_hts_col = col
        if 'SUITES' in col_upper and 'PARCOURS' in col_upper and suites_col is None and 'TOTAL' not in col_upper:
            suites_col = col
        if 'BUDGET' in col_upper and budget_col is None:
            budget_col = col
        if 'RESTE' in col_upper and 'FAIRE' in col_upper and reste_col is None:
            reste_col = col
        if 'TX DE REALISATION' in col_upper and tx_col is None:
            tx_col = col
    
    # Si aucune colonne principale n'est trouvée, afficher une erreur
    available_cols = [c for c in [total_hts_col, suites_col, budget_col, reste_col, tx_col] if c is not None]
    
    if len(available_cols) == 0:
        st.error("❌ Aucune colonne de données reconnue dans Feuil2")
        with st.expander("📋 Colonnes disponibles"):
            st.write(list(df.columns))
        return
    
    # Identifier les financeurs
    financeurs_list = ['B2C - CPF', 'B2C - CPFT', "Marché de l'Alternance", 
                       'Marché des Entreprises', 'Marché Public', 'Pas de financeur']
    
    # Restructurer les données : associer chaque financeur à sa région
    data_restructured = []
    current_region = None
    
    for idx, row in df.iterrows():
        region_name = row['Régions']
        
        # Si c'est une région (pas un financeur)
        if region_name not in financeurs_list:
            # Vérifier que ce n'est pas un total
            if not pd.isna(region_name) and not any(x in str(region_name).lower() for x in ['total', 'ensemble', 'dispositif national']):
                current_region = region_name
        # Si c'est un financeur et qu'on a une région courante
        elif current_region is not None and region_name in financeurs_list:
            data_entry = {
                'Region': current_region,
                'Financeur': region_name
            }
            
            # Ajouter dynamiquement les colonnes disponibles
            if total_hts_col:
                data_entry['Total_HTS_Suites'] = row[total_hts_col]
            if suites_col:
                data_entry['Suites_Parcours'] = row[suites_col]
            if budget_col:
                data_entry['Budget'] = row[budget_col]
            if reste_col:
                data_entry['Reste_A_Faire'] = row[reste_col]
            if tx_col:
                data_entry['TX_Realisation'] = row[tx_col]
            
            data_restructured.append(data_entry)
    
    # Créer un DataFrame restructuré
    df_restructured = pd.DataFrame(data_restructured)
    
    if df_restructured.empty:
        st.warning("⚠️ Aucune donnée à afficher")
        return
    
    # Remplacer les valeurs '-' par 0 pour toutes les colonnes numériques
    for col in df_restructured.columns:
        if col not in ['Region', 'Financeur']:
            df_restructured[col] = pd.to_numeric(df_restructured[col], errors='coerce').fillna(0)
    
    # Options de configuration
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        regions_disponibles = sorted(df_restructured['Region'].unique().tolist())
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            regions_disponibles,
            default=regions_disponibles,  # Toutes les régions par défaut
            key="financeurs_regions_dec"
        )
    
    with col_config2:
        # Créer les options de tri dynamiquement
        sort_options = []
        if 'Total_HTS_Suites' in df_restructured.columns:
            sort_options.append("Total HTS décroissant")
        if 'Budget' in df_restructured.columns:
            sort_options.append("Budget décroissant")
        if 'Reste_A_Faire' in df_restructured.columns:
            sort_options.append("Reste à Faire décroissant")
        sort_options.append("Alphabétique")
        
        sort_order = st.selectbox(
            "📈 Ordre d'affichage:",
            sort_options,
            key="financeurs_sort_dec"
        )
    
    # Filtrer selon les régions sélectionnées
    df_viz = df_restructured[df_restructured['Region'].isin(regions_to_show)].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Calculer le total par région pour le tri (dynamique)
    agg_dict = {}
    for col in df_viz.columns:
        if col not in ['Region', 'Financeur']:
            agg_dict[col] = 'sum'
    
    region_totals = df_viz.groupby('Region').agg(agg_dict).reset_index()
    
    # Tri selon la sélection
    if sort_order == "Total HTS décroissant" and 'Total_HTS_Suites' in region_totals.columns:
        region_order = region_totals.sort_values('Total_HTS_Suites', ascending=False)['Region'].tolist()
    elif sort_order == "Budget décroissant" and 'Budget' in region_totals.columns:
        region_order = region_totals.sort_values('Budget', ascending=False)['Region'].tolist()
    elif sort_order == "Reste à Faire décroissant" and 'Reste_A_Faire' in region_totals.columns:
        region_order = region_totals.sort_values('Reste_A_Faire', ascending=False)['Region'].tolist()
    else:  # Alphabétique
        region_order = sorted(regions_to_show)
    
    # Créer le graphique
    fig = go.Figure()
    
    # Couleurs pour les financeurs - base
    financeur_colors_base = {
        'B2C - CPF': '#3498db',
        'B2C - CPFT': '#2ecc71',
        "Marché de l'Alternance": '#9b59b6',
        'Marché des Entreprises': '#e74c3c',
        'Marché Public': '#f39c12',
        'Pas de financeur': '#95a5a6'
    }
    
    # Déterminer les colonnes de données à afficher
    data_columns = [col for col in df_viz.columns if col not in ['Region', 'Financeur']]
    
    # Ajouter les données pour chaque financeur et chaque colonne
    for financeur in financeurs_list:
        df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
        
        for idx, data_col in enumerate(data_columns):
            y_values = []
            for region in region_order:
                try:
                    if region in df_financeur.index.tolist():
                        val = df_financeur.loc[region, data_col]
                        if pd.notna(val):
                            y_values.append(pd.to_numeric(val, errors='coerce'))
                        else:
                            y_values.append(0)
                    else:
                        y_values.append(0)
                except:
                    y_values.append(0)
            
            # Déterminer le nom de la métrique
            metric_name = data_col.replace('_', ' ').title()
            if data_col == 'Total_HTS_Suites':
                metric_name = 'Total HTS'
            elif data_col == 'Suites_Parcours':
                metric_name = 'Suites'
            elif data_col == 'Budget':
                metric_name = 'Budget'
            elif data_col == 'Reste_A_Faire':
                metric_name = 'Reste'
            elif data_col == 'TX_Realisation':
                metric_name = 'TX'
            
            # Ajuster la couleur selon la métrique
            base_color = financeur_colors_base.get(financeur, '#34495e')
            if idx == 0:
                color = base_color
            elif idx == 1:
                color = base_color + 'cc'  # Légèrement transparent
            else:
                color = base_color + '99'  # Plus transparent
            
            fig.add_trace(go.Bar(
                name=f'{financeur} - {metric_name}',
                x=region_order,
                y=y_values,
                marker_color=color,
                text=[f"{v:,.0f}" if v > 0 else "" for v in y_values],
                textposition='inside',
                legendgroup=financeur,
                showlegend=True
            ))
    
    # Configuration du graphique
    metrics_names = []
    for col in data_columns:
        if col == 'Total_HTS_Suites':
            metrics_names.append('Total HTS & Suites')
        elif col == 'Suites_Parcours':
            metrics_names.append('Suites de Parcours')
        elif col == 'Budget':
            metrics_names.append('Budget')
        elif col == 'Reste_A_Faire':
            metrics_names.append('Reste à Faire')
        elif col == 'TX_Realisation':
            metrics_names.append('TX Réalisation')
    
    title = f"📅 {', '.join(metrics_names)} par Région et Financeur"
    
    fig.update_layout(
        title=title,
        xaxis_title="Régions",
        yaxis_title="Heures",
        height=700,
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis_tickangle=-45,
        yaxis=dict(tickformat=',')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques récapitulatives
    create_financeurs_statistics_decembre(df_restructured, regions_to_show)

def create_financeurs_statistics_decembre(df, regions_filter=None):
    """Crée les statistiques pour l'analyse financeurs décembre"""
    
    st.markdown("### 📊 Statistiques par Région et Financeur")
    
    # Filtrer par régions si spécifié
    if regions_filter:
        df = df[df['Region'].isin(regions_filter)]
    
    # Détecter dynamiquement les colonnes disponibles
    data_columns = [col for col in df.columns if col not in ['Region', 'Financeur']]
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    col_idx = 0
    
    # Afficher les métriques pour chaque colonne disponible (max 4)
    for data_col in data_columns[:4]:
        total_val = df[data_col].sum()
        
        # Déterminer le nom de la métrique
        if data_col == 'Total_HTS_Suites':
            metric_name = "📊 Total HTS & Suites"
            help_text = "Total des HTS et suites de parcours"
        elif data_col == 'Suites_Parcours':
            metric_name = "📋 Suites Parcours"
            help_text = "Total des suites de parcours"
        elif data_col == 'Budget':
            metric_name = "📈 Budget Total"
            help_text = "Budget total"
        elif data_col == 'Reste_A_Faire':
            metric_name = "📋 Reste à Faire Total"
            help_text = "Total du reste à faire"
        elif data_col == 'TX_Realisation':
            metric_name = "🎯 TX Réalisation Moyen"
            help_text = "Taux de réalisation moyen"
            total_val = df[data_col].mean() * 100  # Pour les TX, afficher la moyenne en %
        else:
            metric_name = f"📊 {data_col}"
            help_text = f"Total {data_col}"
        
        with [col_stats1, col_stats2, col_stats3, col_stats4][col_idx]:
            if 'TX' in data_col:
                st.metric(metric_name, f"{total_val:.1f}%", help=help_text)
            else:
                st.metric(metric_name, f"{total_val:,.0f}", help=help_text)
        
        col_idx += 1
        if col_idx >= 4:
            break
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région et Financeur"):
        display_data = df.copy()
        
        columns_to_show = ['Region', 'Financeur']
        column_config = {
            'Region': 'Région',
            'Financeur': 'Financeur'
        }
        
        # Ajouter dynamiquement les colonnes disponibles
        for col in data_columns:
            columns_to_show.append(col)
            
            if col == 'Total_HTS_Suites':
                column_config[col] = st.column_config.NumberColumn('Total HTS & Suites', format="%.0f")
            elif col == 'Suites_Parcours':
                column_config[col] = st.column_config.NumberColumn('Suites Parcours', format="%.0f")
            elif col == 'Budget':
                column_config[col] = st.column_config.NumberColumn('Budget', format="%.0f")
            elif col == 'Reste_A_Faire':
                column_config[col] = st.column_config.NumberColumn('Reste à Faire', format="%.0f")
            elif col == 'TX_Realisation':
                column_config[col] = st.column_config.NumberColumn('TX Réalisation', format="%.1f%%")
            else:
                column_config[col] = st.column_config.NumberColumn(col, format="%.0f")
        
        # Calculer TX et Écart si possible
        if 'Total_HTS_Suites' in df.columns and 'Budget' in df.columns:
            display_data['TX Réalisation (%)'] = (display_data['Total_HTS_Suites'] / 
                                                   display_data['Budget'] * 100).fillna(0)
            display_data['Écart Budget'] = display_data['Total_HTS_Suites'] - display_data['Budget']
            
            columns_to_show.extend(['TX Réalisation (%)', 'Écart Budget'])
            column_config['TX Réalisation (%)'] = st.column_config.NumberColumn('TX Réalisation', format="%.1f%%")
            column_config['Écart Budget'] = st.column_config.NumberColumn('Écart Budget', format="%.0f", help="Total HTS - Budget")
        
        display_data_filtered = display_data[columns_to_show].copy()
        
        # Tri par la première colonne de données disponible
        if len(data_columns) > 0:
            display_data_filtered = display_data_filtered.sort_values(['Region', data_columns[0]], ascending=[True, False])
        else:
            display_data_filtered = display_data_filtered.sort_values('Region')
        
        st.dataframe(
            display_data_filtered,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )

def create_financeurs_statistics(df, regions_filter=None):
    """Crée les statistiques pour l'analyse financeurs"""
    
    st.markdown("### 📊 Statistiques par Région et Financeur")
    
    # Filtrer par régions si spécifié
    if regions_filter:
        df = df[df['Region'].isin(regions_filter)]
    
    # Détecter les colonnes disponibles
    has_hts = 'HTS_Realisees' in df.columns
    has_budget = 'Budget' in df.columns
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    if has_hts:
        total_hts = df['HTS_Realisees'].sum()
        
        with col_stats1:
            st.metric(
                "📊 Total HTS Réalisées",
                f"{total_hts:,.0f}",
                help="Total des heures réalisées"
            )
    
    if has_budget:
        total_budget = df['Budget'].sum()
        
        with col_stats2:
            st.metric(
                "📈 Budget Total",
                f"{total_budget:,.0f}",
                help="Budget total"
            )
    
    if has_hts and has_budget:
        tx_realisation = (total_hts / total_budget * 100) if total_budget > 0 else 0
        
        with col_stats3:
            st.metric(
                "🎯 TX Réalisation",
                f"{tx_realisation:.1f}%",
                help="Taux de réalisation global"
            )
    
    if has_hts:
        # Meilleur financeur
        financeur_totals = df.groupby('Financeur')['HTS_Realisees'].sum()
        if not financeur_totals.empty:
            best_financeur = financeur_totals.idxmax()
            best_financeur_val = financeur_totals.max()
        else:
            best_financeur = "N/A"
            best_financeur_val = 0
        
        with col_stats4:
            st.metric(
                "🏆 Meilleur Financeur",
                best_financeur[:15] + "..." if len(best_financeur) > 15 else best_financeur,
                f"{best_financeur_val:,.0f}h"
            )
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région et Financeur"):
        display_data = df.copy()
        
        columns_to_show = ['Region', 'Financeur']
        column_config = {
            'Region': 'Région',
            'Financeur': 'Financeur'
        }
        
        if 'HTS_Realisees' in df.columns:
            columns_to_show.append('HTS_Realisees')
            column_config['HTS_Realisees'] = st.column_config.NumberColumn(
                'HTS Réalisées',
                format="%.0f"
            )
        
        if 'Budget' in df.columns:
            columns_to_show.append('Budget')
            column_config['Budget'] = st.column_config.NumberColumn(
                'Budget',
                format="%.0f"
            )
        
        if 'HTS_Realisees' in df.columns and 'Budget' in df.columns:
            display_data['TX Réalisation (%)'] = (display_data['HTS_Realisees'] / 
                                                   display_data['Budget'] * 100).fillna(0)
            display_data['Écart'] = display_data['HTS_Realisees'] - display_data['Budget']
            
            columns_to_show.extend(['TX Réalisation (%)', 'Écart'])
            column_config['TX Réalisation (%)'] = st.column_config.NumberColumn(
                'TX Réalisation',
                format="%.1f%%"
            )
            column_config['Écart'] = st.column_config.NumberColumn(
                'Écart',
                format="%.0f",
                help="HTS Réalisées - Budget"
            )
        
        display_data_filtered = display_data[columns_to_show].copy()
        if 'HTS_Realisees' in columns_to_show:
            display_data_filtered = display_data_filtered.sort_values(['Region', 'HTS_Realisees'], ascending=[True, False])
        else:
            display_data_filtered = display_data_filtered.sort_values('Region')
        
        st.dataframe(
            display_data_filtered,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )

def create_landing_visualization_with_financeurs(df):
    """Crée la visualisation d'atterrissage par financeurs avec barres multiples par période"""
    
    st.markdown("### 🎯 Analyse d'Atterrissage par Région et Financeurs")
    
    # Détecter les colonnes TX de réalisation disponibles
    tx_columns = [col for col in df.columns if 'TX DE REALISATION' in col.upper()]
    reste_a_faire_col = next((col for col in df.columns if 'RESTE A FAIRE' in col.upper()), None)
    
    if len(tx_columns) == 0:
        st.error("❌ Aucune colonne TX DE REALISATION trouvée")
        return
    
    # Identifier les financeurs
    financeurs_list = ['B2C - CPF', 'B2C - CPFT', "Marché de l'Alternance", 
                       'Marché des Entreprises', 'Marché Public', 'Pas de financeur']
    
    # Restructurer les données : associer chaque financeur à sa région
    data_restructured = []
    current_region = None
    
    for idx, row in df.iterrows():
        region_name = row['Régions']
        
        # Si c'est une région (pas un financeur)
        if region_name not in financeurs_list:
            if not pd.isna(region_name) and not any(x in str(region_name).lower() for x in ['total', 'ensemble', 'dispositif national']):
                current_region = region_name
        # Si c'est un financeur et qu'on a une région courante
        elif current_region is not None and region_name in financeurs_list:
            data_entry = {
                'Region': current_region,
                'Financeur': region_name
            }
            
            # Ajouter toutes les colonnes TX
            for tx_col in tx_columns:
                data_entry[tx_col] = row[tx_col]
            
            # Ajouter Reste à faire si disponible
            if reste_a_faire_col:
                data_entry[reste_a_faire_col] = row[reste_a_faire_col]
            
            data_restructured.append(data_entry)
    
    # Créer un DataFrame restructuré
    df_restructured = pd.DataFrame(data_restructured)
    
    if df_restructured.empty:
        st.warning("⚠️ Aucune donnée à afficher")
        return
    
    # Remplacer les valeurs '-' par 0 pour toutes les colonnes numériques
    for col in df_restructured.columns:
        if col not in ['Region', 'Financeur']:
            df_restructured[col] = pd.to_numeric(df_restructured[col], errors='coerce').fillna(0)
    
    # Créer les options de tri dynamiquement
    sort_options = []
    for tx_col in tx_columns:
        period = tx_col.replace('TX DE REALISATION', '').strip().replace('/', '').strip()
        if period:
            sort_options.append(f"TX {period} décroissant")
        else:
            sort_options.append("TX décroissant")
    sort_options.append("Alphabétique")
    if reste_a_faire_col:
        sort_options.append("Reste à faire décroissant")
    
    # Options de configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        regions_disponibles = sorted(df_restructured['Region'].unique().tolist())
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            regions_disponibles,
            default=regions_disponibles,
            key="landing_financeurs_regions"
        )
    
    with col_config2:
        financeurs_disponibles = sorted(df_restructured['Financeur'].unique().tolist())
        financeurs_to_show = st.multiselect(
            "💰 Financeurs à afficher:",
            financeurs_disponibles,
            default=financeurs_disponibles,
            key="landing_financeurs_filter"
        )
    
    with col_config3:
        sort_order = st.selectbox(
            "📈 Ordre d'affichage:",
            sort_options,
            key="landing_financeurs_sort"
        )
    
    # Filtrer selon les sélections
    df_viz = df_restructured[
        (df_restructured['Region'].isin(regions_to_show)) & 
        (df_restructured['Financeur'].isin(financeurs_to_show))
    ].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Calculer les totaux par région pour le tri
    agg_dict = {}
    for tx_col in tx_columns:
        agg_dict[tx_col] = 'mean'
    if reste_a_faire_col:
        agg_dict[reste_a_faire_col] = 'sum'
    
    region_totals = df_viz.groupby('Region').agg(agg_dict).reset_index()
    
    # Tri selon la sélection
    if sort_order == "Alphabétique":
        region_order = sorted(regions_to_show)
    elif sort_order == "Reste à faire décroissant" and reste_a_faire_col:
        region_order = region_totals.sort_values(reste_a_faire_col, ascending=False)['Region'].tolist()
    else:
        # Trouver la colonne TX correspondante
        for tx_col in tx_columns:
            period = tx_col.replace('TX DE REALISATION', '').strip().replace('/', '').strip()
            if (period and f"TX {period} décroissant" == sort_order) or (not period and "TX décroissant" == sort_order):
                region_order = region_totals.sort_values(tx_col, ascending=False)['Region'].tolist()
                break
        else:
            region_order = sorted(regions_to_show)
    
    # Créer le graphique avec barres multiples
    fig = go.Figure()
    
    # Couleurs pour les financeurs
    financeur_colors = {
        'B2C - CPF': '#3498db',
        'B2C - CPFT': '#2ecc71',
        "Marché de l'Alternance": '#9b59b6',
        'Marché des Entreprises': '#e74c3c',
        'Marché Public': '#f39c12',
        'Pas de financeur': '#95a5a6'
    }
    
    # Couleurs pour les différentes périodes (pour reste à faire)
    colors_periods = ['#e74c3c', '#2ecc71', '#f39c12']
    
    # Ajouter les barres TX avec axe Y principal (pourcentages) pour chaque colonne et financeur
    for idx, tx_col in enumerate(tx_columns):
        period = tx_col.replace('TX DE REALISATION', '').strip().replace('/', '').strip()
        period_name = period if period else "TX"
        
        for financeur in financeurs_to_show:
            df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
            y_values = []
            
            for region in region_order:
                try:
                    if region in df_financeur.index.tolist():
                        val = df_financeur.loc[region, tx_col]
                        if pd.notna(val):
                            y_values.append(float(val) * 100)
                        else:
                            y_values.append(0)
                    else:
                        y_values.append(0)
                except:
                    y_values.append(0)
            
            # Nom de la trace avec financeur et période
            trace_name = f'{financeur} - {period_name}'
            
            fig.add_trace(go.Bar(
                name=trace_name,
                x=region_order,
                y=y_values,
                marker_color=financeur_colors.get(financeur, '#34495e'),
                text=[f"{v:.1f}%" if v > 0 else "" for v in y_values],
                textposition='inside',
                yaxis='y',
                legendgroup=f'{financeur}_{period_name}',
                showlegend=True,
                opacity=0.9 - (idx * 0.1),
                offsetgroup=idx
            ))
    
    # Reste à faire avec axe Y secondaire (valeurs absolues)
    # Surplus sur axe Y principal pour dépasser les 100%
    if reste_a_faire_col:
        for financeur in financeurs_to_show:
            df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
            y_values_reste = []
            y_values_surplus = []
            
            for region in region_order:
                try:
                    if region in df_financeur.index.tolist():
                        val = df_financeur.loc[region, reste_a_faire_col]
                        tx_val = 0
                        # Récupérer le TX correspondant
                        if len(tx_columns) > 0:
                            tx_val = df_financeur.loc[region, tx_columns[-1]]  # Dernier TX
                            if pd.notna(tx_val):
                                tx_val = float(tx_val) * 100
                            else:
                                tx_val = 0
                        
                        if pd.notna(val):
                            val_float = float(val)
                            # Si positif = reste à faire, si négatif = surplus
                            if val_float > 0:
                                y_values_reste.append(val_float)
                                y_values_surplus.append(0)
                            else:
                                y_values_reste.append(0)
                                # Pour le surplus qui dépasse 100%, on affiche TX + extension
                                surplus_percent = min(abs(val_float) / 10000 * 10, 20)  # Jusqu'à 20% au-dessus
                                y_values_surplus.append(tx_val + surplus_percent)
                        else:
                            y_values_reste.append(0)
                            y_values_surplus.append(0)
                    else:
                        y_values_reste.append(0)
                        y_values_surplus.append(0)
                except:
                    y_values_reste.append(0)
                    y_values_surplus.append(0)
            
            # Couleur différente pour Reste à Faire
            base_color = financeur_colors.get(financeur, '#34495e')
            reste_color = base_color.replace('#3498db', '#1f5f8b').replace('#2ecc71', '#1e8449') \
                                     .replace('#9b59b6', '#6c3483').replace('#e74c3c', '#a93226') \
                                     .replace('#f39c12', '#b9770e').replace('#95a5a6', '#626567')
            
            # Ajouter la barre Reste à Faire (sur axe Y2)
            fig.add_trace(go.Bar(
                name=f'{financeur} - Reste à Faire',
                x=region_order,
                y=y_values_reste,
                marker_color=reste_color,
                text=[f"{v:,.0f}" if v > 0 else "" for v in y_values_reste],
                textposition='outside',
                yaxis='y2',
                legendgroup=f'{financeur}_reste',
                showlegend=True,
                opacity=1.0,
                offsetgroup=len(tx_columns)
            ))
            
            # Ajouter la barre Surplus qui dépasse les 100%
            if any(v > 0 for v in y_values_surplus):
                surplus_color = '#1abc9c'  # Turquoise/cyan
                
                fig.add_trace(go.Bar(
                    name=f'{financeur} - Surplus (dépassement)',
                    x=region_order,
                    y=y_values_surplus,
                    marker_color=surplus_color,
                    text=[f"+{abs(df_financeur.loc[region_order[i], reste_a_faire_col]):,.0f}" if y_values_surplus[i] > 0 and region_order[i] in df_financeur.index else "" for i in range(len(region_order))],
                    textposition='outside',
                    yaxis='y',
                    legendgroup=f'{financeur}_surplus',
                    showlegend=True,
                    opacity=0.8,
                    offsetgroup=len(tx_columns) + 1
                ))
    
    # Configuration du graphique avec double axe Y
    fig.update_layout(
        title={
            'text': "🎯 Analyse d'Atterrissage : TX Réalisation et Reste à Faire par Région et Financeur",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
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
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="right",
            x=1
        ),
        xaxis_tickangle=-45,
        margin=dict(t=150)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques récapitulatives
    create_landing_statistics_financeurs(df_viz, tx_columns, reste_a_faire_col)

def create_landing_statistics_financeurs(df, tx_columns, reste_a_faire_col):
    """Crée les statistiques récapitulatives pour l'atterrissage par financeurs"""
    
    st.markdown("### 📊 Statistiques d'Atterrissage par Financeurs")
    
    # Créer les colonnes pour les métriques
    num_metrics = min(len(tx_columns) + (1 if reste_a_faire_col else 0) + 1, 4)
    cols = st.columns(num_metrics)
    
    col_idx = 0
    
    # Afficher les statistiques pour chaque colonne TX
    for idx, tx_col in enumerate(tx_columns[:2]):
        period = tx_col.replace('TX DE REALISATION', '').strip().replace('/', '').strip()
        period_name = period if period else "TX"
        
        avg_tx = df[tx_col].mean() * 100
        max_idx = df[tx_col].idxmax()
        best_region = f"{df.loc[max_idx, 'Region']} ({df.loc[max_idx, 'Financeur']})"
        best_tx = df.loc[max_idx, tx_col] * 100
        
        with cols[col_idx]:
            st.metric(
                label=f"🎯 TX Moyen {period_name}",
                value=f"{avg_tx:.1f}%",
                help=f"Meilleur: {best_region} avec {best_tx:.1f}%"
            )
        col_idx += 1
        if col_idx >= num_metrics:
            break
    
    # Reste à faire moyen
    if reste_a_faire_col and col_idx < num_metrics:
        avg_reste = df[reste_a_faire_col].mean()
        with cols[col_idx]:
            st.metric(
                label="📋 Reste à Faire Moyen",
                value=f"{avg_reste:,.0f}",
                help="Moyenne du reste à faire par région/financeur"
            )
        col_idx += 1
    
    # Meilleur financeur (basé sur la première colonne TX)
    if col_idx < num_metrics and len(tx_columns) > 0:
        financeur_avg = df.groupby('Financeur')[tx_columns[0]].mean()
        best_financeur = financeur_avg.idxmax()
        best_financeur_tx = financeur_avg.max() * 100
        
        with cols[col_idx]:
            st.metric(
                label="🏆 Meilleur Financeur",
                value=best_financeur,
                help=f"TX moyen: {best_financeur_tx:.1f}%"
            )
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région et Financeur"):
        display_df = df[['Region', 'Financeur'] + tx_columns + ([reste_a_faire_col] if reste_a_faire_col else [])]
        
        # Formater les colonnes TX en pourcentages
        for tx_col in tx_columns:
            display_df[tx_col] = display_df[tx_col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
        
        if reste_a_faire_col:
            display_df[reste_a_faire_col] = display_df[reste_a_faire_col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

def create_landing_visualization(df):
    """Crée la visualisation d'atterrissage avec barres multiples par période"""
    
    st.markdown("### 🎯 Analyse d'Atterrissage par Région")
    
    # Détecter les colonnes TX de réalisation disponibles
    tx_columns = [col for col in df.columns if 'TX DE REALISATION' in col.upper()]
    reste_a_faire_col = next((col for col in df.columns if 'RESTE A FAIRE' in col.upper()), None)
    
    if len(tx_columns) == 0:
        st.error("❌ Aucune colonne TX DE REALISATION trouvée dans les données")
        return
    
    # Filtrer les données pour enlever les totaux
    df_filtered = df[~df['REGION'].str.contains('total|Total|TOTAL|ENSEMBLE', case=False, na=False)].copy()
    df_filtered = df_filtered.dropna(subset=['REGION'])
    
    # Créer les options de tri dynamiquement
    sort_options = []
    for tx_col in tx_columns:
        mois = tx_col.replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
        if not mois:
            mois = "Global"
        sort_options.append(f"TX {mois.title()} décroissant")
    sort_options.extend(["Alphabétique"])
    if reste_a_faire_col:
        sort_options.append("Reste à faire décroissant")
    
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
            sort_options,
            key="landing_sort"
        )
    
    # Filtrer selon les sélections
    df_viz = df_filtered[df_filtered['REGION'].isin(regions_to_show)].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Tri selon la sélection
    if sort_order == "Alphabétique":
        df_viz = df_viz.sort_values('REGION', ascending=True)
    elif sort_order == "Reste à faire décroissant" and reste_a_faire_col:
        df_viz = df_viz.sort_values(reste_a_faire_col, ascending=False)
    else:
        # Tri par TX - trouver la colonne correspondante
        for tx_col in tx_columns:
            mois = tx_col.replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
            if not mois:
                mois = "Global"
            if f"TX {mois.title()} décroissant" == sort_order:
                df_viz = df_viz.sort_values(tx_col, ascending=False)
                break
    
    # Créer le graphique avec barres multiples
    fig = go.Figure()
    
    regions = df_viz['REGION'].tolist()
    
    # Couleurs pour les différentes périodes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b', '#27ae60', '#8e44ad']
    
    # Ajouter les barres TX avec axe Y principal (pourcentages) pour chaque colonne détectée
    for idx, tx_col in enumerate(tx_columns):
        mois = tx_col.replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
        if not mois:
            mois = "Global"
        
        tx_values = (df_viz[tx_col] * 100).tolist()
        
        fig.add_trace(go.Bar(
            x=regions,
            y=tx_values,
            name=f'TX Réalisation {mois.title()} (%)',
            marker_color=colors[idx % len(colors)],
            text=[f"{val:.1f}%" for val in tx_values],
            textposition='outside',
            yaxis='y',
            offsetgroup=idx
        ))
    
    # Reste à faire avec axe Y secondaire (valeurs brutes) si disponible
    if reste_a_faire_col:
        reste_a_faire = df_viz[reste_a_faire_col].tolist()
        
        # Créer deux listes : une pour les valeurs positives (reste à faire) et une pour les négatives (surplus)
        reste_positif = [val if val > 0 else 0 for val in reste_a_faire]
        surplus_absolu = [abs(val) if val < 0 else 0 for val in reste_a_faire]
        
        # Ajouter les barres pour le reste à faire (orange)
        fig.add_trace(go.Bar(
            x=regions,
            y=reste_positif,
            name='Reste à Faire (valeurs)',
            marker_color='#f39c12',  # Orange
            text=[f"{val:,.0f}" if val > 0 else "" for val in reste_a_faire],
            textposition='outside',
            yaxis='y2',
            offsetgroup=len(tx_columns)
        ))
        
        # Ajouter les barres pour le surplus (vert)
        fig.add_trace(go.Bar(
            x=regions,
            y=surplus_absolu,
            name='Surplus (dépassement)',
            marker_color='#27ae60',  # Vert
            text=[f"+{abs(val):,.0f}" if val < 0 else "" for val in reste_a_faire],
            textposition='outside',
            yaxis='y2',
            offsetgroup=len(tx_columns)
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
    
    # Détecter les colonnes TX de réalisation disponibles
    tx_columns = [col for col in df.columns if 'TX DE REALISATION' in col.upper()]
    reste_a_faire_col = next((col for col in df.columns if 'RESTE A FAIRE' in col.upper()), None)
    
    if len(tx_columns) == 0:
        st.warning("⚠️ Aucune colonne TX DE REALISATION disponible pour les statistiques")
        return
    
    # Créer les colonnes pour les métriques (maximum 4)
    num_metrics = min(len(tx_columns) + (1 if reste_a_faire_col else 0) + 1, 4)
    cols = st.columns(num_metrics)
    
    col_idx = 0
    
    # Afficher les statistiques pour chaque colonne TX
    for idx, tx_col in enumerate(tx_columns[:2]):  # Limiter à 2 premières colonnes TX
        mois = tx_col.replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
        if not mois:
            mois = "Global"
        
        avg_tx = df[tx_col].mean() * 100
        
        with cols[col_idx]:
            st.metric(
                f"📊 TX Moyen {mois.title()}",
                f"{avg_tx:.1f}%",
                help=f"Taux de réalisation moyen à fin {mois.lower()}"
            )
        col_idx += 1
    
    # Meilleure région (basée sur la première colonne TX)
    if col_idx < num_metrics:
        best_idx = df[tx_columns[0]].idxmax()
        best_region = df.loc[best_idx, 'REGION']
        best_val = df.loc[best_idx, tx_columns[0]] * 100
        
        mois_ref = tx_columns[0].replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
        if not mois_ref:
            mois_ref = "Global"
        
        with cols[col_idx]:
            st.metric(
                f"🏆 Meilleure Région",
                best_region[:15] + "..." if len(best_region) > 15 else best_region,
                f"{best_val:.1f}%"
            )
        col_idx += 1
    
    # Reste à faire moyen
    if reste_a_faire_col and col_idx < num_metrics:
        avg_reste_brut = df[reste_a_faire_col].mean()
        
        with cols[col_idx]:
            st.metric(
                "📋 Reste à Faire Moyen",
                f"{avg_reste_brut:,.0f}",
                help="Valeur moyenne du reste à faire (valeurs brutes)"
            )
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région"):
        # Préparer les données pour l'affichage
        display_data = df.copy()
        
        columns_to_show = ['REGION']
        column_config = {'REGION': 'Région'}
        
        # Ajouter dynamiquement les colonnes TX
        for tx_col in tx_columns:
            mois = tx_col.replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
            if not mois:
                mois = "Global"
            
            col_name = f'TX {mois.title()} (%)'
            display_data[col_name] = display_data[tx_col] * 100
            columns_to_show.append(col_name)
            column_config[col_name] = st.column_config.NumberColumn(
                col_name,
                format="%.1f%%"
            )
        
        # Ajouter l'écart si au moins 2 colonnes TX
        if len(tx_columns) >= 2:
            mois1 = tx_columns[0].replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip() or "Premier"
            mois2 = tx_columns[-1].replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip() or "Dernier"
            
            ecart_col = f'Écart {mois1[:3]}-{mois2[:3]}'
            display_data[ecart_col] = (display_data[tx_columns[-1]] - display_data[tx_columns[0]]) * 100
            columns_to_show.append(ecart_col)
            column_config[ecart_col] = st.column_config.NumberColumn(
                ecart_col,
                format="%.1f%%",
                help=f"Différence entre {mois2.title()} et {mois1.title()}"
            )
        
        # Ajouter le reste à faire si disponible
        if reste_a_faire_col:
            display_data['Reste à Faire (valeurs)'] = display_data[reste_a_faire_col]
            columns_to_show.append('Reste à Faire (valeurs)')
            column_config['Reste à Faire (valeurs)'] = st.column_config.NumberColumn(
                'Reste à Faire (valeurs)',
                format="%.0f",
                help="Valeurs brutes du reste à faire"
            )
        
        display_data_filtered = display_data[columns_to_show].copy()
        display_data_filtered = display_data_filtered.sort_values(columns_to_show[1], ascending=False)
        
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
        - Au moins une colonne 'TX DE REALISATION A FIN [MOIS]' avec les taux (0-1)
          - Exemples: 'TX DE REALISATION A FIN SEPTEMBRE', 'TX DE REALISATION A FIN DECEMBRE', etc.
        - Une colonne 'RESTE A FAIRE / NOUVELLES ENTREES' avec les valeurs numériques (optionnelle)
        - **Format des taux** : Décimaux entre 0 et 1 (ex: 0.85 pour 85%)
        - **Tous les mois sont acceptés** : janvier, février, mars, avril, mai, juin, juillet, août, septembre, octobre, novembre, décembre
        """)
        return
    
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible dans le fichier")
        return
    
    # Validation du format des données
    st.success(f"✅ Fichier chargé avec succès : {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Détection dynamique des colonnes de TX de réalisation (tous les mois)
    tx_columns = [col for col in df.columns if 'TX DE REALISATION' in col.upper()]
    reste_a_faire_col = next((col for col in df.columns if 'RESTE A FAIRE' in col.upper()), None)
    
    # Vérifications de compatibilité
    warnings = []
    if 'REGION' not in df.columns:
        warnings.append("❌ Colonne 'REGION' ou 'Régions' non trouvée - impossible de créer les analyses")
    
    if len(tx_columns) == 0:
        warnings.append("❌ Aucune colonne 'TX DE REALISATION' trouvée - impossible de créer les analyses")
    
    if not reste_a_faire_col:
        warnings.append("⚠️ Colonne 'RESTE A FAIRE' non trouvée - certaines statistiques seront limitées")
        
    if warnings:
        for warning in warnings:
            if "❌" in warning:
                st.error(warning)
            else:
                st.warning(warning)
        
        if any("❌" in w for w in warnings):
            st.info("💡 Votre fichier doit contenir au minimum une colonne REGION et une colonne TX DE REALISATION")
            
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
        
        # Statistiques rapides - sécurisé avec vérification dynamique
        if len(tx_columns) > 0:
            st.subheader("📊 Aperçu Rapide")
            
            # Afficher les statistiques pour chaque colonne TX trouvée
            for tx_col in tx_columns:
                # Extraire le nom du mois de la colonne
                mois = tx_col.replace('TX DE REALISATION', '').replace('A FIN', '').replace('/AU BUDGET A FIN', '').strip()
                if not mois:
                    mois = "Global"
                    
                avg_tx = df[tx_col].mean() * 100
                st.metric(f"TX Moyen {mois.title()}", f"{avg_tx:.1f}%")
            
            # Si au moins 2 colonnes, afficher l'évolution
            if len(tx_columns) >= 2:
                first_tx = df[tx_columns[0]].mean() * 100
                last_tx = df[tx_columns[-1]].mean() * 100
                st.metric("📈 Évolution", f"+{last_tx - first_tx:.1f}%")
    
    # Génération de l'analyse
    st.markdown("---")
    
    # Tabs pour différentes vues
    tab1, tab2 = st.tabs(["📊 Analyse TX Réalisation", "🎯 Analyse par Financeurs (Feuil2)"])
    
    with tab1:
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
    
    with tab2:
        # Charger la Feuil2 pour l'analyse d'atterrissage par financeurs
        try:
            if import_method == "Upload d'un nouveau fichier":
                df_financeurs = pd.read_excel(uploaded_file, sheet_name='Feuil2')
            else:
                df_financeurs = pd.read_excel(custom_path, sheet_name='Feuil2')
            
            # Graphique d'atterrissage avec filtres par financeurs
            create_landing_visualization_with_financeurs(df_financeurs)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de l'analyse par financeurs: {str(e)}")
            st.info("💡 Assurez-vous que votre fichier contient une feuille 'Feuil2' avec les colonnes requises")
            
            # Debug info
            with st.expander("🔧 Informations de Debug"):
                st.write("Erreur:", str(e))