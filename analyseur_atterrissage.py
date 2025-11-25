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
    
    # Identifier les financeurs
    financeurs_list = ['B2C - CPF', 'B2C - CPFT', "Marché de l'Alternance", 
                       'Marché des Entreprises', 'Marché Public']
    
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
            data_restructured.append({
                'Region': current_region,
                'Financeur': region_name,
                'HTS_Realisees': row['HTS REALISEES TOTALES (AVEC PAE)'],
                'Budget_Septembre': row['BUDGET A FIN SEPTEMBRE']
            })
    
    # Créer un DataFrame restructuré
    df_restructured = pd.DataFrame(data_restructured)
    
    if df_restructured.empty:
        st.warning("⚠️ Aucune donnée à afficher")
        return
    
    # Remplacer les valeurs '-' par 0
    df_restructured['HTS_Realisees'] = pd.to_numeric(df_restructured['HTS_Realisees'], errors='coerce').fillna(0)
    df_restructured['Budget_Septembre'] = pd.to_numeric(df_restructured['Budget_Septembre'], errors='coerce').fillna(0)
    
    # Options de configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        regions_disponibles = sorted(df_restructured['Region'].unique().tolist())
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            regions_disponibles,
            default=regions_disponibles,  # Toutes les régions par défaut
            key="financeurs_regions"
        )
    
    with col_config2:
        metric_choice = st.selectbox(
            "📊 Métrique à afficher:",
            ["HTS Réalisées", "Budget Septembre", "Les deux"],
            key="metric_choice"
        )
    
    with col_config3:
        sort_order = st.selectbox(
            "� Ordre d'affichage:",
            ["HTS Réalisées décroissant", "Budget Septembre décroissant", "Alphabétique"],
            key="financeurs_sort"
        )
    
    # Filtrer selon les régions sélectionnées
    df_viz = df_restructured[df_restructured['Region'].isin(regions_to_show)].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Calculer le total par région pour le tri
    region_totals = df_viz.groupby('Region').agg({
        'HTS_Realisees': 'sum',
        'Budget_Septembre': 'sum'
    }).reset_index()
    
    # Tri selon la sélection
    if sort_order == "HTS Réalisées décroissant":
        region_order = region_totals.sort_values('HTS_Realisees', ascending=False)['Region'].tolist()
    elif sort_order == "Budget Septembre décroissant":
        region_order = region_totals.sort_values('Budget_Septembre', ascending=False)['Region'].tolist()
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
        'Marché Public': '#f39c12'
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
                        # Convertir en nombre Python natif
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
    
    if metric_choice in ["Budget Septembre", "Les deux"]:
        for financeur in financeurs_list:
            df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
            y_values = []
            for region in region_order:
                try:
                    if region in df_financeur.index.tolist():
                        val = df_financeur.loc[region, 'Budget_Septembre']
                        # Convertir en nombre Python natif
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
                legendgroup=financeur if metric_choice == "Budget Septembre" else f'{financeur}_budget',
                showlegend=True,
                opacity=0.7 if metric_choice == "Les deux" else 1.0
            ))
    
    # Configuration du graphique
    title = "💰 "
    if metric_choice == "HTS Réalisées":
        title += "HTS Réalisées par Région et Financeur"
    elif metric_choice == "Budget Septembre":
        title += "Budget Septembre par Région et Financeur"
    else:
        title += "HTS Réalisées vs Budget Septembre par Région et Financeur"
    
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
    
    st.markdown("### 📅 Analyse Décembre - Total HTS & Suites de Parcours")
    
    # Identifier les financeurs
    financeurs_list = ['B2C - CPF', 'B2C - CPFT', "Marché de l'Alternance", 
                       'Marché des Entreprises', 'Marché Public']
    
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
            data_restructured.append({
                'Region': current_region,
                'Financeur': region_name,
                'Total_HTS_Suites': row['TOTAL HTS & SUITES DE PARCOURS'],
                'Budget_Decembre': row['BUDGET A FIN DECEMBRE'],
                'Reste_A_Faire': row['RESTE A FAIRE / NOUVELLES ENTREES']
            })
    
    # Créer un DataFrame restructuré
    df_restructured = pd.DataFrame(data_restructured)
    
    if df_restructured.empty:
        st.warning("⚠️ Aucune donnée à afficher")
        return
    
    # Remplacer les valeurs '-' par 0
    df_restructured['Total_HTS_Suites'] = pd.to_numeric(df_restructured['Total_HTS_Suites'], errors='coerce').fillna(0)
    df_restructured['Budget_Decembre'] = pd.to_numeric(df_restructured['Budget_Decembre'], errors='coerce').fillna(0)
    df_restructured['Reste_A_Faire'] = pd.to_numeric(df_restructured['Reste_A_Faire'], errors='coerce').fillna(0)
    
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
        sort_order = st.selectbox(
            "📈 Ordre d'affichage:",
            ["Total HTS décroissant", "Budget Décembre décroissant", "Reste à Faire décroissant", "Alphabétique"],
            key="financeurs_sort_dec"
        )
    
    # Filtrer selon les régions sélectionnées
    df_viz = df_restructured[df_restructured['Region'].isin(regions_to_show)].copy()
    
    if df_viz.empty:
        st.warning("⚠️ Aucune donnée à afficher avec les filtres sélectionnés")
        return
    
    # Calculer le total par région pour le tri
    region_totals = df_viz.groupby('Region').agg({
        'Total_HTS_Suites': 'sum',
        'Budget_Decembre': 'sum',
        'Reste_A_Faire': 'sum'
    }).reset_index()
    
    # Tri selon la sélection
    if sort_order == "Total HTS décroissant":
        region_order = region_totals.sort_values('Total_HTS_Suites', ascending=False)['Region'].tolist()
    elif sort_order == "Budget Décembre décroissant":
        region_order = region_totals.sort_values('Budget_Decembre', ascending=False)['Region'].tolist()
    elif sort_order == "Reste à Faire décroissant":
        region_order = region_totals.sort_values('Reste_A_Faire', ascending=False)['Region'].tolist()
    else:  # Alphabétique
        region_order = sorted(regions_to_show)
    
    # Créer le graphique
    fig = go.Figure()
    
    # Couleurs pour les financeurs avec 3 nuances par financeur (pour les 3 métriques)
    financeur_colors = {
        'B2C - CPF': {
            'Total_HTS': '#3498db',      # Bleu
            'Budget': '#5dade2',          # Bleu clair
            'Reste': '#85c1e9'            # Bleu très clair
        },
        'B2C - CPFT': {
            'Total_HTS': '#2ecc71',      # Vert
            'Budget': '#58d68d',          # Vert clair
            'Reste': '#82e0aa'            # Vert très clair
        },
        "Marché de l'Alternance": {
            'Total_HTS': '#9b59b6',      # Violet
            'Budget': '#bb8fce',          # Violet clair
            'Reste': '#d7bde2'            # Violet très clair
        },
        'Marché des Entreprises': {
            'Total_HTS': '#e74c3c',      # Rouge
            'Budget': '#ec7063',          # Rouge clair
            'Reste': '#f1948a'            # Rouge très clair
        },
        'Marché Public': {
            'Total_HTS': '#f39c12',      # Orange
            'Budget': '#f8b739',          # Orange clair
            'Reste': '#fad7a0'            # Orange très clair
        }
    }
    
    # Ajouter les 3 métriques pour chaque financeur
    for financeur in financeurs_list:
        df_financeur = df_viz[df_viz['Financeur'] == financeur].set_index('Region')
        
        # Métrique 1: Total HTS & Suites
        y_values_hts = []
        for region in region_order:
            try:
                if region in df_financeur.index.tolist():
                    val = df_financeur.loc[region, 'Total_HTS_Suites']
                    # Convertir en nombre Python natif
                    if pd.notna(val):
                        y_values_hts.append(pd.to_numeric(val, errors='coerce'))
                    else:
                        y_values_hts.append(0)
                else:
                    y_values_hts.append(0)
            except:
                y_values_hts.append(0)
        
        fig.add_trace(go.Bar(
            name=f'{financeur} - Total HTS',
            x=region_order,
            y=y_values_hts,
            marker_color=financeur_colors[financeur]['Total_HTS'],
            text=[f"{v:,.0f}" if v > 0 else "" for v in y_values_hts],
            textposition='inside',
            legendgroup=financeur,
            showlegend=True
        ))
        
        # Métrique 2: Budget Décembre
        y_values_budget = []
        for region in region_order:
            try:
                if region in df_financeur.index.tolist():
                    val = df_financeur.loc[region, 'Budget_Decembre']
                    if pd.notna(val):
                        y_values_budget.append(pd.to_numeric(val, errors='coerce'))
                    else:
                        y_values_budget.append(0)
                else:
                    y_values_budget.append(0)
            except:
                y_values_budget.append(0)
        
        fig.add_trace(go.Bar(
            name=f'{financeur} - Budget Déc',
            x=region_order,
            y=y_values_budget,
            marker_color=financeur_colors[financeur]['Budget'],
            text=[f"{v:,.0f}" if v > 0 else "" for v in y_values_budget],
            textposition='inside',
            legendgroup=financeur,
            showlegend=True
        ))
        
        # Métrique 3: Reste à Faire
        y_values_reste = []
        for region in region_order:
            try:
                if region in df_financeur.index.tolist():
                    val = df_financeur.loc[region, 'Reste_A_Faire']
                    if pd.notna(val):
                        y_values_reste.append(pd.to_numeric(val, errors='coerce'))
                    else:
                        y_values_reste.append(0)
                else:
                    y_values_reste.append(0)
            except:
                y_values_reste.append(0)
        
        fig.add_trace(go.Bar(
            name=f'{financeur} - Reste à Faire',
            x=region_order,
            y=y_values_reste,
            marker_color=financeur_colors[financeur]['Reste'],
            text=[f"{v:,.0f}" if v > 0 else "" for v in y_values_reste],
            textposition='inside',
            legendgroup=financeur,
            showlegend=True
        ))
    
    # Configuration du graphique
    fig.update_layout(
        title="📅 Total HTS & Suites, Budget Décembre et Reste à Faire par Région et Financeur",
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
    
    st.markdown("### 📊 Statistiques Décembre par Région et Financeur")
    
    # Filtrer par régions si spécifié
    if regions_filter:
        df = df[df['Region'].isin(regions_filter)]
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    total_hts = df['Total_HTS_Suites'].sum()
    total_budget = df['Budget_Decembre'].sum()
    total_reste = df['Reste_A_Faire'].sum()
    tx_realisation = (total_hts / total_budget * 100) if total_budget > 0 else 0
    
    # Meilleur financeur pour Total HTS
    financeur_totals = df.groupby('Financeur')['Total_HTS_Suites'].sum()
    
    if not financeur_totals.empty:
        best_financeur = financeur_totals.idxmax()
        best_financeur_val = financeur_totals.max()
    else:
        best_financeur = "N/A"
        best_financeur_val = 0
    
    with col_stats1:
        st.metric(
            "📊 Total HTS & Suites",
            f"{total_hts:,.0f}",
            help="Total des HTS et suites de parcours"
        )
    
    with col_stats2:
        st.metric(
            "📈 Budget Total Déc",
            f"{total_budget:,.0f}",
            help="Budget total à fin décembre"
        )
    
    with col_stats3:
        st.metric(
            "📋 Reste à Faire Total",
            f"{total_reste:,.0f}",
            help="Total du reste à faire"
        )
    
    with col_stats4:
        st.metric(
            "🏆 Meilleur Financeur",
            best_financeur[:15] + "..." if len(best_financeur) > 15 else best_financeur,
            f"{best_financeur_val:,.0f}h"
        )
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région et Financeur"):
        display_data = df.copy()
        display_data['TX Réalisation (%)'] = (display_data['Total_HTS_Suites'] / 
                                               display_data['Budget_Decembre'] * 100).fillna(0)
        display_data['Écart Budget'] = display_data['Total_HTS_Suites'] - display_data['Budget_Decembre']
        
        columns_to_show = [
            'Region', 'Financeur', 'Total_HTS_Suites', 'Budget_Decembre', 
            'Reste_A_Faire', 'TX Réalisation (%)', 'Écart Budget'
        ]
        
        display_data_filtered = display_data[columns_to_show].copy()
        display_data_filtered = display_data_filtered.sort_values(['Region', 'Total_HTS_Suites'], ascending=[True, False])
        
        column_config = {
            'Region': 'Région',
            'Financeur': 'Financeur',
            'Total_HTS_Suites': st.column_config.NumberColumn(
                'Total HTS & Suites',
                format="%.0f"
            ),
            'Budget_Decembre': st.column_config.NumberColumn(
                'Budget Décembre',
                format="%.0f"
            ),
            'Reste_A_Faire': st.column_config.NumberColumn(
                'Reste à Faire',
                format="%.0f"
            ),
            'TX Réalisation (%)': st.column_config.NumberColumn(
                'TX Réalisation',
                format="%.1f%%"
            ),
            'Écart Budget': st.column_config.NumberColumn(
                'Écart Budget',
                format="%.0f",
                help="Total HTS - Budget Décembre"
            )
        }
        
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
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    total_hts = df['HTS_Realisees'].sum()
    total_budget = df['Budget_Septembre'].sum()
    tx_realisation = (total_hts / total_budget * 100) if total_budget > 0 else 0
    
    # Meilleur financeur
    financeur_totals = df.groupby('Financeur')['HTS_Realisees'].sum()
    if not financeur_totals.empty:
        best_financeur = financeur_totals.idxmax()
        best_financeur_val = financeur_totals.max()
    else:
        best_financeur = "N/A"
        best_financeur_val = 0
    
    with col_stats1:
        st.metric(
            "📊 Total HTS Réalisées",
            f"{total_hts:,.0f}",
            help="Total des heures réalisées avec PAE"
        )
    
    with col_stats2:
        st.metric(
            "📈 Budget Total Sept",
            f"{total_budget:,.0f}",
            help="Budget total à fin septembre"
        )
    
    with col_stats3:
        st.metric(
            "🎯 TX Réalisation",
            f"{tx_realisation:.1f}%",
            help="Taux de réalisation global"
        )
    
    with col_stats4:
        st.metric(
            "🏆 Meilleur Financeur",
            best_financeur[:15] + "..." if len(best_financeur) > 15 else best_financeur,
            f"{best_financeur_val:,.0f}h"
        )
    
    # Tableau détaillé
    with st.expander("📋 Données Détaillées par Région et Financeur"):
        display_data = df.copy()
        display_data['TX Réalisation (%)'] = (display_data['HTS_Realisees'] / 
                                               display_data['Budget_Septembre'] * 100).fillna(0)
        display_data['Écart'] = display_data['HTS_Realisees'] - display_data['Budget_Septembre']
        
        columns_to_show = [
            'Region', 'Financeur', 'HTS_Realisees', 'Budget_Septembre', 
            'TX Réalisation (%)', 'Écart'
        ]
        
        display_data_filtered = display_data[columns_to_show].copy()
        display_data_filtered = display_data_filtered.sort_values(['Region', 'HTS_Realisees'], ascending=[True, False])
        
        column_config = {
            'Region': 'Région',
            'Financeur': 'Financeur',
            'HTS_Realisees': st.column_config.NumberColumn(
                'HTS Réalisées',
                format="%.0f"
            ),
            'Budget_Septembre': st.column_config.NumberColumn(
                'Budget Septembre',
                format="%.0f"
            ),
            'TX Réalisation (%)': st.column_config.NumberColumn(
                'TX Réalisation',
                format="%.1f%%"
            ),
            'Écart': st.column_config.NumberColumn(
                'Écart',
                format="%.0f",
                help="HTS Réalisées - Budget"
            )
        }
        
        st.dataframe(
            display_data_filtered,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )

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
    tab1, tab2 = st.tabs(["📊 Analyse TX Réalisation", "💰 Analyse Financeurs"])
    
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
        # Charger la Feuil2 pour l'analyse financeurs
        try:
            if import_method == "Upload d'un nouveau fichier":
                df_financeurs = pd.read_excel(uploaded_file, sheet_name='Feuil2')
            else:
                df_financeurs = pd.read_excel(custom_path, sheet_name='Feuil2')
            
            # Premier graphique : HTS Réalisées vs Budget Septembre
            create_financeurs_visualization(df_financeurs)
            
            # Séparateur
            st.markdown("---")
            
            # Deuxième graphique : Analyse Décembre
            create_financeurs_visualization_decembre(df_financeurs)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de l'analyse financeurs: {str(e)}")
            st.info("💡 Assurez-vous que votre fichier contient une feuille 'Feuil2' avec les colonnes requises")
            
            # Debug info
            with st.expander("🔧 Informations de Debug"):
                st.write("Erreur:", str(e))