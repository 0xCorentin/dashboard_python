# -*- coding: utf-8 -*-
"""
Analyseur Entrées-Sorties-Abandons
Module dédié à l'analyse des flux de stagiaires par région et financeur
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots

# Import des fonctions utilitaires
from utils import (
    load_entrees_sorties_data, load_monthly_data, load_feuil3_data, 
    compute_totals
)

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
        
        # ========== AJOUTER LA COURBE DE MOYENNE SI C'EST "ENTRÉES" ==========
        if analysis_type == "Entrées":
            # Calculer la moyenne des entrées pour chaque région sur tous les mois
            region_averages = []
            region_names = []
            
            for region in regions_to_show:
                region_data = df_filtered[df_filtered['Region'] == region]
                if not region_data.empty:
                    region_row = region_data.iloc[0]
                    
                    # Calculer la moyenne des entrées sur tous les mois sélectionnés
                    total_entries = 0
                    valid_months = 0
                    
                    for month in months_to_show:
                        col_name = f'{month}_Entrées'
                        if col_name in region_row.index and pd.notna(region_row[col_name]):
                            total_entries += region_row[col_name]
                            valid_months += 1
                    
                    if valid_months > 0:
                        average = total_entries / valid_months
                        region_averages.append(average)
                        region_names.append(region)
            
            if region_averages:
                # Calculer un décalage pour éviter que les textes se chevauchent avec la courbe
                max_bar_value = 0
                for month in months_to_show:
                    for region in regions_to_show:
                        region_data = df_filtered[df_filtered['Region'] == region]
                        if not region_data.empty:
                            col_name = f'{month}_Entrées'
                            if col_name in region_data.iloc[0].index:
                                value = region_data.iloc[0][col_name]
                                max_bar_value = max(max_bar_value, value)
                
                # Décaler les moyennes vers le haut pour éviter les chevauchements
                text_offset = max_bar_value * 0.08  # 8% du max pour le décalage
                adjusted_averages = [avg + text_offset for avg in region_averages]
                
                # Ajouter la courbe de moyenne sur le même graphique
                fig.add_trace(go.Scatter(
                    x=region_names,
                    y=region_averages,
                    mode='lines+markers',
                    name='📊 Moyenne des Entrées',
                    line=dict(color='#FF4444', width=4, dash='solid'),
                    marker=dict(
                        size=12,
                        color='#FF4444',
                        symbol='diamond',
                        line=dict(width=3, color='white')
                    ),
                    hovertemplate='<b>%{x}</b><br>Moyenne: %{y:,.0f} entrées<br><i>Calculée sur ' + str(len(months_to_show)) + ' mois</i><extra></extra>',
                    yaxis='y2'  # Utiliser un axe Y secondaire pour la courbe
                ))
                
                # Ajouter les textes des moyennes séparément avec décalage
                fig.add_trace(go.Scatter(
                    x=region_names,
                    y=adjusted_averages,
                    mode='text',
                    name='Valeurs Moyennes',
                    text=[f"{avg:,.0f}" for avg in region_averages],
                    textposition='middle center',
                    textfont=dict(
                        color='#FF4444', 
                        size=12, 
                        family='Arial Black'
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    yaxis='y2'
                ))
                
                # Configurer l'axe Y secondaire pour la courbe
                fig.update_layout(
                    yaxis2=dict(
                        title="Moyenne des Entrées",
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        color='#2E86AB'
                    )
                )
        
        metric_emojis = {
            "Entrées": "📥", 
            "Sorties": "📤", 
            "Abandons": "❌", 
            "Écart": "📊"
        }
        
        title_suffix = " avec Courbe de Moyenne" if analysis_type == "Entrées" else ""
        
        fig.update_layout(
            title=f"{metric_emojis.get(analysis_type, '📊')} {analysis_type} par Région et Mois{title_suffix}",
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
        
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Options de configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        regions_to_show = st.multiselect(
            "🏷️ Régions à afficher:",
            all_regions,
            default=all_regions,
            key="feuil3_regions_select"
        )
    
    with col_config2:
        financeurs_to_show = st.multiselect(
            "💰 Financeurs à afficher:",
            all_financeurs,
            default=all_financeurs,
            key="feuil3_financeurs_select"
        )
    
    with col_config3:
        metric_choice = st.selectbox(
            "📊 Métrique à analyser:",
            metric_cols,
            index=0,
            key="feuil3_metric"
        )
    
    # Détecter automatiquement si des colonnes 2024 et 2025 existent
    cols_2024 = [col for col in metric_cols if '2024' in str(col) and 'Ecart' not in str(col)]
    cols_2025 = [col for col in metric_cols if '2025' in str(col) and 'Ecart' not in str(col)]
    cols_ecart = [col for col in metric_cols if 'Ecart' in str(col) or 'écart' in str(col).lower()]
    
    can_compare_years = len(cols_2024) > 0 and len(cols_2025) > 0
    
    # Option de comparaison d'années
    st.markdown("### 🔄 Options de Comparaison")
    col_comp1, col_comp2 = st.columns(2)
    
    comparison_metric = None  # Initialiser la variable
    
    with col_comp1:
        if can_compare_years:
            comparison_mode = st.selectbox(
                "📊 Mode d'analyse:",
                ["Vue simple", "Comparaison 2024 vs 2025"],
                key="feuil3_comparison_mode"
            )
        else:
            comparison_mode = "Vue simple"
            st.info("ℹ️ Mode comparaison indisponible (colonnes 2024/2025 manquantes)")

    with col_comp2:
        if comparison_mode == "Comparaison 2024 vs 2025" and can_compare_years:
            # Déterminer les métriques de base (sans l'année)
            base_metrics = set()
            for col in cols_2024:
                # Enlever "2024" et nettoyer le nom
                base_name = col.replace('2024', '').replace('_2024', '').strip()
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
            st.info("💡 Sélectionnez 'Comparaison 2024 vs 2025' pour comparer les années")
    
    # Toujours utiliser la vue simple
    graph_type = "Vue simple (métrique sélectionnée)"
    
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
    
    # ========== MODE COMPARAISON 2024 vs 2025 ==========
    if comparison_mode == "Comparaison 2024 vs 2025" and can_compare_years and comparison_metric:
            
        st.markdown("### 📊 Comparaison 2024 vs 2025")
        
        # Construire les noms de colonnes pour 2024 et 2025
        # Chercher les colonnes correspondantes à la métrique choisie
        possible_cols_2024 = []
        possible_cols_2025 = []
        
        # Essayer plusieurs variantes de recherche
        if comparison_metric:
            search_patterns = [
                comparison_metric,  # Nom exact
                f"{comparison_metric} 2024",  # Avec année
                comparison_metric.replace('.', '').replace(',', ''),  # Sans ponctuation
            ]
            
            for pattern in search_patterns:
                if not possible_cols_2024:
                    possible_cols_2024 = [c for c in cols_2024 if pattern.lower() in c.lower()]
                if not possible_cols_2025:
                    possible_cols_2025 = [c for c in cols_2025 if pattern.lower() in c.lower()]
                
                if possible_cols_2024 and possible_cols_2025:
                    break
            
            # Si on n'a toujours pas trouvé, chercher juste avec une partie du nom
            if not possible_cols_2024 or not possible_cols_2025:
                # Prendre le premier mot significatif
                first_word = comparison_metric.split()[0] if ' ' in comparison_metric else comparison_metric
                possible_cols_2024 = [c for c in cols_2024 if first_word.lower() in c.lower()]
                possible_cols_2025 = [c for c in cols_2025 if first_word.lower() in c.lower()]
        
        # Vérifier qu'on a bien trouvé des colonnes
        if not possible_cols_2024 or not possible_cols_2025:
            st.error(f"❌ Impossible de trouver les colonnes pour '{comparison_metric}'")
            
            with st.expander("🔍 Informations de Debug - Colonnes Détectées"):
                st.write(f"**Métrique recherchée:** {comparison_metric}")
                st.write(f"**Colonnes 2024 disponibles:** {', '.join(cols_2024) if cols_2024 else 'Aucune'}")
                st.write(f"**Colonnes 2025 disponibles:** {', '.join(cols_2025) if cols_2025 else 'Aucune'}")
                st.write(f"**Colonnes d'écart disponibles:** {', '.join(cols_ecart) if cols_ecart else 'Aucune'}")
                st.write(f"**Toutes les colonnes métriques:** {', '.join(metric_cols)}")
            
            return
        
        col_2024 = possible_cols_2024[0]
        col_2025 = possible_cols_2025[0]
        
        # Chercher la colonne d'écart si elle existe
        col_ecart = None
        if cols_ecart:
            ecart_matches = [c for c in cols_ecart if '2025' in c and '2024' in c]
            if ecart_matches:
                col_ecart = ecart_matches[0]
        
        st.info(f"📊 Comparaison: **{col_2024}** vs **{col_2025}**" + 
                (f" | Écart: **{col_ecart}**" if col_ecart else ""))
        
        # Calculer l'ordre des régions par total 2025
        region_totals_2025 = df_filtered.groupby('REGION')[col_2025].sum().sort_values(ascending=False)
        region_order = region_totals_2025.index.tolist()
        

        
        # Afficher le graphique en vue simple
        st.markdown("### 📊 Comparaison 2024 vs 2025")
        
        fig_compare = go.Figure()
        
        # Barres pour 2024 (toutes régions)
        regions_2024 = []
        values_2024 = []
        for region in region_order:
            region_data = df_filtered[df_filtered['REGION'] == region]
            if not region_data.empty and col_2024 in region_data.columns:
                regions_2024.append(region)
                values_2024.append(region_data[col_2024].sum())
        
        fig_compare.add_trace(go.Bar(
            x=regions_2024,
            y=values_2024,
            name='2024',
            marker_color='#FF6B6B',
            text=[f"{val:,.0f}" for val in values_2024],
            textposition='outside',
            offsetgroup=0
        ))
        
        # Barres pour 2025 (toutes régions)
        regions_2025 = []
        values_2025 = []
        for region in region_order:
            region_data = df_filtered[df_filtered['REGION'] == region]
            if not region_data.empty and col_2025 in region_data.columns:
                regions_2025.append(region)
                values_2025.append(region_data[col_2025].sum())
        
        fig_compare.add_trace(go.Bar(
            x=regions_2025,
            y=values_2025,
            name='2025',
            marker_color='#4ECDC4',
            text=[f"{val:,.0f}" for val in values_2025],
            textposition='outside',
            offsetgroup=1
        ))
        
        # Calculer et ajouter les écarts (2025 - 2024)
        regions_ecart = []
        values_ecart = []
        for region in region_order:
            region_data = df_filtered[df_filtered['REGION'] == region]
            if not region_data.empty:
                value_2024 = region_data[col_2024].sum() if col_2024 in region_data.columns else 0
                value_2025 = region_data[col_2025].sum() if col_2025 in region_data.columns else 0
                ecart = value_2025 - value_2024
                regions_ecart.append(region)
                values_ecart.append(ecart)
        
        fig_compare.add_trace(go.Bar(
            x=regions_ecart,
            y=values_ecart,
            name='Écart (2025-2024)',
            marker_color='#FFB347',  # Orange pour l'écart
            text=[f"{val:+,.0f}" for val in values_ecart],  # Afficher le signe + ou -
            textposition='outside',
            offsetgroup=2
        ))
        
        # Configuration du graphique simple
        fig_compare.update_layout(
            title=f"📊 Comparaison {comparison_metric} : 2024 vs 2025 avec Écart",
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
        
        # ========== NOUVEAU GRAPHIQUE PAR FINANCEURS ==========
        st.markdown("---")
        st.markdown("### 📊 Analyse Détaillée par Financeurs")
        
        try:
            # Charger les données du fichier Excel spécifique
            excel_path = 'c:/Dashboard/analyseur_hebdomadaire/flux_model.xlsx'
            df_excel = pd.read_excel(excel_path, sheet_name='Feuil3')
            
            # Traiter les données pour avoir une structure propre Region-Financeur
            processed_data = []
            current_region = None
            
            for _, row in df_excel.iterrows():
                if pd.notna(row['Régions']) and row['Régions'] != 'Total général':
                    current_region = row['Régions']
                    # La première ligne de région contient aussi un financeur
                    processed_data.append({
                        'REGION': current_region,
                        'FINANCEURS': row['Financeurs'],
                        'Entrées 2024': row['Nb. de stagiaires entrés 2024'],
                        'Entrées 2025': row['Nb. de stagiaires entrés 2025'],
                        'Écart': row['Ecart\n(2025-2024)']
                    })
                elif pd.notna(row['Financeurs']) and current_region:
                    # Lignes suivantes avec des financeurs pour la même région
                    processed_data.append({
                        'REGION': current_region,
                        'FINANCEURS': row['Financeurs'],
                        'Entrées 2024': row['Nb. de stagiaires entrés 2024'],
                        'Entrées 2025': row['Nb. de stagiaires entrés 2025'],
                        'Écart': row['Ecart\n(2025-2024)']
                    })
            
            df_financeurs = pd.DataFrame(processed_data)
            
            # Filtrer selon les régions sélectionnées
            df_financeurs_filtered = df_financeurs[df_financeurs['REGION'].isin(regions_to_show)].copy()
            
            if not df_financeurs_filtered.empty:
                # Créer le graphique par financeurs
                fig_financeurs = go.Figure()
                
                # Couleurs spécifiques pour chaque financeur
                financeur_colors = {
                    'B2C - CPF': '#FF6B6B',
                    'B2C - CPFT': '#4ECDC4', 
                    'Marché de l\'Alternance': '#45B7D1',
                    'Marché des Entreprises': '#96CEB4',
                    'Marché Public': '#FFEAA7'
                }
                
                # Ordre des régions par total 2025 décroissant
                region_totals_financeurs = df_financeurs_filtered.groupby('REGION')['Entrées 2025'].sum().sort_values(ascending=False)
                region_order_financeurs = region_totals_financeurs.index.tolist()
                
                # Créer deux graphiques séparés : un pour 2024 et un pour 2025
                
                # ========== GRAPHIQUE 2024 ==========
                st.markdown("#### 📅 Année 2024")
                fig_2024 = go.Figure()
                
                for i, financeur in enumerate(df_financeurs_filtered['FINANCEURS'].unique()):
                    df_fin = df_financeurs_filtered[df_financeurs_filtered['FINANCEURS'] == financeur]
                    
                    regions_fin = []
                    values_fin = []
                    
                    for region in region_order_financeurs:
                        region_data = df_fin[df_fin['REGION'] == region]
                        if not region_data.empty:
                            regions_fin.append(region)
                            values_fin.append(region_data['Entrées 2024'].iloc[0])
                    
                    if regions_fin:
                        color_base = financeur_colors.get(financeur, '#999999')
                        
                        fig_2024.add_trace(go.Bar(
                            x=regions_fin,
                            y=values_fin,
                            name=financeur,
                            marker_color=color_base,
                            text=[f"{val:,.0f}" for val in values_fin],
                            textposition='outside',
                            hovertemplate=f'<b>{financeur}</b><br>2024: %{{y:,.0f}}<br>Région: %{{x}}<extra></extra>'
                        ))
                
                fig_2024.update_layout(
                    title="📊 Entrées par Financeur - Année 2024",
                    xaxis_title="Régions",
                    yaxis_title="Nombre de stagiaires entrés",
                    height=600,
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
                
                st.plotly_chart(fig_2024, use_container_width=True)
                
                # ========== GRAPHIQUE 2025 ==========
                st.markdown("#### 📅 Année 2025")
                fig_2025 = go.Figure()
                
                for i, financeur in enumerate(df_financeurs_filtered['FINANCEURS'].unique()):
                    df_fin = df_financeurs_filtered[df_financeurs_filtered['FINANCEURS'] == financeur]
                    
                    regions_fin = []
                    values_fin = []
                    
                    for region in region_order_financeurs:
                        region_data = df_fin[df_fin['REGION'] == region]
                        if not region_data.empty:
                            regions_fin.append(region)
                            values_fin.append(region_data['Entrées 2025'].iloc[0])
                    
                    if regions_fin:
                        color_base = financeur_colors.get(financeur, '#999999')
                        
                        fig_2025.add_trace(go.Bar(
                            x=regions_fin,
                            y=values_fin,
                            name=financeur,
                            marker_color=color_base,
                            text=[f"{val:,.0f}" for val in values_fin],
                            textposition='outside',
                            hovertemplate=f'<b>{financeur}</b><br>2025: %{{y:,.0f}}<br>Région: %{{x}}<extra></extra>'
                        ))
                
                fig_2025.update_layout(
                    title="📊 Entrées par Financeur - Année 2025",
                    xaxis_title="Régions",
                    yaxis_title="Nombre de stagiaires entrés",
                    height=600,
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
                
                st.plotly_chart(fig_2025, use_container_width=True)
                
                # ========== GRAPHIQUE ÉCART (2025 - 2024) ==========
                st.markdown("#### 📊 Évolution (Écart 2025-2024)")
                fig_ecart = go.Figure()
                
                for i, financeur in enumerate(df_financeurs_filtered['FINANCEURS'].unique()):
                    df_fin = df_financeurs_filtered[df_financeurs_filtered['FINANCEURS'] == financeur]
                    
                    regions_fin = []
                    values_fin = []
                    
                    for region in region_order_financeurs:
                        region_data = df_fin[df_fin['REGION'] == region]
                        if not region_data.empty:
                            regions_fin.append(region)
                            values_fin.append(region_data['Écart'].iloc[0])
                    
                    if regions_fin:
                        # Couleur orange pour les écarts, avec différentes nuances par financeur
                        base_orange = '#FFB347'
                        if i == 0:
                            marker_color = '#FF8C00'  # Orange foncé
                        elif i == 1:
                            marker_color = '#FFB347'  # Orange moyen
                        elif i == 2:
                            marker_color = '#FFA500'  # Orange
                        elif i == 3:
                            marker_color = '#FF7F50'  # Coral
                        else:
                            marker_color = '#FF6347'  # Tomato
                        
                        fig_ecart.add_trace(go.Bar(
                            x=regions_fin,
                            y=values_fin,
                            name=financeur,
                            marker_color=marker_color,
                            text=[f"{val:+,.0f}" for val in values_fin],
                            textposition='outside',
                            hovertemplate=f'<b>{financeur}</b><br>Écart (2025-2024): %{{y:+,.0f}}<br>Région: %{{x}}<extra></extra>'
                        ))
                
                fig_ecart.update_layout(
                    title="📊 Évolution par Financeur - Écart (2025-2024)",
                    xaxis_title="Régions",
                    yaxis_title="Écart (Nombre de stagiaires)",
                    height=600,
                    barmode='group',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    xaxis_tickangle=-45,
                    # Ajouter une ligne de référence à y=0
                    shapes=[
                        dict(
                            type="line",
                            x0=-0.5,
                            y0=0,
                            x1=len(region_order_financeurs)-0.5,
                            y1=0,
                            line=dict(color="black", width=2, dash="dash")
                        )
                    ]
                )
                
                st.plotly_chart(fig_ecart, use_container_width=True)
                
            else:
                st.warning("⚠️ Aucune donnée financeur disponible pour les régions sélectionnées")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données financeurs : {str(e)}")
            st.info("💡 Vérifiez que le fichier 'c:/Dashboard/analyseur_hebdomadaire/flux_model.xlsx' existe et contient la feuille 'Feuil3'")
        
        # KPI de Comparaison pour la vue simple
        st.markdown("### 📈 KPI de Comparaison 2024 vs 2025")
        
        total_2024 = df_filtered[col_2024].sum() if col_2024 in df_filtered.columns else 0
        total_2025 = df_filtered[col_2025].sum() if col_2025 in df_filtered.columns else 0
        evolution = total_2025 - total_2024
        evolution_pct = (evolution / total_2024 * 100) if total_2024 > 0 else 0
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric(
                "📊 Total 2024",
                f"{total_2024:,.0f}",
                help=f"Total {comparison_metric} en 2024"
            )
        
        with kpi_col2:
            st.metric(
                "📊 Total 2025",
                f"{total_2025:,.0f}",
                delta=f"{evolution:+,.0f}",
                help=f"Total {comparison_metric} en 2025"
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
            avg_2025 = total_2025 / len(regions_to_show) if len(regions_to_show) > 0 else 0
            avg_2024 = total_2024 / len(regions_to_show) if len(regions_to_show) > 0 else 0
            avg_evolution_pct = ((avg_2025 - avg_2024) / avg_2024 * 100) if avg_2024 > 0 else 0
            st.metric(
                "📊 Moy. par Région 2025",
                f"{avg_2025:,.0f}",
                delta=f"{avg_evolution_pct:+.1f}%",
                help="Moyenne par région en 2025"
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
    
    required_metrics = ['Entrées', 'Sorties', 'Abandons']
    for metric in required_metrics:
        if metric not in df.columns:
            warnings.append(f"❌ Colonne '{metric}' non trouvée - impossible de créer les analyses")
    
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