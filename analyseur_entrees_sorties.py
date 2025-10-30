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
    
    # ========== MODE VUE SIMPLE ==========
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