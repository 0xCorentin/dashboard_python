# -*- coding: utf-8 -*-
"""
Analyseur Production Mensuelle
Module dédié à l'analyse de la production HORS PAE par région et financeur
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Import des fonctions utilitaires
from utils import load_monthly_production_data

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
    col_config1, col_config2, col_config3 = st.columns(3)
    
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
        
        if regions_ordered:  # Seulement si on a des données pour ce financeur
            fig.add_trace(go.Bar(
                x=regions_ordered,  # Régions en X dans l'ordre choisi
                y=percentages_ordered,   # Pourcentage en Y
                name=financeur,
                marker_color=colors[i % len(colors)],
                text=[f"{val:.1f}%" for val in percentages_ordered],
                textposition='outside',
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
            range=[0, 110]  # Limite à 110% pour laisser de l'espace aux labels
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
        xaxis_tickangle=-45  # Rotation des noms de régions
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