# -*- coding: utf-8 -*-
"""
Fonctions utilitaires communes pour les analyseurs
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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