import pandas as pd

# Charger le fichier
df = pd.read_excel(r'c:\Users\2501555\OneDrive - AFPA\Documents\data_format_analyseur\data_test_atterissage.xlsx', sheet_name='Feuil2')

print('=== COLONNES ===')
for col in df.columns:
    print(repr(col))

# Détecter les colonnes comme dans le code
hts_col = None
budget_col = None

for col in df.columns:
    col_upper = col.upper()
    if 'HTS' in col_upper and 'REALIS' in col_upper and hts_col is None:
        hts_col = col
    if 'BUDGET' in col_upper and budget_col is None:
        budget_col = col

print(f'\nColonne HTS détectée: {repr(hts_col)}')
print(f'Colonne Budget détectée: {repr(budget_col)}')

# Simuler le parsing
financeurs_list = ['B2C - CPF', 'B2C - CPFT', "Marché de l'Alternance", 
                   'Marché des Entreprises', 'Marché Public', 'Pas de financeur']

data_restructured = []
current_region = None

for idx, row in df.iterrows():
    region_name = row['Régions']
    
    # Ignorer les lignes complètement vides
    if pd.isna(region_name) or str(region_name).strip() == '':
        continue
    
    # Nettoyer le nom de région
    region_name = str(region_name).strip()
    
    # Si c'est une région (pas un financeur)
    if region_name not in financeurs_list:
        # Vérifier que ce n'est pas un total à exclure
        if any(x in str(region_name).lower() for x in ['total', 'ensemble']):
            # Réinitialiser current_region pour éviter d'associer les financeurs suivants à la mauvaise région
            current_region = None
            print(f"Région EXCLUE (filtrée): {region_name}")
        else:
            current_region = region_name
            print(f"Région détectée: {current_region}")
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
        
        if current_region == 'Corse':
            print(f"  -> Financeur: {region_name}, HTS: {row[hts_col]}, Budget: {row[budget_col] if budget_col else 'N/A'}")

# Créer le DataFrame
df_restructured = pd.DataFrame(data_restructured)

print('\n=== DONNÉES CORSE EXTRAITES ===')
corse_data = df_restructured[df_restructured['Region'] == 'Corse']
print(corse_data)

print('\n=== DONNÉES CORSE AVANT CONVERSION ===')
print(corse_data[['Region', 'Financeur', 'HTS_Realisees', 'Budget']].to_string())

# Conversion numérique
df_restructured['HTS_Realisees'] = pd.to_numeric(df_restructured['HTS_Realisees'], errors='coerce').fillna(0)
if 'Budget' in df_restructured.columns:
    df_restructured['Budget'] = pd.to_numeric(df_restructured['Budget'], errors='coerce').fillna(0)

print('\n=== DONNÉES CORSE APRÈS CONVERSION ===')
corse_data_after = df_restructured[df_restructured['Region'] == 'Corse']
print(corse_data_after[['Region', 'Financeur', 'HTS_Realisees', 'Budget']].to_string())
