import pandas as pd

df = pd.read_excel(r'c:\Users\2501555\OneDrive - AFPA\Documents\data_format_analyseur\data_test_atterissage.xlsx', sheet_name='Feuil2')

print("Lignes 28 à 42:")
for i in range(28, 42):
    print(f"{i}: {df.iloc[i]['Régions']}")
