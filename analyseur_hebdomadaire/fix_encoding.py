#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour corriger les problèmes d'encodage dans le fichier analyseur_hebdomadaire.py
"""

import codecs
import os

def fix_encoding(input_file, output_file):
    """Corrige les problèmes d'encodage"""
    
    try:
        # Lire le fichier original
        with codecs.open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Remplacer les caractères problématiques
        replacements = {
            '�': '📊',
            ''': "'",
            ''': "'", 
            '`': "'",
            '"': '"',
            '"': '"'
        }
        
        for old_char, new_char in replacements.items():
            content = content.replace(old_char, new_char)
        
        # Écrire le fichier corrigé
        with codecs.open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fichier corrigé créé avec succès : {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction : {e}")
        return False

if __name__ == "__main__":
    input_file = r"c:\Dashboard\analyseur_hebdomadaire\analyseur_hebdomadaire.py"
    output_file = r"c:\Dashboard\analyseur_hebdomadaire\analyseur_hebdomadaire_fixed.py"
    
    if os.path.exists(input_file):
        fix_encoding(input_file, output_file)
    else:
        print(f"❌ Fichier d'entrée non trouvé : {input_file}")