@echo off
:: =============================================================================
:: 📊 LANCEUR - ANALYSEUR HEBDOMADAIRE  
:: Script de lancement pour l'analyse hebdomadaire avec import de fichiers Excel
:: Créé le: 20 octobre 2025 - Mis à jour: Import de fichiers flexible
:: =============================================================================

echo.
echo ========================================================
echo   📊 ANALYSEUR HEBDOMADAIRE - PRODUCTION REGIONALE
echo ========================================================
echo.
echo ⏳ Lancement de l'application...
echo.

:: Vérification de l'existence de l'application
if not exist "analyseur_hebdomadaire.py" (
    echo ❌ ERREUR: Le fichier analyseur_hebdomadaire.py est introuvable
    echo.
    pause
    exit /b 1
)

:: Information sur l'import de fichiers
echo ✅ Application d'analyse avec import de fichiers Excel
echo 📁 Vous pourrez importer vos fichiers dans l'interface
echo.

:: Lancement de l'application Streamlit
echo 🚀 Ouverture de l'analyseur dans votre navigateur...
echo.
echo 📋 Instructions:
echo    - L'application va s'ouvrir automatiquement dans votre navigateur
echo    - Pour arrêter l'application: Ctrl+C dans cette fenêtre
echo    - URL d'accès: http://localhost:8502
echo.

:: Lancement avec port dédié pour éviter les conflits
python -m streamlit run analyseur_hebdomadaire.py --server.port 8502 --server.headless true

:: En cas d'erreur
if errorlevel 1 (
    echo.
    echo ❌ Erreur lors du lancement de l'application
    echo.
    echo 💡 Solutions possibles:
    echo    1. Vérifiez que Python est installé
    echo    2. Installez Streamlit: pip install streamlit
    echo    3. Vérifiez que le port 8502 n'est pas utilisé
    echo.
    pause
)

echo.
echo 👋 Application fermée. Appuyez sur une touche pour quitter...
pause >nul