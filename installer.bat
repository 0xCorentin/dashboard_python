@echo off
echo ========================================
echo    INSTALLATION ANALYSEUR HEBDOMADAIRE
echo ========================================
echo.

echo Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Veuillez installer Python depuis le Microsoft Store
    pause
    exit /b 1
)

echo Python detecte avec succes!
echo.

echo Installation des dependances...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERREUR: Echec de l'installation des dependances
    pause
    exit /b 1
)

echo.
echo ========================================
echo    INSTALLATION TERMINEE AVEC SUCCES
echo ========================================
echo.
echo Pour lancer l'analyseur:
echo   - Double-cliquez sur lancer_analyseur.bat
echo.
echo   L'application sera accessible sur: http://localhost:8502
echo.
pause