@echo off
echo =========================================
echo BISFT Random Forest Forecasting Pipeline
echo =========================================

echo.
echo [1/3] Deleting legacy Prophet and LSTM files...
python scripts\delete_old_models.py

echo.
echo [2/3] Engineering Multivariate Features from Olist Data...
python src\forecasting\rf_feature_engineering.py

echo.
echo [3/3] Training Random Forest Regressor and Tuning Hyperparameters...
python src\models\rf_forecaster.py

echo.
echo =========================================
echo Pipeline Complete! You can now start the FastAPI backend.
echo =========================================
pause
