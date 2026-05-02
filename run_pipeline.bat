@echo off
color 0B
echo ========================================================
echo   HE THONG MLOPS: DU BAO PM2.5 BAC KINH
echo ========================================================
echo [1] Khoi dong Ha Tang Docker (MinIO, Spark, Cassandra, Grafana)
echo [2] Chay Data Pipeline (Tien xu ly, Ingestion, Spark ETL)
echo [3] Huan Luyen Mo Hinh AI
echo [4] Danh gia va Ve Bieu do (Evaluation)
echo [5] Du bao Thuc te va Luu DB (Inference)
echo ========================================================
set /p action="Chon chuc nang (1-5): "

if "%action%"=="1" (
    docker-compose up -d
    echo Doi 45 giay de he thong khoi dong hoan toan...
    pause
)
if "%action%"=="2" (
    cd src
    echo Dang tien xu ly... & python 0_data_preprocessing.py
    echo Dang day len MinIO... & python 1_ingestion_minio.py
    echo PySpark dang chay... & python 2_spark_etl.py
    pause
)
if "%action%"=="3" (
    cd src
    echo A. XGBoost
    echo B. LightGBM
    echo C. PyTorch LSTM
    set /p model="Chon mo hinh can huan luyen (A/B/C): "
    if /I "%model%"=="A" python 3a_train_xgboost.py
    if /I "%model%"=="B" python 3b_train_lightgbm.py
    if /I "%model%"=="C" python 3c_train_lstm.py
    pause
)
if "%action%"=="4" (
    cd src
    python 5_evaluate_visualize.py
    pause
)
if "%action%"=="5" (
    cd src
    python 4_batch_inference.py
    pause
)