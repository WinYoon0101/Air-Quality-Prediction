# --- BƯỚC 1: FIX LỖI PYTHON 3.12 (PHẢI ĐẶT TRÊN CÙNG) ---
import sys
try:
    import asyncore
except ImportError:
    import pyasyncore
    sys.modules['asyncore'] = pyasyncore
# ------------------------------------------------------

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import s3fs
import uuid
import warnings
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.io.asyncioreactor import AsyncioConnection
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME, CASSANDRA_HOST

warnings.filterwarnings('ignore')

def run_batch_inference():
    print("🚀 [BATCH INFERENCE] Khởi động quy trình dự báo...")
    
    # 1. Kết nối và Dọn dẹp Cassandra
    try:
        cluster = Cluster([CASSANDRA_HOST], port=9042)
        cluster.connection_class = AsyncioConnection 
        session = cluster.connect()
        
        session.execute("""
            CREATE KEYSPACE IF NOT EXISTS air_quality 
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
        """)
        session.execute("""
            CREATE TABLE IF NOT EXISTS air_quality.pm25_forecast (
                id UUID PRIMARY KEY, 
                timestamp timestamp, 
                actual float, 
                predicted float
            )
        """)

        print("🧹 Đang dọn dẹp dữ liệu cũ...")
        session.execute("TRUNCATE air_quality.pm25_forecast")
        print("✅ Cassandra đã sẵn sàng.")
    except Exception as e:
        print(f"❌ Thất bại khi kết nối Cassandra: {e}")
        return

    # 2. Lấy dữ liệu từ Data Lake (MinIO)
    print("📂 Đang truy xuất dữ liệu từ Gold Layer...")
    try:
        fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': MINIO_ENDPOINT}, 
            key=MINIO_ACCESS_KEY, 
            secret=MINIO_SECRET_KEY
        )
        path = f"{BUCKET_NAME}/gold/features.parquet"
        df_full = pd.read_parquet(path, filesystem=fs)
        
        # Lấy 200 dòng cuối để dự báo
        df = df_full.tail(200).copy()
        
        # Gộp thời gian để làm nhãn hiển thị (nhưng vẫn giữ các cột gốc để làm feature)
        df['actual_timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        start_date = df['actual_timestamp'].min()
        end_date = df['actual_timestamp'].max()
        print(f"📅 Dữ liệu thực tế: Từ {start_date} đến {end_date}")

        target_col = "PM2_5"
        if target_col not in df.columns and "PM2.5" in df.columns:
            df = df.rename(columns={"PM2.5": target_col})
    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu: {e}")
        return

    # 3. Chạy mô hình dự báo (Tự động khớp Feature)
    print(f"🧠 Đang dự báo bằng LightGBM...")
    try:
        model = lgb.Booster(model_file="model_lightgbm_pm25.txt")
        
        # Lấy danh sách tên feature mà model yêu cầu
        expected_features = model.feature_name()
        print(f"📊 Model yêu cầu {len(expected_features)} features. Đang chuẩn bị dữ liệu...")

        # Ép kiểu dữ liệu category cho các cột cần thiết
        cat_features = ['station', 'wd']
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Lọc đúng và đủ các cột mà model cần (bao gồm cả year, month, day, hour nếu có)
        df_inference = df[expected_features]
        
        # Dự báo
        df['predicted'] = model.predict(df_inference)
        print("✅ Dự báo thành công.")
    except Exception as e:
        print(f"❌ Lỗi dự báo: {e}")
        return

    # 4. Nạp dữ liệu vào Cassandra
    print(f"💾 Đang nạp {len(df)} bản ghi vào DB...")
    insert_stmt = session.prepare("""
        INSERT INTO air_quality.pm25_forecast (id, timestamp, actual, predicted) 
        VALUES (?, ?, ?, ?)
    """)
    
    count = 0
    for _, row in df.iterrows():
        try:
            session.execute(insert_stmt, [
                uuid.uuid4(), 
                row['actual_timestamp'], 
                float(row[target_col]), 
                float(row['predicted'])
            ])
            count += 1
        except Exception as e:
            continue
            
    print(f"🔥 HOÀN THÀNH! Đã nạp {count} dòng vào Cassandra.")
    cluster.shutdown()

if __name__ == "__main__":
    run_batch_inference()