import pandas as pd
import xgboost as xgb
import s3fs
from sklearn.model_selection import train_test_split
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME

def train_xgboost():
    print("🚀 [XGBOOST] Bắt đầu huấn luyện...")
    
    # 1. Kết nối MinIO qua s3fs
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT}, 
        key=MINIO_ACCESS_KEY, 
        secret=MINIO_SECRET_KEY
    )
    
    # 2. SỬA LỖI: Bỏ tiền tố 's3://' trong đường dẫn khi dùng kèm filesystem=fs
    # Thay vì: f"s3://{BUCKET_NAME}/gold/features.parquet"
    # Hãy dùng: f"{BUCKET_NAME}/gold/features.parquet"
    path = f"{BUCKET_NAME}/gold/features.parquet"
    
    print(f"📂 Đang đọc dữ liệu từ: {path}")
    df = pd.read_parquet(path, filesystem=fs)
    
    # 3. Đổi tên cột mục tiêu (PM2_5 thay vì PM2.5 như đã sửa ở bước Spark)
    target_col = "PM2_5" 
    
    # Kiểm tra xem cột có tồn tại không để tránh lỗi
    if target_col not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột {target_col}. Các cột hiện có: {df.columns.tolist()}")
        return

    features_columns = [col for col in df.columns if col not in ['No', 'station', 'wd', target_col]]
    
    # 4. Chia tập dữ liệu
    train_data, test = train_test_split(df, test_size=0.1, shuffle=False)
    train, valid = train_test_split(train_data, test_size=0.1, shuffle=False)
    
    # Chuẩn bị dữ liệu cho XGBoost
    d_train = xgb.DMatrix(train[features_columns], label=train[target_col])
    d_val = xgb.DMatrix(valid[features_columns], label=valid[target_col])
    
    XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': ["mae", "rmse"], 
        'learning_rate': 0.05, 
        'max_depth': 6, 
        'tree_method': 'hist', 
        'seed': 42
    }
    
    # 5. Huấn luyện
    model = xgb.train(
        XGB_PARAMS, 
        d_train, 
        evals=[(d_val, "validation")], 
        num_boost_round=500, 
        verbose_eval=50, 
        early_stopping_rounds=20
    )
    
    model.save_model("model_xgboost_pm25.json")
    print("✅ Đã huấn luyện xong và lưu model: model_xgboost_pm25.json")

if __name__ == "__main__":
    train_xgboost()