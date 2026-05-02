import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocess_raw_data():
    print("🛠️ [PREPROCESSING] Đang gộp file và xử lý ngoại lai...")
    DATA_PATH = "../data/raw/" 
    df = pd.concat([pd.read_csv(os.path.join(DATA_PATH, f)) for f in os.listdir(DATA_PATH) if f.endswith(".csv")], axis=0)
    
    q1, q3 = df['PM2.5'].quantile(0.25), df['PM2.5'].quantile(0.75)
    iqr = q3 - q1
    df = df.loc[~((df['PM2.5'] < q1 - 1.5 * iqr) | (df['PM2.5'] > q3 + 1.5 * iqr))]
    
    print("🛠️ [PREPROCESSING] Mã hóa biến phân loại và Nội suy KNN...")
    for cat_col in df.select_dtypes(include=['object']).columns:
        df[cat_col] = LabelEncoder().fit_transform(df[cat_col])
    df['wd'] = df['wd'].fillna(df['wd'].mode()[0])
    
    na_cols = [c for c in df.columns if df[c].isnull().any() and c != 'wd']
    if na_cols:
        df[na_cols] = KNNImputer(n_neighbors=5).fit_transform(df[na_cols].values)
        
    df.to_csv("../data/airquality_data.csv", index=False)
    print("✅ [PREPROCESSING] Đã lưu file sạch tại ../data/airquality_data.csv")

if __name__ == "__main__":
    preprocess_raw_data()