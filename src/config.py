# Cấu hình MinIO (S3)
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME = "air-quality-lake"

# Đường dẫn Data Lake
BRONZE_PATH = f"s3a://{BUCKET_NAME}/bronze/airquality_data.csv"
GOLD_PATH = f"s3a://{BUCKET_NAME}/gold/features.parquet"


# Cấu hình Cassandra
CASSANDRA_HOST = "localhost"
# CASSANDRA_KEYSPACE = "air_quality"
# CASSANDRA_TABLE = "pm25_forecast"
#(Tài khoản/Mật khẩu mặc định thường là admin / admin)