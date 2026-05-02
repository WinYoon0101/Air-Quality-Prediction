import boto3
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME

def upload_to_datalake():
    print("🔄 [INGESTION] Đang đẩy dữ liệu lên MinIO Bronze...")
    s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
    
    if BUCKET_NAME not in [b['Name'] for b in s3.list_buckets()['Buckets']]:
        s3.create_bucket(Bucket=BUCKET_NAME)

    s3.upload_file("../data/airquality_data.csv", BUCKET_NAME, "bronze/airquality_data.csv")
    print(f"✅ Đã tải lên Data Lake thành công.")

if __name__ == "__main__":
    upload_to_datalake()