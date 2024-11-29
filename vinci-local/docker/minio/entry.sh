#!/bin/sh

minio server /data --console-address :9001 &

# 等待 MinIO 服务（9000 端口）启动
while ! curl -s http://localhost:9000/minio/health/live; do
  echo "等待 MinIO 服务启动..."
  sleep 3
done

mc alias set local_minio http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

# 创建bucket
if mc ls local_minio/vinci > /dev/null 2>&1; then
  echo "Bucket 'vinci' 已经存在，跳过创建步骤"
else
  mc mb local_minio/vinci
fi

echo "设置 bucket 'vinci' 为公共访问权限"
mc anonymous set public local_minio/vinci

tail -f /dev/null
