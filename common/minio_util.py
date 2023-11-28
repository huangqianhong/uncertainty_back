from minio import Minio
import os


def set_minio_client():
    """
      设置MinIO服务器的访问信息
    :return:
    """
    minio_client = Minio(endpoint='169.254.29.96:9000',
                         access_key='minioadmin',
                         secret_key='minioadmin',
                         secure=False)  # 如果使用https，请将secure设置为True
    return minio_client


def list_minio_bugckets_by_name(bucket_name):
    """
     minio文件根据bucket_name获取桶中所有的对象
    :param bucket_name:
    :return:
    """
    minio_client = set_minio_client()
    objects = minio_client.list_objects(bucket_name)
    url_list = []
    # 遍历所有对象并打印它们的名称
    for obj in objects:
        url_list.append(obj.object_name)
        print(obj.object_name)

    minio_client._http.clear()
    return url_list


def upload_files_minio_bucket_by_dir(path_dir, bucket_name):
    """

    :return:
    """
    client = set_minio_client()

    # 检查桶是否存在，如果不存在，则创建
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")

    url_list = []

    if client.bucket_exists(bucket_name):
        file_name_list = os.listdir(path_dir)
        for file_name in file_name_list:
            path_file = os.path.join(path_dir, file_name)
            with open(path_file, "rb") as file_data:
                bytes_length = os.path.getsize(path_file)
                client.put_object(bucket_name, file_name, file_data, bytes_length)
                url = client.presigned_get_object(bucket_name, file_name)
                print(url)
                url_list.append(url)
    client._http.clear()
    return url_list


def upload_files_minio_bucket_by_file(path_file, file_name, bucket_name):
    """

    :return:
    """
    client = set_minio_client()

    # 检查桶是否存在，如果不存在，则创建
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")

    url = ''

    if client.bucket_exists(bucket_name):
        with open(path_file, "rb") as file_data:
            bytes_length = os.path.getsize(path_file)
            client.put_object(bucket_name, file_name, file_data, bytes_length)
            url = client.presigned_get_object(bucket_name, file_name)
            print(url)
    else:
        print("bucket not exists")
    client._http.clear()
    return url


if __name__ == '__main__':
    endpoint = '169.254.29.96:9000',
    access_key = 'minioadmin',
    secret_key = 'minioadmin',
    secure = False
    path_dir = "F:\\paper\\pythonProject\\file\\"
    bucket_name = "hqh"
    upload_files_minio_bucket_by_dir(path_dir, bucket_name)
