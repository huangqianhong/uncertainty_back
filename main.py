# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import minio
import os
from datetime import timedelta

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')



    # client = minio.Minio(
    #     endpoint='169.254.29.96:9000',
    #     access_key='minioadmin',
    #     secret_key='minioadmin',
    #     secure=False
    # )

    # if client.bucket_exists("hqh"):
    #     path_dir = "F:\\paper\\pythonProject\\file\\"
    #     file_name_list = os.listdir(path_dir)
    #     for file_name in file_name_list:
    #         path_file = path_dir +file_name
    #         with open(path_file ,"rb") as file_data:
    #             bytes_length = os.path.getsize(path_file)
    #             client.put_object("hqh", file_name, file_data, bytes_length)
    #             url = client.presigned_get_object("hqh", file_name)
    #             print(url)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

    # 设置要列出的桶的名称和文件夹路径
    # bucket_name = 'hqh'
    # folder_path = 'ti'
    #
    # # 使用 list_objects 方法列出桶中指定文件夹的所有对象
    # objects = client.list_objects(bucket_name, prefix=folder_path, recursive=True)
    #
    # # 遍历所有对象并打印它们的名称
    # for obj in objects:
    #     print(obj.object_name)
    #
    #     presigned_url = client.presigned_get_object(bucket_name, object_name=obj.object_name,
    #                                                expires=timedelta(days=7))
    #     print(f"Presigned URL for {obj.object_name}: {presigned_url}")
    #
    #
    # print("===")
    path =r'C:\Users\hqh\Desktop\czs\1_cd.sgems'
    new_path = r'C:\Users\hqh\Desktop\czs\new_01\1_cd.sgems'
    with open(path,'r') as f:
        all = f.readlines()
        with open(new_path,"w+") as data:
            for i in range(0,6):
                data.write(all[i])
            for i in range(6,len(all)):
                ttt = all[i].strip().split(" ")
                for j in range(0,len(ttt)):
                    if j == 3 and ttt[j] == str(255.0):
                        data.write(str(1.0))
                    elif j == 2:
                        data.write(str(int(ttt[j])+1)+" ")
                    else:
                        data.write(ttt[j]+" ")
                data.write("\n")


