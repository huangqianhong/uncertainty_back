# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import minio
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')



    client = minio.Minio(
        endpoint='169.254.29.96:9000',
        access_key='minioadmin',
        secret_key='minioadmin',
        secure=False
    )

    if client.bucket_exists("hqh"):
        path_dir = "F:\\paper\\pythonProject\\file\\"
        file_name_list = os.listdir(path_dir)
        for file_name in file_name_list:
            path_file = path_dir +file_name
            with open(path_file ,"rb") as file_data:
                bytes_length = os.path.getsize(path_file)
                client.put_object("hqh", file_name, file_data, bytes_length)
                url = client.presigned_get_object("hqh", file_name)
                print(url)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
