import numpy as np
import uuid
import base64

# 文件夹存在的名字
file_dir_name = "F:\\paper\\pythonProject\\file\\"
user_name = "hqh"
# 展示Flask如何读取服务器本地图片, 并返回图片流给前端显示的例子
def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """

    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

def get_uuid():
    get_timestamp_uuid = uuid.uuid1()  # 根据 时间戳生成 uuid , 保证全球唯一
    return get_timestamp_uuid


def read_file_to_arr_2d(path_name):
    """
    将文件数据转换成二维numpy数据
    :param path_name:
    :return:
    """
    final_list = []
    with open(path_name) as files:
        arr = files.readlines()
        x_str, y_str, z_str = arr[0].split(" ")
        for i in range(3, len(arr)):
            final_list.append(float(arr[i]))
        final_arr = np.array(final_list)
    x, y = int(x_str), int(y_str)
    final_arr = final_arr.reshape(y, x)
    # final_arr = final_arr.reshape(x, y)
    return final_arr, y, x

