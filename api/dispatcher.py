import json

from flask import Flask, request,send_file
from flask_cors import  cross_origin,CORS

from function_model.mph import mph_dir, mph_file
from common.file_util import  file_dir_name
import minio
"""
Flask 分发api文件
"""
app = Flask(__name__)
CORS(app,resources=r'/*')

@app.route('/test_mph_dir', methods=['POST'])
@cross_origin()
def dispatch_mph_path_dir():
    """
    入口函数 训练图像和文件下的数据进行特征匹配
    """
    data = request.get_json()
    ti_path = data['ti_path']  # 训练图像路径
    path_dir = data['path_dir']  # 实现数据文件夹
    mph_dir(ti_path, path_dir)


@app.route("/test_mph_file", methods=['post'])
@cross_origin()
def dispatch_mph_path_file():
    """
    入口函数 训练图像和单个实现数据下的对比
    :return:
    """
    data = request.get_json()
    ti_path = data['ti_path']
    realization_path = data['real_path']
    dis,file_name_list  = mph_file(ti_path, realization_path)
    print("========")
    # 我们从file_name_list里面获取名字，通过输入流获取文件内容
    urls = []
    for file_name in file_name_list:
        # url = send_file(file_dir_name + file_name, mimetype='image/png')
        # if(len(urls) >= 1): continue
        urls.append(file_name)
    # 将文件信息传递到浏览器
    # return json.dumps({'status':"ok","urls":json.dumps(urls),"dis":json.dumps(dis)})
    return json.dumps({'status':"ok","urls":urls,"dis":json.dumps(dis)})

#

if __name__ == '__main__':
    app.run(port=5055)
    # app.config['DEBUG'] = True
