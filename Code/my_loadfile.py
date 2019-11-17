import pandas as pd
import pickle
import numpy as np

def LoadTrains():
    # 训练集
    train_path = r'./security_train.csv'
    labels=[]
    apis=[]
    data = pd.read_csv(train_path)
    # 分组成文件集
    files = data.groupby('file_id')
    for file_id, files_info in files:
        newfile_info = files_info.sort_values(['tid', 'index'])
    #     print(newfile_info)
        api_sequence = ' '.join(newfile_info['api'])
        # print(len(api_sequence))
    #     print(files_info['label'].values[0])
        labels.append(files_info['label'].values[0])
        apis.append(api_sequence)
    # 格式[5,6,...]
    # print(fileId)
    # 格式['getaddr loadexe loaddll','...',...]
#     print(apis)
    with open('my_security_train1.pkl', 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(apis, f)
#     print(apis)
    return labels, apis
    # 确定API序列的最大长度
    # max = 0
    # for i in apis:
    #     # print(len(i[0]))
    #     if max<len(i):
    #         max = len(i)
    #     print(max)

def LoadTests():
    # c测试集
    test_path = r'./security_test1.csv'
    apis=[]
    test_fileId = []
    data = pd.read_csv(test_path)
    # 分组成文件集
    files = data.groupby('file_id')
    for file_id, files_info in files:
        newfile_info = files_info.sort_values(['tid', 'index'])
    #     print(newfile_info)
    #     print(file_id)
        api_sequence = ' '.join(newfile_info['api'])
        test_fileId.append(file_id)
        apis.append(api_sequence)
    # 格式['getaddr loadexe loaddll','...',...]
#     print(apis)
    with open('my_security_test1.pkl', 'wb') as f:
        pickle.dump(test_fileId, f)
        pickle.dump(apis, f)
    return test_fileId , apis

if __name__ == '__main__':
    train_labels = []
    train_apis = []
    test_files = []
    test_apis =[]
    train_labels, train_apis = LoadTrains()
#     print(train_labels)
#     print(train_apis)
    test_files, test_apis = LoadTests()
#     print(test_apis)