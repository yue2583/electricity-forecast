import pandas as pd
from log import log

def data_preprocess(path):
    """
    1.指定路径读取数据
    2.时间格式化，转为2024-12-20 09:00:00这种格式
    3.按时间升序排序
    4.去重
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    data.sort_values(by=['time'], inplace=True)
    data.drop_duplicates(inplace=True)
    return data


if __name__ == '__main__':
    data = data_preprocess('../data/train.csv')
    log.info(data.head())
