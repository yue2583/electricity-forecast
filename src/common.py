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
    data = data.sort_values(by=['time'])
    data = data.drop_duplicates()
    return data


def _test_to_datetime():
    # dt = pd.to_datetime('2024-12-20 09:00:00')
    dt = pd.to_datetime('2013/9/2 3:10')
    # dt = pd.to_datetime(pd.Series(['2013/9/2 3:10']))
    print(type(dt))
    print(dt)
    print(dt.year)
    print(dt.weekday())
    # print(dt.dtype)
    # print(dt.dt.year)
    # print(dt.dt.hour)


if __name__ == '__main__':
    _data = data_preprocess('../data/train.csv')
    log.info(_data.head())
    # _test_to_datetime()
    pass
