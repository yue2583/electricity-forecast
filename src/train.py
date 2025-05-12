import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from log import log
import joblib
from common import data_preprocess

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


class PowerLoadModel:
    def __init__(self, file_path):
        self.data_source = data_preprocess(file_path)


def ana_data(data: pd.DataFrame):
    """
    :param data: 数据源
    :return:
    """
    data = data.copy(deep=True)

    fig = plt.figure(figsize=(20, 30), dpi=100)

    log.info('开始绘制负荷分布直方图')
    sub1 = fig.add_subplot(411)
    sub1.hist(data['power_load'], bins=100)
    sub1.set_title("负荷分布直方图")

    log.info('开始绘制按月分组折线图')
    sub2 = fig.add_subplot(412)
    data['month'] = data['time'].str[5:7]
    month_avg = data.groupby('month', as_index=False)['power_load'].mean()
    sub2.plot(month_avg['month'], month_avg['power_load'])
    sub2.set_title("按月分组折线图")

    log.info('开始绘制按星期分组折线图')
    sub3 = fig.add_subplot(413)
    data['weekday'] = data['time'].apply(lambda x: pd.to_datetime(x).weekday() + 1)
    weekday_avg = data.groupby('weekday', as_index=False)['power_load'].mean()
    sub3.plot(weekday_avg['weekday'], weekday_avg['power_load'])
    sub3.set_title('按星期分组')

    log.info('开始绘制按小时分组折线图')
    sub4 = fig.add_subplot(414)
    data['hour'] = data['time'].str[11:13]
    hour_avg = data.groupby('hour', as_index=False)['power_load'].mean()
    sub4.plot(hour_avg['hour'], hour_avg['power_load'])
    sub4.set_title('按小时分组折线图')
    plt.savefig('../data/fig/analyze.png')


def feature_engineering(data: pd.DataFrame):
    """
    增加月份，周几，小时特征
    增加前 n 小时的负载
    :param data:
    :return:
    """
    result = data.copy(deep=True)

    log.info('提取月份特征')
    result['month'] = data['time'].str[5:7]
    month_encoding = pd.get_dummies(result['month'])
    month_encoding.columns = ['month_' + str(int(i)) for i in month_encoding.columns]

    log.info('提取周几特征')
    result['weekday'] = data['time'].apply(lambda x: pd.to_datetime(x).weekday() + 1)
    weekday_encoding = pd.get_dummies(result['weekday'])
    weekday_encoding.columns = ['weekday_' + str(int(i)) for i in weekday_encoding.columns]

    log.info('提取小时特征')
    result['hour'] = data['time'].str[11:13]
    hour_encoding = pd.get_dummies(result['hour'])
    hour_encoding.columns = ['hour_' + str(int(i)) for i in hour_encoding.columns]

    log.info('提取前n个小时负载')
    windows_size = 3
    last_n_hour_load_list = [data['power_load'].shift(i) for i in range(1, windows_size + 1)]
    last_n_hour_load_data = pd.concat(last_n_hour_load_list, axis=1)
    last_n_hour_load_data.columns = [f'前{i}小时负载' for i in range(1, windows_size + 1)]

    log.info('提取前n天同时刻负载')
    time_load_dict = data.set_index('time')['power_load'].to_dict()
    last_day_time_list = []
    for i in range(1, windows_size + 1):
        item = data['time'].apply(lambda x: (pd.to_datetime(x) - pd.to_timedelta(f'{i}d'))
                                  .strftime('%Y-%m-%d %H:%M:%S'))
        item = item.apply(lambda x: time_load_dict.get(x))
        last_day_time_list.append(item)

    last_day_time_data = pd.concat(last_day_time_list, axis=1)
    last_day_time_data.columns = [f'前{i}天同时刻负载' for i in range(1, windows_size + 1)]

    log.info('特征合并')
    result = pd.concat([data, month_encoding, weekday_encoding, hour_encoding,
                        last_n_hour_load_data, last_day_time_data, ], axis=1)
    result = result.dropna()
    return result


def model_train(data):
    log.info(data.columns)

    x = data.iloc[:, 2:]
    y = data['power_load']

    log.info('切分训练集，测试集')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    xgb = XGBRegressor()
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.01]
    }
    grid_cv = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5)
    log.info('网格搜索，交叉验证')
    grid_cv.fit(x_train, y_train)
    log.info(grid_cv.best_params_)  # {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
    log.info(grid_cv.best_score_)  # 0.8681287181558108

    xgb = XGBRegressor(**grid_cv.best_params_)
    xgb.fit(x_train, y_train)
    y_pred_train = xgb.predict(x_train)
    y_pred_test = xgb.predict(x_test)
    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_pred_train)
    mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)
    mpe_test = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_test)
    log.info("=========================模型训练完成=============================")
    log.info(f"模型在训练集上的均方误差：{mse_train}")
    log.info(f"模型在训练集上的平均绝对误差：{mae_train}")
    log.info(f"模型在测试集上的均方误差：{mse_test}")
    log.info(f"模型在测试集上的平均绝对误差：{mae_test}")
    log.info(f"模型在测试集上的平均绝对百分比误差：{mpe_test}")

    # 5.模型保存
    joblib.dump(xgb, '../model/xgb.pkl')


if __name__ == '__main__':
    _data = PowerLoadModel("../data/train.csv").data_source
    ana_data(_data)
    # _data = feature_engineering(_data)
    # model_train(_data)
