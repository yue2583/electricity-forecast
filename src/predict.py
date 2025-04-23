import pandas as pd
from common import data_preprocess
from train import feature_engineering
import matplotlib.ticker as mick
from sklearn.metrics import mean_absolute_error
import joblib
from log import log
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


def fill_lack_features(data):
    month_features = [f'month_{i}' for i in range(1, 13)]
    for feature in month_features:
        if feature not in data.columns:
            data[feature] = False

    order = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
             'month_9', 'month_10', 'month_11', 'month_12', 'weekday_1', 'weekday_2', 'weekday_3',
             'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7', 'hour_0', 'hour_1', 'hour_2',
             'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10',
             'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
             'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', '前1小时负载', '前2小时负载',
             '前3小时负载', '前1天同时刻负载', '前2天同时刻负载', '前3天同时刻负载']
    return data[order]


def prediction_plot(data):
    """
    绘制时间与预测负荷折线图，时间与真实负荷折线图，展示预测效果
    :param data: 数据一共有三列：时间、真实值、预测值
    :return:
    """
    # 绘制在新数据下
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot()
    # 绘制时间与真实负荷的折线图
    ax.plot(data['时间'], data['真实值'], label='真实值')
    # 绘制时间与预测负荷的折线图
    ax.plot(data['时间'], data['预测值'], label='预测值')
    ax.set_ylabel('负荷')
    ax.set_title('预测负荷以及真实负荷的折线图')
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('../data/fig/预测效果.png')


def do_predict(data):
    data = feature_engineering(data)
    data = data[pd.to_datetime(data['time']) > pd.to_datetime('2015-07-31 23:59:59')]
    x = fill_lack_features(data)
    model = joblib.load('../model/xgb.pkl')
    y = data.iloc[:, 1]
    y_pred = model.predict(x)

    time = data['time'].reset_index(drop=True)
    true_val = y.reset_index(drop=True)
    pred_val = pd.Series(y_pred)

    result = pd.concat([time, true_val, pred_val], axis=1)
    result.columns = ['时间', '真实值', '预测值']
    mae_score = mean_absolute_error(result['真实值'], result['预测值'])
    log.info(f"模型对新数据进行预测的平均绝对误差：{mae_score}")  # 56.62082605471557
    log.info(result)
    return result


if __name__ == '__main__':
    _data = data_preprocess('../data/test.csv')
    _data = do_predict(_data)
    prediction_plot(_data)
