#encoding=utf-8
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import time

import logging
logging.getLogger().setLevel(logging.INFO)

from time_tool import Time_tool
t = Time_tool()

from sql import sql_oracle
'''
import my_alpha
from my_alpha import compute_jinzhi_pct_change, get_jinzhi, get_trade_dates_weekly
from sql import sql_oracle
'''

# 暂时不清楚market_RBSA的标准如何选取

def market_fetch(stock_index):
    """
    功能
    --------
    得到某A股指数的全部收盘价数据

    参数
    --------
    stock_index:指数代码，字符格式，如'000300'，不带后面的后缀

    返回值
    --------
    返回一个dataframe,只有一列，是指数收盘价数据，列名为指数代码，index是日期

    参看
    --------
    无关联函数

    示例
    --------
    >>>market_fetch('000300').head()
            000300
    日期
    20020104  1316.455
    20020107  1302.084
    20020108  1292.714
    20020109  1272.645
    20020110  1281.261
    """
    sql1 = """
    SELECT
      T0.F2_1425 日期,
      T0.F7_1425 AS 复权收盘价
      FROM
        wind.TB_OBJECT_1425 T0
      LEFT JOIN wind.TB_OBJECT_1090 T1 ON T1.F2_1090 = T0.F1_1425
      WHERE
        T1.F16_1090 = '%(index_code)s'
      AND T1.F4_1090 = 'S'
      ORDER BY
        T0.F2_1425
    """ % {'index_code': stock_index}
    market = pd.DataFrame(sql_oracle.cu.execute(sql1).fetchall(), columns=['日期', '指数收盘价'])
    market.index = market['日期']
    del market['日期']
    market.columns = [stock_index]
    return market

def get_zhong_zheng_800(trade_dates):
    # 中证800
    market2 = market_fetch('000906')
    market1 = market_fetch('000001')
    market1 = market1[market1.index < '20050105']
    a = market2[market2.index == '20050104'].iloc[0, 0] / market1[market1.index== '20050104'].iloc[-1, 0]
    market1['000906'] = market1['000001'] * a
    index = market1[market1.index == '20050104'].index.tolist()[0]
    market_800 = pd.concat([market1[['000906']] , market2], axis=0).drop([index])
    # market_800_in_trade_dates = market_800[[x for x in trade_dates], :]
    # 中正指标按trade_dates取样,另外进行排序处理
    tmp_market = market_800.loc[trade_dates, :]
    tmp_market['riqi'] = tmp_market.index
    tmp_market = tmp_market.sort_values(by='riqi', ascending=True) #从前到后
    del tmp_market['riqi']
    return tmp_market
    
def compute_pct_change(df):
    """计算收益率，df的index为日期，单列为净值
    注意: 数据按时间是有序，计算才有意义
    """
    df['收益率'] = df.iloc[:, 0].pct_change()
    return pd.DataFrame(df['收益率'])

'''
def RBSA(year_lag):
    """
    功能
    --------
    计算近几年的RBSA回归结果，要先定义好net1和codes，net1是需要基金的基金列表，其中有成绩年限，codes为net1中的基金代码

    参数
    --------
    year_lag:整数，如1,2,3

    返回值
    --------
    返回一个dateframe，其中包含每一只基金回归的R2和回归系数

    参看
    --------
    无关联函数

    示例
    --------
    >>>net1 = net0[net0.二级分类.isin(['标准配置型','可转债型','环球股票','普通债券型','股票型','纯债型','灵活配置型', '激进配置型','激进债券型',
    '保守配置型', '沪港深股票型','沪港深配置型',' 纯债型'])]
    >>>codes = net1['基金代码']
    >>>RBSA(1).head()
   近1年回归系数（标普A股100纯成长）  近1年回归系数（标普A股100纯价值）     ...    近1年R^2    基金代码
        0         1.152305e-01             0.283733                         ...    0.896055     000001
        1         0.000000e+00             0.169836                         ...    0.734744     000003
        2         6.484767e-19             0.019776                         ...    0.463100     000005
        3         3.230120e-17             0.000000                         ...    0.959403     000008
        4         3.147889e-01             0.233594                         ...    0.889642     000011
     """
    end = get_friday(end_date)
    start = date_gen(years=year_lag, end=end_date)
    col = ["近" + str(year_lag) + "年回归系数（标普A股100纯成长）"
    ,"近" + str(year_lag) + "年回归系数（标普A股100纯价值）"
    , "近" + str(year_lag) + "年回归系数（标普A股200纯成长）"
    , "近" + str(year_lag) + "年回归系数（标普A股200纯价值）"
    , "近" + str(year_lag) + "年回归系数（标普A股小盘纯成长）"
    ,"近" + str(year_lag) + "年回归系数（标普A股小盘纯价值）"
    ,"近" + str(year_lag) + "年回归系数（中债-新综合财富(1年以下)）"
    ,"近" + str(year_lag) + "年回归系数（中债-新综合财富(1-3年)）"
    ,"近" + str(year_lag) + "年回归系数（中债-新综合财富(3-5年)）"
    ,"近" + str(year_lag) + "年回归系数（中债-新综合财富(5-7年)）"
    , "近" + str(year_lag) + "年回归系数（中债-新综合财富(7-10年)）"
    , "近" + str(year_lag) + "年回归系数（中债-新综合财富(10年以上)）"
    , "近" + str(year_lag) + "年回归系数（恒生指数）"
    , "近" + str(year_lag) + "年回归系数（标普500）"
    , "近" + str(year_lag) + "年回归系数（一年定存利率）"
    ]
    col2 = ['基金代码',"近" + str(year_lag) + "年R^2"]
    R2_result =pd.DataFrame(columns=col2.extend(col))
    for i in codes:
        global net1
        index1 = net1[net1.基金代码 == i].index.tolist()[0]
        if net1.loc[index1,'成立年限'] < year_lag:
            pass
        else:
            dates = get_weekly_yeild(i).loc[start:end]
            dates = dates.join(market_RBSA)
            dates = dates.drop(dates.index[0])
            df_big = dates
            df_big=df_big.dropna(axis=0,how='any')
            X_big = df_big.drop(str(i),1)
            y_big = df_big[str(i)]
            #进行带约束条件的回归
            x0 = np.random.rand(15)
            x0 /= sum(x0)
            X = np.mat(X_big)
            Y = np.mat(y_big)
            func = lambda x: ((Y.T - X * (np.mat(x).T)).T * (Y.T - X * (np.mat(x).T))).sum()
            cons4 = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: x[3]},
            {'type': 'ineq', 'fun': lambda x: x[4]},
            {'type': 'ineq', 'fun': lambda x: x[5]},
            {'type': 'ineq', 'fun': lambda x: x[6]},
            {'type': 'ineq', 'fun': lambda x: x[7]},
            {'type': 'ineq', 'fun': lambda x: x[8]},
            {'type': 'ineq', 'fun': lambda x: x[9]},
            {'type': 'ineq', 'fun': lambda x: x[10]},
            {'type': 'ineq', 'fun': lambda x: x[11]},
            {'type': 'ineq', 'fun': lambda x: x[12]},
            {'type': 'ineq', 'fun': lambda x: x[13]},
            {'type': 'ineq', 'fun': lambda x: x[14]},

            {'type': 'ineq', 'fun': lambda x: 1-x[0]},
            {'type': 'ineq', 'fun': lambda x: 1-x[1]},
            {'type': 'ineq', 'fun': lambda x: 1-x[2]},
            {'type': 'ineq', 'fun': lambda x: 1-x[3]},
            {'type': 'ineq', 'fun': lambda x: 1-x[4]},
            {'type': 'ineq', 'fun': lambda x: 1-x[5]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[6]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[7]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[8]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[9]},
            {'type': 'ineq', 'fun': lambda x: 1- x[10]},
            {'type': 'ineq', 'fun': lambda x: 1- x[11]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[12]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[13]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[14]},
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]-1})
            res = minimize(func, x0, method='SLSQP', constraints=cons4)
            R2 = 1 - res.fun / ((np.ravel(y_big).var()) * len(y_big))
            if R2 <0:
                R2 = 0
            res.x[res.x < 0] = 0
            df3 = pd.DataFrame(res.x)
            df3 = df3.T
            df3.columns = col
            df3["近" + str(year_lag) + "年R^2"] = R2
            df3['基金代码'] = i
            R2_result = R2_result.append(df3,ignore_index=True)
    return R2_result

def get_weekly_rate(code, trade_dates):
    weekly_rate = compute_jinzhi_pct_change(get_jinzhi(trade_dates, code))
    return weekly_rate

def compute_r_square(df):
    i = '收益率'
    df_big = df
    df_big=df_big.dropna(axis=0,how='any')
    X_big = df_big.drop(str(i),1) #删除列
    y_big = df_big[str(i)]
    #进行带约束条件的回归
    import pdb; pdb.set_trace()
    x0 = np.random.rand(15)
    x0 /= sum(x0)
    X = np.mat(X_big)
    Y = np.mat(y_big)
    import pdb; pdb.set_trace()
    func = lambda x: ((Y.T - X * (np.mat(x).T)).T * (Y.T - X * (np.mat(x).T))).sum()
    cons4 = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: x[3]},
            {'type': 'ineq', 'fun': lambda x: x[4]},
            {'type': 'ineq', 'fun': lambda x: x[5]},
            {'type': 'ineq', 'fun': lambda x: x[6]},
            {'type': 'ineq', 'fun': lambda x: x[7]},
            {'type': 'ineq', 'fun': lambda x: x[8]},
            {'type': 'ineq', 'fun': lambda x: x[9]},
            {'type': 'ineq', 'fun': lambda x: x[10]},
            {'type': 'ineq', 'fun': lambda x: x[11]},
            {'type': 'ineq', 'fun': lambda x: x[12]},
            {'type': 'ineq', 'fun': lambda x: x[13]},
            {'type': 'ineq', 'fun': lambda x: x[14]},

            {'type': 'ineq', 'fun': lambda x: 1-x[0]},
            {'type': 'ineq', 'fun': lambda x: 1-x[1]},
            {'type': 'ineq', 'fun': lambda x: 1-x[2]},
            {'type': 'ineq', 'fun': lambda x: 1-x[3]},
            {'type': 'ineq', 'fun': lambda x: 1-x[4]},
            {'type': 'ineq', 'fun': lambda x: 1-x[5]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[6]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[7]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[8]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[9]},
            {'type': 'ineq', 'fun': lambda x: 1- x[10]},
            {'type': 'ineq', 'fun': lambda x: 1- x[11]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[12]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[13]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[14]},
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]-1})
    res = minimize(func, x0, method='SLSQP', constraints=cons4)
    R2 = 1 - res.fun / ((np.ravel(y_big).var()) * len(y_big))
    import pdb; pdb.set_trace()
    if R2 <0:
        R2 = 0
    res.x[res.x < 0] = 0
    df3 = pd.DataFrame(res.x)
    df3 = df3.T
    df3.columns = col
    df3["近" + str(year_lag) + "年R^2"] = R2
    df3['基金代码'] = i
    R2_result = R2_result.append(df3,ignore_index=True)
    return

def run_RBSA(code, start, end):
    return

def main_run():
    code, start, end = ('001688', '20180211', '20190527')
    trade_dates = get_trade_dates_weekly(start, end)
    market_zhongzheng_800 = get_zhong_zheng_800(trade_dates)
    market_zz_800_pct = compute_pct_change(market_zhongzheng_800)
    market_zz_800_pct.columns = ['中正800收益率']
    weekly_rate = get_weekly_rate(code, trade_dates)
    # 按日期整合数据,中正_800收益率和单支股票收益率都是按trade_dates取样
    t_weekly_rate = weekly_rate.join(market_zz_800_pct)
    t_weekly_rate.drop(t_weekly_rate.index[0])
    compute_r_square(t_weekly_rate)
'''

def compute_r_square(df):
    df_big = df
    df_big=df_big.dropna(axis=0,how='any')
    X = np.array(df['中正800收益率']).reshape(df['中正800收益率'].shape[0], 1)
    Y = df['周收益率']
    regr = LinearRegression()
    regr.fit(X, Y)
    y_test = regr.predict(X)
    # mse = metrics.mean_absolute_error(Y, y_test)
    # rmse = np.sqrt(mse)
    # Sum of Squares forRegression 回归平方
    ssr = ((y_test - Y.mean()) ** 2).sum()
    # Sum of Squaresfor Total 总偏差平方和
    sst = ((Y - Y.mean()) ** 2).sum()
    r2 = ssr/sst
    return r2
    # X_big = df_big.drop(str(i),1)
    # y_big = df_big[str(i)]
    #进行带约束条件的回归
    """
    X_big = pd.DataFrame(df_big['中正800收益率'])
    y_big = df_big['周收益率']
    # x0 = np.random.rand(1)
    x0 = np.random.rand()
    # x0 /= sum(x0)
    X = np.mat(X_big)
    Y = np.mat(y_big)
    import pdb; pdb.set_trace()
    cons4 = ({'type': 'ineq', 'fun': lambda x: x[0]},
             {'type': 'ineq', 'fun': lambda x: 1-x[0]},
             {'type': 'eq', 'fun': lambda x: x[0]-1})
    # func = lambda x: ((Y.T - X * (np.mat(x).T)).T * (Y.T - X * (np.mat(x).T))).sum()
    func = lambda x: ((Y.T - X * (np.mat(x).T)).T * (Y.T - X * (np.mat(x).T))).sum()
    import pdb; pdb.set_trace()
    res = minimize(func, x0, method='SLSQP', constraints=cons4)
    R2 = 1 - res.fun / ((np.ravel(y_big).var()) * len(y_big))
    import pdb; pdb.set_trace()
    """

def main_test():
    s_time = time.time()
    for i in range(20):
        code, start, end = ('001688', '20160211', '20190527')
        trade_dates = t.get_week_trade_days(start, end)
        pre_day = t.get_pre_week_trade_day(start)
        time_list = list(trade_dates['日期'])
        time_list.insert(0, pre_day)
        trade_dates = time_list
        logging.info('trade_dates: {}'.format(trade_dates))
        
        market_zhongzheng_800 = get_zhong_zheng_800(trade_dates)
        market_zz_800_pct = compute_pct_change(market_zhongzheng_800)
        market_zz_800_pct.columns = ['中正800收益率']  
        # weekly_rate = get_week_return()
        
        from week_return import Week_return
        wr = Week_return()
        
        weekly_rate = wr.get_week_value_fund(code, time_list)
        # get_week_value_fund 有多余的数据需要过滤
        # weekly_rate.columns = ['日期', '复权单位净值', '上周复权单位净值', '周收益率']
        weekly_rate.set_index('日期', inplace=True)
        del weekly_rate['复权单位净值']
        del weekly_rate['上周复权单位净值']
        t_weekly_rate = weekly_rate.join(market_zz_800_pct)
        #   t_weekly_rate.drop(t_weekly_rate.index[0])  估计是想删掉第一行为0的
        rst = compute_r_square(t_weekly_rate)
    print(rst)
    print('time cost: {}'.format(time.time()-s_time))

if __name__ == '__main__':
    print('hello world')
    # main_run()
    main_test()
