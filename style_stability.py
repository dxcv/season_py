import time
import os
import pandas as pd
import argparse
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from WindPy import w
w.start()
from season_label import SeasonLabel
from sql import sql_oracle

_season_label_obj = SeasonLabel()
_cu_wind = sql_oracle.cu
_cu_pra = sql_oracle.cu_pra_sel

def market_fetch_CBA(stock_index):
    sql1 = '''
        SELECT
          T0.F2_1655 日期,
          T0.F3_1655 AS 复权收盘价
          FROM
            wind.TB_OBJECT_1655 T0
          LEFT JOIN wind.TB_OBJECT_1090 T1 ON T1.F2_1090 = T0.F1_1655
          WHERE
            T1.F16_1090 = '%(index_code)s'
          AND T1.F4_1090 = 'S'
          ORDER BY
            T0.F2_1655
        ''' % {'index_code': stock_index}
    market = pd.DataFrame(_cu_wind.execute(sql1).fetchall(), columns=['日期', '指数收盘价'])
    market.index = market['日期']
    del market['日期']
    market.columns = [stock_index]
    return market

def get_market_wind(code):
    s_date, e_date = '20100101', '20191231'
    df = w.wsd(code, "close", s_date, e_date, "","Currency=CNY", usedf=True)[1]
    df.index = pd.Series(df.index).apply(lambda x: str(x)[:4] + str(x)[5:7] + str(x)[8:10])
    df.columns = [code]
    return df

def gen_market_rbsa():
    ''' 不算利率 13 个指标
    '''
    print('gen index market for rbsa...')

    #标普500
    # SPX  = get_market_wind('SPX.GI')

    #标普中国A股100纯成长总收益指数
    grow_100 = get_market_wind('818100PGTR.CI')

    #标普中国A股100纯价值总收益指数
    value_100 = get_market_wind('818100PVTR.CI')

    #标普中国A股200纯成长总收益指数
    grow_200 = get_market_wind('818200PGTR.CI')

    #标普中国A股200纯价值总收益指数
    value_200 = get_market_wind('818200PVTR.CI')

    #标普中国A股小盘纯成长总收益指数
    grow_small = get_market_wind('818300PGTR.CI')

    #标普中国A股小盘纯价值总收益指数
    value_small = get_market_wind('818300PVTR.CI')

    # 恒生指数
    HSI  = get_market_wind('HSI.HI')

    #se4_4中债-新综合财富指数
    bond_1 = market_fetch_CBA('CBA00111')
    bond_1_3 = market_fetch_CBA('CBA00121')
    bond_3_5 = market_fetch_CBA('CBA00131')
    bond_5_7 = market_fetch_CBA('CBA00141')
    bond_7_10  = market_fetch_CBA('CBA00151')
    bond_10 = market_fetch_CBA('CBA00161')


    r_rate = get_market_wind('M0043808')

    market_rbsa = grow_100.join([value_100, grow_200,
        value_200, grow_small, value_small,
        bond_1, bond_1_3, bond_3_5, bond_5_7,
        bond_7_10, bond_10, HSI, r_rate])
    # print(market_rbsa)
    return market_rbsa

# 初始化一些公用基金指标数据
MARKET_RBSA = gen_market_rbsa()

def gen_time_list(s_date, e_date):
    t_list = _season_label_obj.gen_time_list(s_date, e_date, s_type='weekly', pre_flag=True)
    return t_list

def get_market_rbsa(t_df):
    ''' 用时间dt截取 rbsa
    '''
    # rbsa = MARKET_RBSA[(MARKET_RBSA.index >= s_date) & (MARKET_RBSA.index <= e_date)]
    t_df.set_index(['日期'], drop=True, inplace=True)
    # rbsa = MARKET_RBSA.join(t_df, how='inner')
    rbsa = t_df.join(MARKET_RBSA)
    # print(rbsa)
    return rbsa

def get_fund_value(code, t_df):
    print('get fund value...')
    sql = f'''
        select
        f13_1101 as 截止日期, f21_1101 as 复权单位净值 
        from
        wind.tb_object_1101
        left join wind.tb_object_1090
        on f2_1090 = f14_1101
        where 
        F16_1090= '{code}'
        order by f13_1101
        '''
    sql_rst = _cu_wind.execute(sql).fetchall()
    if not sql_rst:
        return
    fund_price = pd.DataFrame(sql_rst, columns=['日期', '复权单位净值'])
    fund_price = pd.merge(t_df, fund_price, on=['日期'], how='outer')
    fund_price.fillna(method='ffill', inplace=True)
    fund_price = pd.merge(t_df, fund_price, on=['日期'], how='left')
    fund_price.set_index(['日期'], inplace=True)
    return fund_price

def compute_r_2(df):
    df_big = df
    df_big=df_big.dropna(axis=0, how='any')
    X_big = df_big.drop(['复权单位净值'], 1) #删除列
    y_big = df_big['复权单位净值']

    # rand 数字等于 基准数量+1
    x0 = np.random.rand(14)
    x0 /= sum(x0)
    X = np.mat(X_big)
    Y = np.mat(y_big)

    # 参数多少取决于基准数量
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

            {'type': 'ineq', 'fun': lambda x: 1-x[0]},
            {'type': 'ineq', 'fun': lambda x: 1-x[1]},
            {'type': 'ineq', 'fun': lambda x: 1-x[2]},
            {'type': 'ineq', 'fun': lambda x: 1-x[3]},
            {'type': 'ineq', 'fun': lambda x: 1-x[4]},
            {'type': 'ineq', 'fun': lambda x: 1-x[5]},
            {'type': 'ineq', 'fun': lambda x: 1-x[6]},
            {'type': 'ineq', 'fun': lambda x: 1-x[7]},
            {'type': 'ineq', 'fun': lambda x: 1-x[8]},
            {'type': 'ineq', 'fun': lambda x: 1-x[9]},
            {'type': 'ineq', 'fun': lambda x: 1-x[10]},
            {'type': 'ineq', 'fun': lambda x: 1-x[11]},
            {'type': 'ineq', 'fun': lambda x: 1-x[12]},
            {'type': 'ineq', 'fun': lambda x: 1-x[13]},
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]-1})
    res = minimize(func, x0, method='SLSQP', constraints=cons4)
    R2 = 1 - res.fun / ((np.ravel(y_big).var()) * len(y_big))
    print('R2', R2)
    return R2

def style_stabily(code, start_date, end_date):
    t_list = gen_time_list(start_date, end_date)
    time_df = pd.DataFrame(t_list, columns=['日期'])

    market_rbsa = get_market_rbsa(time_df)
    fund_price_df = get_fund_value(code, time_df)
    input_df = fund_price_df.join(market_rbsa, how='inner')
    input_df_pct = input_df.pct_change()
    r2 = compute_r_2(input_df_pct)
    return r2

def main_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='start', default='20170101', help='start date')
    parser.add_argument('-e', dest='end', default='20170630', help='end date')
    parser.add_argument('-c', dest='code', default='110011', help='fund code')
    args = parser.parse_args()
    style_stabily(args.code, args.start, args.end)
    return

if __name__ == '__main__':
    main_run()
