import os
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sql import sql_oracle
from time_tool import Time_tool
try:
    from WindPy import w
    w.start()
except:
    w = None

# 季度标签
class SeasonLabel(object):
    def __init__(self):
        self.cu_wind = sql_oracle.cu
        self.cu_pra = sql_oracle.cu_pra_sel
        self.time_tool = Time_tool()
        self.init_funcs()
        self.market = pd.DataFrame()
        self.gen_market()

    def init_funcs(self):
        '''初始化各种外挂方法'''
        classes = [LeverageRatio, StandardDeviation, MaxDrawDown, IntervalProfit, AlphaCategroy, SharpRation, DownStd, InfomationRatio]
        for class_ in classes:
            class_(self).add_func()

    def gen_time_list(self, start_date, end_date, s_type='daily', pre_flag=True):
        '''生成交易日序列
        s_type: daily, 日频; weekly， 周频
        pre_flag: True， 需要计算收益率的标签在取样时需要往前多取一个交易日
        attention: 算法不需要用前一交易日信息,但需要用前一日净值信息来替代, 
        '''
        t_list = []
        if s_type == 'daily':
            t_df = self.time_tool.get_trade_days(start_date, end_date)
            t_list = list(t_df.iloc[:, 0])
            if pre_flag == True:
                pre_day = self.time_tool.get_pre_trade_day(start_date)
                t_list.insert(0, pre_day)
        if s_type == 'weekly':
            t_df = self.time_tool.get_week_trade_days(start_date, end_date)
            t_list = list(t_df.iloc[:, 0])
            if pre_flag == True:
                pre_day = self.time_tool.get_pre_week_trade_day(start_date)
                t_list.insert(0, pre_day)
        return t_list

    def get_fund_price(self, code, start_date='', end_date='', time_list=[]):
        '''计算波动和回测时用到的sql查询方法'''
        if time_list:
            start_date = time_list[0]
            end_date = time_list[-1]
        sql = '''
        select
        f13_1101 as 截止日期, f21_1101 as 复权单位净值 
        from
        wind.tb_object_1101
        left join wind.tb_object_1090
        on f2_1090 = f14_1101
        where 
        F16_1090= '%(code)s'
        and
        F13_1101 >= '%(start_date)s'
        and
        f13_1101 <= '%(end_date)s'
        order by f13_1101
        ''' % {'end_date': end_date, 'code': code, 'start_date': start_date}
        fund_price = pd.DataFrame(self.cu_wind.execute(sql).fetchall(), columns=['日期', '复权单位净值'])
        fund_price.set_index('日期', inplace=True)
        if fund_price.empty:
            raise Exception('empty fund_value. code: {}, start_date: {}, end_date: {}'
                    .format(code, start_date, end_date))
        if time_list:
            # 基金净值数据是按照周频发布
            # 基金成立在时间段中间，导致获取净值数据不足
            # if len(time_list) - fund_price.shape[0] > len(time_list) * 0.1:
            time_list_dt = pd.DataFrame(time_list, columns=['日期'])
            fund_price = pd.merge(time_list_dt, fund_price, on=['日期'], how='outer')
            if fund_price.shape[0] / len(time_list) < 0.9:
                raise Exception('lack of data, timelist>>fund_price.shape[0]')
            fund_price.fillna(method='ffill', inplace=True)
            fund_price = pd.merge(time_list_dt, fund_price, on=['日期'], how='left')
            fund_price.set_index(['日期'], inplace=True)
            # fund_price = fund_price.reindex(time_list)
        return fund_price

    def get_fund_price_biwi(self, code, start_date='', end_date='', time_list=[]):
        '''获取基金基准
        '''
        if time_list:
            t1, t2 = time_list[0], time_list[-1]
        else:
            t1, t2 = start_date, end_date
        sql_code = code + 'BI.WI'
        sql_week_close_bibw = f"""
        select  trade_dt,s_dq_close from wind.chinamutualfundbenchmarkeod 
        where s_info_windcode = '{sql_code}' and (trade_dt >= '{t1}' and trade_dt <='{t2}')
        order by trade_dt 
        """
        sql_res = self.cu_wind.execute(sql_week_close_bibw).fetchall()
        assert sql_res, f'{code}基准查询结果为空,请改变基准'
        df = pd.DataFrame(sql_res, columns=['日期', '收盘价'])
        if df.empty:
            raise Exception('基金基准查询结果为空')
        if df.shape[0] < len(time_list) * 0.9:
            raise Exception('基准数据不足')
        time_list_dt = pd.DataFrame(time_list, columns=['日期'])
        df = pd.merge(time_list_dt, df, on=['日期'], how='outer')
        df.fillna(method='ffill', inplace=True)
        df = pd.merge(time_list_dt, df, on=['日期'], how='left')
        df.set_index(['日期'], inplace=True)
        return df

    def get_ejfl_type(self, code, start_date, end_date):
        '''获取区间内第一个匹配的二级分类类型'''
        end_date = end_date[0: 4] + '1231'
        sql_string = f'''
        select ejfl, rptdate from (
            select ejfl, rptdate from t_fund_classify_his
            where rptdate <= {end_date}
            and cpdm = {code}
            order by rptdate desc
        )
        where rownum=1 
        '''
        sql_rst = self.cu_pra.execute(sql_string).fetchall()
        if sql_rst:
            ejfl, rptdate = sql_rst[0]
        else:
            ejfl, rptdate = None, None
        return ejfl, rptdate

    # 获取所有同类基金
    def get_type_funds(self, fund_type, rptdate):
        sql_string = f'''
        select cpdm from t_fund_classify_his
        where rptdate = '{rptdate}'
        and ejfl = '{fund_type}'
        '''
        sql_rst = self.cu_pra.execute(sql_string).fetchall()
        rst = [i[0] for i in list(sql_rst)]
        return rst

    def get_mean_market(self, fund_type, rptdate, time_list):
        ''' 获取市场同类平均
        fund_type: 二级分类
        rptdate: 报告日期，用来对齐二级分类
        time_list: 用来生成净值序列的时间序列
        '''
        same_type_funds = self.get_type_funds(fund_type, rptdate)
        rst = pd.DataFrame()
        rst['日期'] = time_list
        rst.set_index(['日期'], drop=True, inplace=True)
        # 计算同类平均，净值数据没有的直接跳过
        print('len(same_type_funds: {}'.format(len(same_type_funds)))
        for fund_code in same_type_funds:
            try:
                fund_price = self.get_fund_price(fund_code, time_list=time_list)
            except Exception as err:
                print(err)
                # raise(err)
                continue
            fund_price.columns = [fund_code]
            rst = rst.join(fund_price, on=['日期'], how='left')
        rst['mean'] = rst.mean(axis=1)
        return pd.DataFrame(rst['mean'].pct_change())

    def get_market(self, fund_type, time_list):
        """获取同类基准
         """
        if self.market.empty:
            self.gen_market()
        time_list_dt = pd.DataFrame(time_list, columns=['日期'])
        market_tmp = pd.merge(time_list_dt, self.market, on=['日期'], how='outer')
        market_tmp.fillna(method='ffill', inplace=True)
        market_tmp = pd.merge(time_list_dt, market_tmp, on=['日期'], how='left')
        market_all = pd.DataFrame()
        for i in ['中证800','中证国债','恒生指数','中证综合债','中证短债','中证可转债','MSCI']:
            market_all[str(i)+'收益率'] = market_tmp[str(i)].pct_change()
        if fund_type =='股票型':
            market_type  = market_all['中证800收益率']
        elif fund_type == '激进配置型':
            market_type = market_all['中证800收益率']*0.8+market_all['中证国债收益率']*0.2
        elif fund_type =='标准配置型':
            market_type = market_all['中证800收益率']*0.6+market_all['中证国债收益率']*0.4
        elif fund_type == '保守配置型':
            market_type = market_all['中证800收益率'] * 0.2 + market_all['中证国债收益率'] * 0.8
        elif fund_type == '灵活配置型':
            market_type = market_all['中证800收益率'] * 0.5 + market_all['中证国债收益率'] * 0.5
        elif fund_type == '沪港深股票型':
            market_type = market_all['中证800收益率'] * 0.45 + market_all['中证国债收益率'] * 0.1+market_all['恒生指数收益率']*0.45
        elif fund_type == '沪港深配置型':
            market_type = market_all['中证800收益率'] * 0.35 + market_all['中证国债收益率'] * 0.3 + market_all['恒生指数收益率'] * 0.35
        elif fund_type =='纯债型':
            market_type = market_all['中证综合债收益率']
        elif fund_type == '普通债券型':
            market_type = market_all['中证综合债收益率']*0.9+market_all['中证800收益率'] * 0.1
        elif fund_type == '激进债券型':
            market_type = market_all['中证综合债收益率']*0.8+market_all['中证800收益率'] * 0.2
        elif fund_type =='短债型':
            market_type = market_all['中证短债收益率']
        elif fund_type =='可转债型':
            market_type = market_all['中证可转债收益率']
        elif fund_type =='环球股票':
            market_type = market_all['MSCI收益率']
        else:
            market_type = np.array([])
        # rst = pd.DataFrame(market_type, columns=['市场组合收益率'])
        rst = pd.DataFrame()
        rst['市场组合收益率'] = market_type
        rst['日期'] = time_list
        rst.set_index(['日期'], inplace=True)
        return rst

    def gen_market(self):
        '''计算同类基准
        中证800收益率, 中证国债收益率, 恒生指数收益率, 中证综合债收益率, 中证短债收益率
        MSCI收益率
        没有用到 中证可转债
        '''
        # 中证800
        market2 = self.market_fetch('000906')
        market1 = self.market_fetch('000001')
        market1 = market1[market1.index < '20050105']
        a = market2[market2.index == '20050104'].iloc[0, 0] / market1[market1.index== '20050104'].iloc[-1, 0]
        market1['000906'] = market1['000001'] * a
        index = market1[market1.index == '20050104'].index.tolist()[0]
        market_800 = pd.concat([market1[['000906']] , market2], axis=0).drop([index])

        # 中证国债
        m_country = self.market_fetch('h11006')
        # 恒生指数
        HSI  = self.get_market_wind('HSI.HI')
        #中证综合债
        m_bond = self.market_fetch('h11009')
        # 中证短债
        market_short = self.market_fetch('h11015')
        # 中证可转债
        market_tran = self.market_fetch('000832')
        # MSCI
        MSCI = self.get_market_wind("892400.MI")
        self.market = market_800.join([m_country,HSI,m_bond,market_short,market_tran,MSCI])
        self.market.columns = ['中证800','中证国债','恒生指数','中证综合债','中证短债','中证可转债','MSCI']
        return 

    def market_fetch(self, stock_index):
        '''获取所有交易日指数数据'''
        sql1 = '''
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
        ''' % {'index_code': stock_index}
        market = pd.DataFrame(self.cu_wind.execute(sql1).fetchall(), columns=['日期', '指数收盘价'])
        market.index = market['日期']
        del market['日期']
        market.columns = [stock_index]
        # market = market.pct_change()
        return market

    def get_market_wind(slef, code, index_start_date='19900101', end_date='20190331'):
        try:
            df = w.wsd(code, "close", index_start_date, end_date, "","Currency=CNY", usedf=True)[1]
            #df = w.wsd(code, "close", "2015-01-01", end_date, "","Currency=CNY", usedf=True)[1]
            df.index = pd.Series(df.index).apply(lambda x: str(x)[:4] + str(x)[5:7] + str(x)[8:10])
            df.columns = [code]
        except:
            df = pd.DataFrame(columns=[code])
        return df

    def get_fund_price_fof(self, code, start_date='', end_date='', time_list=[]):
        '''fof 基金计算波动和回测时用到的sql查询方法'''
        if time_list:
            start_date = time_list[0]
            end_date = time_list[-1]
        sql = '''
        select
        tradedate, closeprice
        from
        t_fof_value_info 
        where 
        fundid = '%(code)s'
        and
        tradedate >= '%(start_date)s'
        and
        tradedate <= '%(end_date)s'
        ''' % {'end_date': end_date, 'code': code, 'start_date': start_date}
        fund_price = pd.DataFrame(self.cu_pra.execute(sql).fetchall(), columns=['日期', '复权单位净值'])
        fund_price.set_index('日期', inplace=True)
        if fund_price.empty:
            raise Exception('lact of fund value')
        if time_list:
            if len(time_list) - fund_price.shape[0] > 30:
                raise Exception('lack of data')
            fund_price = fund_price.reindex(time_list)
        return fund_price

    def get_fund_price_index(self, code, start_date='', end_date='', time_list=[]):
        '''获取指数的净值数据'''
        if time_list:
            start_date = time_list[0]
            end_date = time_list[-1]
        sql =f'''
        select f2_1425,f7_1425 from wind.tb_object_1425  left join wind.tb_object_1090  on f1_1425 = f2_1090
        where f16_1090 = '{code}'
        and (f2_1425 >='{start_date}' and f2_1425 <= '{end_date}')
        and f4_1090 = 'S'
        order by f2_1425
        '''
        fund_price = pd.DataFrame(self.cu_wind.execute(sql).fetchall(), columns=['日期', '复权单位净值'])
        fund_price.set_index('日期', inplace=True)
        if fund_price.empty:
            raise Exception('lact of fund value')
        if time_list:
            if len(time_list) - fund_price.shape[0] > 30:
                raise Exception('lack of data')
            fund_price = fund_price.reindex(time_list)
        return fund_price

# 杠杆率标签
class LeverageRatio(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable
        self.cu_wind = season_lable.cu_wind
        self.cu_pra = season_lable.cu_pra

    def add_func(self):
        self.season_lable.leverage_ratio = self.leverage_ratio

    def leverage_ratio(self, codes, start_date='', end_date=''):
        '''计算杠杆率
        codes: 用来取特定代码的杠杆率，现在是统一求出来, 后面再加处理
        '''
        holded_kind = self.get_data_from_sql(codes, start_date, end_date)
        lev_mean = holded_kind.groupby('基金代码').mean()
        lev_mean.reset_index(inplace=True)
        rst_df = lev_mean[['基金代码', '杠杆率']]
        return rst_df

    def get_data_from_sql(self, codes, start_date, end_date):
        '''get lvereage ratio info
        wind.TB_OBJECT_1104 是按半年度更新
        '''
        if not codes:
            raise Exception('invalid imput code')
        if type(codes) != list:
            codes = [codes]
        codes_str = '(' + ','.join(["'" + str(i) + "'" for i in codes]) + ')'
        sql_kind='''
        select F16_1090 as 基金代码,
        F14_1104 as 截止时间,
        F5_1104 as 股票占比,
        F11_1104 as 现金占比,
        F13_1104 as 其他资产占比,
        F32_1104 as 债券市值占比,
        F45_1104 as 权证占比,
        F52_1104 as 基金占比,
        F55_1104 as 货币市场工具占比
        from wind.TB_OBJECT_1090 left join wind.TB_OBJECT_1104
        on F15_1104 = F2_1090
        where
        F14_1104 >= %(start_date)s
        and F14_1104 <= %(end_date)s
        and F16_1090 in %(codes_str)s
        '''%{'start_date': start_date, 'end_date': end_date, 'codes_str': codes_str}
        sql_res = self.cu_wind.execute(sql_kind).fetchall()
        res = pd.DataFrame(sql_res,
                           columns=['基金代码','报告期','股票%','现金%','其他资产%','债券%','权证%','基金%','货币市场工具%']
              ).sort_values(by='基金代码', ascending=True).reset_index(drop=True)
        res = res.fillna(0)
        res['杠杆率'] = res['股票%']+res['现金%']+res['其他资产%']+res['债券%']+res['权证%']+res['基金%']+res['货币市场工具%']
        return res

# 波动率标签
class StandardDeviation(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable
        self.get_fund_price_fof = season_lable.get_fund_price_fof
        self.get_fund_price = season_lable.get_fund_price

    def add_func(self):
        self.season_lable.standard_deviation = self.standard_deviation
        self.season_lable.compute_standard_deviation = self.compute_standard_deviation

    def standard_deviation(self, code, start_date, end_date, fof=False):
        '''计算波动率, 判断是否为fof基金走两个分支'''
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='daily', pre_flag=False)
        if not fof:
            # fund_price = self.get_fund_price(code, start_date, end_date)
            fund_price = self.get_fund_price(code, time_list=time_list)
        else:
            fund_price = self.get_fund_price_fof(code, time_list=time_list)

        zhou_bodong = self.compute_standard_deviation(fund_price)
        return zhou_bodong

    def compute_standard_deviation(self, df):
        ''' 计算波动率，dataframe.columnes=['日期', '复权单位净值’], 返回numpy.float64'''
        df2 = df.sort_values(by=['日期']).reset_index(drop=True)
        df2['fund_return'] = df2.复权单位净值.diff() / df2.复权单位净值.shift(1)
        df2.dropna(axis=0, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        bodong = df2.fund_return.std() * (math.sqrt(250))
        return bodong

# 最大回撤标签
class MaxDrawDown(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable
        self.get_fund_price_fof = season_lable.get_fund_price_fof
        self.get_fund_price = season_lable.get_fund_price
        self.get_fund_price_index = season_lable.get_fund_price_index

    def add_func(self):
        '''将接口暴露给上一级对象'''
        # 计算基金最大回撤
        self.season_lable.max_draw_down = self.max_draw_down
        # 批量计算接口
        self.season_lable.compute_max_draw_down = self.compute_max_draw_down
        # 计算指数最大回撤
        self.season_lable.max_draw_down_index = self.max_draw_down_index
        return

    def max_draw_down(self, code, start_date, end_date, fof=False):
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='daily', pre_flag=False)
        if not fof:
            # fund_price = self.get_fund_price(code, start_date, end_date)
            fund_price = self.get_fund_price(code, time_list=time_list)
        else:
            fund_price = self.get_fund_price_fof(code, time_list=time_list)
        mdd = self.compute_max_draw_down(fund_price)
        return mdd

    def max_draw_down_index(self, code, start_date, end_date):
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='daily', pre_flag=False)
        fund_price = self.get_fund_price_index(code, time_list=time_list)
        mdd = self.compute_max_draw_down(fund_price)
        return mdd

    def compute_max_draw_down(self, df):
        df2 = df.sort_values(by=['日期']).reset_index(drop=True)
        df2.columns = ['复权单位净值']
        df2.dropna(axis=0, how='any', inplace=True)
        price_list = df2['复权单位净值'].tolist()
        i = np.argmax((np.maximum.accumulate(price_list) - price_list) / np.maximum.accumulate(price_list))  # 结束位置
        if i == 0:
            max_down_value = 0
        else:
            j = np.argmax(price_list[:i])  # 开始位置
            max_down_value = (price_list[j] - price_list[i]) / (price_list[j])
        return max_down_value

# 区间收益标签
class IntervalProfit(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable
        self.get_fund_price = season_lable.get_fund_price
        self.get_fund_price_fof = season_lable.get_fund_price_fof

    def add_func(self):
        self.season_lable.interval_profit = self.interval_profit
        self.season_lable.interval_profit_index = self.interval_profit_index
        self.season_lable.compute_interva_profit = self.compute_interva_profit
        return

    def interval_profit(self, code, start_date, end_date, fof=False):
        # start_date 如果不是交易日，向前取一天作为替代
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='daily', pre_flag=True)
        start_date = start_date if time_list[1] == start_date else time_list[0]
        end_date = time_list[-1]
        if not fof:
            s_price_dt = self.get_fund_price(code, start_date, start_date, [])
            e_price_dt = self.get_fund_price(code, end_date, end_date, [])
        else:
            s_price_dt = self.get_fund_price_fof(code, start_date, start_date, [])
            e_price_dt = self.get_fund_price_fof(code, end_date, end_date, [])
        s_price = s_price_dt.iat[0, 0]
        e_price = e_price_dt.iat[0, 0]
        pft, annual_pft = self.compute_interva_profit(s_price, e_price, len(time_list))
        return pft, annual_pft

    def interval_profit_index(self, code, start_date, end_date):
        """计算指标的绝对收益率"""
        # 找出[起始，终止] 前闭后闭 的所有交易日，如果pre_flag为True 会自动往前去一个交易日
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='daily', pre_flag=True)
        # 重新定义起始时间
        start_date = start_date if time_list[1] == start_date else time_list[0]

        sql =f"""
        select f2_1425,f7_1425 from wind.tb_object_1425  left join wind.tb_object_1090  on f1_1425 = f2_1090
        where f16_1090 = '{code}'
        and (f2_1425 >='{start_date}' and f2_1425 <= '{end_date}')
        and f4_1090 = 'S'
        order by f2_1425
        """
        sql_res = self.season_lable.cu_wind.execute(sql).fetchall()

        s_close = sql_res[0][1]
        e_close = sql_res[-1][1]

        pft, annual_pft = self.compute_interva_profit(s_close, e_close, len(time_list))
        return pft,annual_pft

    def compute_interva_profit(self, s_price, e_price, day_cnt):
        '''计算收益率，返回年化和非年化两种结果'''
        itv_pft = e_price / s_price - 1
        annualized_itv_pft = itv_pft / day_cnt * math.sqrt(250)
        return itv_pft, annualized_itv_pft

# 超额收益alpha
class AlphaCategroy(object):
    '''提供3种接口进行计算，目前实现两种
    type=2 基金基准进行计算 基金基准不存在改取同类基准
    type=0 同类基准
    type=1 同类平均基准(未实现)
    '''
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable

    def add_func(self):
        self.season_lable.compute_alpha_categroy = self.compute_alpha_categroy
        return

    def compute_alpha_categroy(self, code, start_date, end_date, run_type=2, fof=False):
        ''' mainly process
        '''
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='weekly', pre_flag=True)
        # 基金基准
        if run_type == 2:
            if fof:
                fund_value = self.season_lable.get_fund_price_fof(code, time_list=time_list)
            else:
                fund_value = self.season_lable.get_fund_price(code, time_list=time_list)
            # 取不到 基金标准 换用同类标准
            try:
                fund_index_value = self.season_lable.get_fund_price_biwi(code, time_list=time_list)
                fund_index_rate = fund_index_value.pct_change()
            except Exception as err:
                print(err)
                ejfl_type = self.season_lable.get_ejfl_type(code, start_date, end_date)
                print('type: {}'.format(ejfl_type))
                fund_index_rate = self.season_lable.get_market(ejfl_type, time_list)
        # 同类基准
        elif run_type == 0:
            if fof:
                fund_value = self.season_lable.get_fund_price_fof(code, time_list=time_list)
            else:
                fund_value = self.season_lable.get_fund_price(code, time_list=time_list)
            ejfl_type, _ = self.season_lable.get_ejfl_type(code, start_date, end_date)
            print('type: {}'.format(ejfl_type))
            # 同类基准
            fund_index_rate = self.season_lable.get_market(ejfl_type, time_list)
        # 同类平均
        elif run_type == 1:
            if fof:
                fund_value = self.season_lable.get_fund_price_fof(code, time_list=time_list)
            else:
                fund_value = self.season_lable.get_fund_price(code, time_list=time_list)
            ejfl_type, rptdate = self.season_lable.get_ejfl_type(code, start_date, end_date)
            # 同类平均
            fund_index_rate = self.season_lable.get_mean_market(ejfl_type, rptdate, time_list)

        fund_rate = fund_value.pct_change()
        market = fund_rate.join(fund_index_rate)
        market.columns = ['基金收益率', '市场组合收益率']
        market = market.dropna(axis=0, how='any')
        rst = self.compute_alpha(market)
        return rst

    def compute_alpha(self, df):
        '''
        计算超额收益，取算数平均，返回年化结果
        '''
        df['超额收益'] = df['基金收益率'] - df['市场组合收益率']
        temp = df['超额收益'].sum() / df['超额收益'].size
        ### todo 将week_return 里数据移植过来，添加fund_value与index_value对比
        return temp * 52

#计算夏普比率
class SharpRation(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable

    def add_func(self):
        # 对外接口
        self.season_lable.sharp_ratio = self.sharp_ratio
        return

    def get_periodic_interest_rate_from_wind(self, start_date, end_date):
        '''从wind客户端获取一年定期利率
        '''
        rst = w.edb("M0043808", start_date, end_date, usedf=True)[1]
        rst = rst.reset_index(drop=True)
        rst.index = pd.Series(rst['时间']).apply(lambda x: str(x)[:4] + str(x)[5:7] + str(x)[8:10])
        rst.columns = ['一年定存利率','时间']
        return rst

    def get_periodic_interest_rate(self, time_list, excel_path=''):
        ''' 从wind客户端获取定存利率
            不能计算就返回空值
        '''
        if time_list:
            index_start_date, end_date = time_list[0], time_list[-1]
        else:
            return
        if not excel_path:
            rst = w.edb("M0043808", index_start_date, end_date, usedf=True)[1]
            rst = rst.reset_index(drop=True)
            rst.index = pd.Series(rst['时间']).apply(lambda x: str(x)[:4] + str(x)[5:7] + str(x)[8:10])
            rst.columns = ['一年定存利率','时间']
        else:
            if not os.path.exists(excel_path):
                raise Exception('Invalid pi excel_path')
            rst = pd.read_excel(excel_path, dtype='object')
            rst['日期'] = rst['时间'].apply(lambda x: str(x)[:4] + str(x)[5:7] + str(x)[8:10])
        rst = rst[(rst.日期 >= index_start_date) & (rst.日期 <= end_date)]
        interest = rst['一年定存利率'].mean() / (100 * 52)
        return interest 

    def get_week_fund_value(self, code, time_list):
        '''获取净值接口，后续添加其他方式'''
        fund_value_df = self.season_lable.get_fund_price(code, time_list=time_list)
        return fund_value_df

    def sharp_ratio(self, code, start_date, end_date):
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='weekly', pre_flag=True)
        week_fund_value = self.get_week_fund_value(code, time_list=time_list)
        file_path = os.path.join('lingyao_input', 'sharp_ratio', '2014-2019_per_interest.xlsx')
        periodic_interest = self.get_periodic_interest_rate(time_list=time_list, excel_path=file_path)
        earn_rate = week_fund_value.pct_change()
        sharp_ration = self.compute_sharp_raitio(earn_rate, periodic_interest)
        return sharp_ration

    def compute_sharp_raitio(self, earn_rate, interest):
        earn_rate.columns = ['基金收益率']
        stv = earn_rate['基金收益率'].std() * np.sqrt(52)
        rst = ((earn_rate['基金收益率'].mean() - interest) / stv) * 52
        return rst

# 计算下行波动率
class DownStd(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable
        # 取净值时往前多取
        self.offset = 5

    def add_func(self):
        # 对外接口
        self.season_lable.down_std= self.down_std
        self.season_lable.down_std_from_excel = self.down_std_from_excel
        return

    def down_std_get_fund_value(self, code, time_list):
        '''获取净值入口,数据来源变化在这里修改'''
        fund_value = self.season_lable.get_fund_price(code, time_list=time_list) 
        return fund_value

    def get_value_from_excel(self, excel_path, start_date, end_date):
        ex_df = pd.read_excel(excel_path)
        time_ser = ex_df['日期'].dt.strftime('%Y%m%d')
        ex_df['日期'] = time_ser
        ex_df = ex_df[['日期', '复权单位净值']]
        if start_date:
            ex_df = ex_df[ex_df.日期 >= start_date]
        if end_date:
            ex_df = ex_df[ex_df.日期 <= end_date]
        ex_df = ex_df.set_index(['日期'])
        return ex_df

    def down_std(self, code, start_date, end_date):
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        start_new = start_dt - timedelta(weeks=self.offset)
        start_str = start_new.strftime('%Y%m%d')
        time_list = self.season_lable.gen_time_list(start_str, end_date, s_type='daily', pre_flag=True)
        fund_value = self.down_std_get_fund_value(code, time_list)
        fund_value = fund_value.loc[fund_value.index >= start_date][:]
        fund_rate = fund_value.pct_change()
        down_stv = self.compute_down_std(fund_rate)
        return down_stv

    def down_std_from_excel(self, excel_path, start_date='', end_date=''):
        if not os.path.exists(excel_path):
            err = 'input error, file not exists {}'.format(excel_path)
        fund_value = self.get_value_from_excel(excel_path, start_date, end_date)
        fund_rate = fund_value.pct_change()
        down_stv = self.compute_down_std(fund_rate)
        return down_stv

    def compute_down_std(self, df):
        df.columns = ['基金收益率']
        net_np = np.array(df['基金收益率'].dropna(axis=0, how='any'))
        down_net = np.array(np.delete(net_np, np.where(net_np >= 0)[0]))
        down_stv = pow(np.power(down_net, 2).sum() / (len(df)-1), 1 / 2) * np.sqrt(52)
        return down_stv

# 计算信息比率
class InfomationRatio(object):
    def __init__(self, season_lable: SeasonLabel):
        self.season_lable = season_lable
        self.offset = 5

    def add_func(self):
        # 对外接口
        self.season_lable.info_ratio = self.info_ratio
        return

    def add_func(self):
        self.season_lable.info_ratio = self.info_ratio
        return

    def info_ratio(self, code, start_date, end_date):
        time_list = self.season_lable.gen_time_list(start_date, end_date, s_type='weekly', pre_flag=True)
        fund_value = self.season_lable.get_fund_price(code, time_list=time_list)
        try:
            fund_index_value = self.season_lable.get_fund_price_biwi(code, time_list=time_list)
            fund_index_rate = fund_index_value.pct_change()
        except Exception as err:
            print(err)
            ejfl_type = self.season_lable.get_ejfl_type(code, start_date, end_date)
            print('type: {}'.format(ejfl_type))
            fund_index_rate = self.season_lable.get_market(ejfl_type, time_list)
        fund_rate = fund_value.pct_change()
        market = fund_rate.join(fund_index_rate)
        market.columns = ['基金收益率', '市场组合收益率']
        market = market.dropna(axis=0, how='any')
        ir = self.compute_ir(market)
        return ir

    def compute_ir(self, df):
        df.columns = ['基金收益率', '市场组合收益率']
        df['差额'] = df['基金收益率'] - df['市场组合收益率']
        # 追踪误差
        tracking_error = df['差额'].std()
        print('tracking_error: {}'.format(tracking_error))
        df['基金净值'] = df['基金收益率']+1
        df['市场组合净值'] = df['市场组合收益率']+1
        r_p = df['基金净值'].product()/ df.iloc[0,-1]
        r_m = df['市场组合净值'].product() / df.iloc[0,-1]
        ir = (r_p - r_m) / (tracking_error * np.sqrt(52))
        return ir

tmp_local_season_label = SeasonLabel()
# 获取杠杆率数据
leverage_ratio = tmp_local_season_label.leverage_ratio

# 获取波动率数据
standard_deviation = tmp_local_season_label.standard_deviation
# 波动率数据接口
compute_standard_deviation = tmp_local_season_label.compute_standard_deviation

# 计算最大回撤
max_draw_down = tmp_local_season_label.max_draw_down
max_draw_down_index = tmp_local_season_label.max_draw_down_index
compute_max_draw_down = tmp_local_season_label.compute_max_draw_down

# 计算区间收益
interval_profit = tmp_local_season_label.interval_profit
# 计算指标的区间收益
interval_profit_index = tmp_local_season_label.interval_profit_index
# 计算区间收益数据接口
compute_interva_profit = tmp_local_season_label.compute_interva_profit

# 计算超额收益
compute_alpha_categroy = tmp_local_season_label.compute_alpha_categroy

# 计算下行波动率
down_std = tmp_local_season_label.down_std
down_std_from_excel = tmp_local_season_label.down_std_from_excel

#计算夏普率
sharp_ratio = tmp_local_season_label.sharp_ratio

# 计算信息比率
info_ratio = tmp_local_season_label.info_ratio


if __name__ == '__main__':
    print('hello world')
