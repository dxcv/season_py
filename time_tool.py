# encoding=utf-8
"""时间相关的工具类"""
from sql import sql_oracle

import pandas as pd
from datetime import datetime, timedelta


class Time_tool:
    """时间相关的工具类"""

    def __init__(self):
        self.cu = sql_oracle.cu

        self.time_table = 'wind.tb_object_1010'
        self.time_name = 'f1_1010'

    def set_db_info(self, table, name):
        """设置数据库表格和字段"""
        self.time_table = table
        self.time_name = name

    def get_trade_days(self, start_date: str, end_date: str):
        """后期周期内的所有交易日"""
        assert end_date >= start_date, '截止日必须大于起始日'

        sql_get_trade_days = f"""
        select {self.time_name} from {self.time_table}
        where {self.time_name} <= '{end_date}' and {self.time_name} >='{start_date}'
        order by {self.time_name} asc 
        """

        sql_res = self.cu.execute(sql_get_trade_days).fetchall()
        time_df = pd.DataFrame(sql_res, columns=['日期'])

        return time_df

    def get_week_trade_days(self, start_date: str, end_date: str):
        """获取周期内，每个交易周中的最后一个交易日"""
        assert end_date > start_date, '截止日必须大于起始日期'

        time_df = self.get_trade_days(start_date, end_date)

        time_df.set_index(pd.DatetimeIndex(time_df['日期']), inplace=True)

        time_list = pd.DataFrame()

        time_list['日期'] = time_df['日期'].resample('W').last()
        time_list.dropna(axis=0, inplace=True)
        time_list.sort_index(axis=0, ascending=True, inplace=True)
        time_list.reset_index(drop=True, inplace=True)

        return time_list

    def get_pre_week_trade_day(self, today):
        """获取前一个交易周的最后一个交易日"""
        # 格式化时间
        today = datetime.strptime(today, '%Y%m%d')
        # 找到上周天
        pre_week_day = today + timedelta(weeks=-1)
        # 计算与周五的差距，并将时间改为周五
        target_day = pre_week_day + timedelta(days=5 - pre_week_day.weekday())
        # 再格式化时间
        target_day = target_day.strftime('%Y%m%d')

        sql_get_pre_week_trade_day = f"""
        select {self.time_name} from {self.time_table}
        where {self.time_name} <={target_day}   and rownum <=1
        order by {self.time_name} desc 
        """

        res = self.cu.execute(sql_get_pre_week_trade_day).fetchall()
        return res[0][0]

    def get_pre_trade_day(self, today):
        """获取一个交易日之前的首个交易日"""
        sql_get_pre_trade_day = f"""
        select {self.time_name} from {self.time_table}
        where {self.time_name} < {today}   and rownum <=1
        order by {self.time_name} desc 
        """
        res = self.cu.execute(sql_get_pre_trade_day).fetchall()
        return res[0][0]

    def is_trade_day(self, date: str):
        """判断是否为交易日"""

        sql_is_trade_day = f"""
        select {self.time_name} from {self.time_table}
        where {self.time_name} = '{date}'        
        """

        sql_res = self.cu.execute(sql_is_trade_day).fetchall()
        res = True if sql_res else False
        return res


if __name__ == '__main__':
    t = Time_tool()

    time_df = t.get_week_trade_days('20190101', '20190601')
    print(time_df)
    pre_day = t.get_pre_week_trade_day('20190101')
    print(pre_day)
    is_trade = t.is_trade_day('20190601')
    print(is_trade)
