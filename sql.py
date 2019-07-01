# encoding=utf-8

"""数据库"""

import cx_Oracle
from functions import single


@single
class Sql_oracle:
    """sql工具"""

    def __init__(self):
        self.connect_db()

    def connect_db(self):
        """链接db"""
        # wind 库
        [userName, password, hostIP, dbName, tablePrefix] = ['reader', 'reader', '172.16.50.232:1521', 'dfcf', 'wind']
        self.fund_db = cx_Oracle.connect(user=userName, password=password, dsn=hostIP + '/' + dbName)
        self.cu = self.fund_db.cursor()

        # pra库
        [userName, password, hostIP, dbName] = ['pra_info', 'pra_info', '172.16.126.23:1521', 'pra']
        self.fund_db_pra = cx_Oracle.connect(user=userName, password=password, dsn=hostIP + '/' + dbName)
        self.cu_pra_sel = self.fund_db_pra.cursor()

    def close(self):
        """断开链接"""
        self.cu.close()
        self.cu_pra_sel.close()

        self.fund_db.close()
        self.fund_db_pra.close()

    def sql_cu(self, sql):
        """
        wind库的查询方法
        :param sql: sqk语句
        :return: 二维数组
        """
        res = self.cu.execute(sql).fetchall()
        return res

    def sql_pra(self, sql):
        """
        pra库的查询方法
        :param sql: sqk语句
        :return: 二维数组
        """
        res = self.cu_pra_sel.execute(sql).fetchall()
        return res

    def get_value_fund(self,code,start,end):
        pass


    def __del__(self):
        print('退出，断开数据库')
        self.close()


sql_oracle = Sql_oracle()
