import sys
import time
import pandas as pd

from season_label import leverage_ratio, standard_deviation, max_draw_down, interval_profit 
from season_label import compute_alpha_categroy, down_std, down_std_from_excel
from season_label import sharp_ratio, info_ratio

def get_year_time_list(in_year):
    return [in_year+'0101', in_year+'1231']

def test_leverage_ratio():
    print('hello world')

# 测试区间收益率
def test_interval_profit():
    code, start, end = '163807', '20170101', '20171231'
    res, an = interval_profit(code, start, end)
    print('res: {}, {}'.format(res, an))

    # 测试 fof 接口
    code, start, end = '04C9489501FOFIF3', '20170101', '20180101'
    f_res, an = interval_profit(code, start, end, fof=True)
    print('f_res: {}, {}'.format(f_res, an))
    return

def test_max_draw_down():
    code, start, end = '163807', '20170101', '20171231'
    res = max_draw_down(code, start, end)
    print('res: {}'.format(res))
    return

def test_standard_deviation():
    # 测试普通接口
    print(start)
    code, start, end = '163807', '20170101', '20171231'
    for i in range(10):
        res = standard_deviation(code, start, end)
    print('res: {}'.format(res))
    """

    code, start, end = '163807', '20170101', '20180110'
    res = standard_deviation(code, start, end)
    print('res: {}'.format(res))

    code, start, end = '163807', '20180101', '20181231'
    res = standard_deviation(code, start, end)
    print('res: {}'.format(res))

    code, start, end = '163807', '20170101', '20181231'
    res = standard_deviation(code, start, end)
    print('res: {}'.format(res))

    code, start, end = '163807', '20160101', '20181231'
    res = standard_deviation(code, start, end)
    print('res: {}'.format(res))

    code, start, end = '005757', '20180101', '20181231'
    res = standard_deviation(code, start, end)
    print('res: {}'.format(res))

    # 测试返回为nah
    # 净值数据缺失，周发布
    code, start, end = '004940', '20170630', '20180630'
    res = standard_deviation(code, start, end)
    print('res: {}'.format(res))
    """
    """
    # 测试 fof 接口
    code, start, end = '04C9489501FOFIF3', '20170101', '20180101'
    f_res = standard_deviation(code, start, end, fof=True)
    print('f_res: {}'.format(f_res))
    """

def test_leverage_ration():
    """测试杠杆率"""
    codes = ['960042', 'F000355']
    start, end = '20180101', '20190101'
    rst = leverage_ratio(codes, start, end)
    print(rst)

    codes = '960042' 
    start, end = '20180101', '20190101'
    rst = leverage_ratio(codes, start, end)
    print(rst)

# 测试超额收益
def test_compute_alpha_categroy():
    codes = '960042' 
    codes = '370010'
    codes = '110011'
    codes = '002073'
    start, end = '20151231', '20181231'

    # run_type: 0 同类基准, 1 同类平均, 2 基金基准
    # 基金基准 空值过多
    codes, start, end = '519644', '20150101', '20161231'
    codes, start, end = '370010', '20151001', '20180930'
    '''
    run_type = 2
    rst = compute_alpha_categroy(codes, start, end, run_type)
    print(rst)
    run_type = 1
    rst = compute_alpha_categroy(codes, start, end, run_type)
    print(rst)
    '''
    run_type = 0
    rst = compute_alpha_categroy(codes, start, end, run_type)
    print(rst)
    '''
    # 同类基准
    run_type = 0
    rst = compute_alpha_categroy(codes, start, end, run_type)
    print(rst)
    # 测试是否根据不同净值取不同基准
    start, end = '20120101', '20180101'
    run_type = 0
    rst = compute_alpha_categroy(codes, start, end, run_type)
    print(rst)
    return
    '''
    '''
    # 同类平均, 时间开销测试
    s_time = time.time()
    start, end = '20160331', '20190331'
    run_type = 1
    rst = compute_alpha_categroy(codes, start, end, run_type)
    print(rst)
    e_time = time.time()
    print('caculate cost {}'.format(e_time - s_time))
    '''
    return

def test_info_ratio():
    # 信息比特率偏大
    '''
    code = '110011'
    start, end = '20120101', '20180101'
    rst = info_ratio(code, start, end)
    print(rst)
    '''
    code = '002073'
    start, end = '20170101', '20171231'
    rst = info_ratio(code, start, end)
    print(rst)
    return

def test_down_std():
    # file_path = 'D:\\Documents\\Tencent Files\\183866190\\FileRecv\\产品复权单位净值1.xls'
    file_path = 'D:\\Documents\\Tencent Files\\183866190\\FileRecv\\产品复权单位净值1.xls'
    with open('gaoyangwen.txt', 'a') as f:
        start_date, end_date = '20140101', '20141231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
        start_date, end_date = '20150101', '20151231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
        start_date, end_date = '20160101', '20161231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
        start_date, end_date = '20170101', '20171231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
        start_date, end_date = '20180101', '20181231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
        start_date, end_date = '20190101', '20191231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
        start_date, end_date = '20140101', '20191231'
        rst = down_std_from_excel(file_path, start_date, end_date)
        print(rst)
        print(rst, file=f)
    return

# 测试夏普接口
def test_sharp():
    code, start, end = '163807', '20170101', '20171231'
    res, an = sharp_ratio(code, start, end)
    print('res: {}, {}'.format(res, an))
    return

def test_dt_input():
    test_dt  = pd.DataFrame()
    import random
    value_list = []
    for i in range(10):
        value_list.append(random.random() * 1000)
    time_list = list(range(10))
    test_dt['复权单位净值'] = value_list
    test_dt['日期'] = time_list

    # 计算波动率
    from season_label import compute_standard_deviation
    std = compute_standard_deviation(test_dt)
    print('std: {}'.format(std))

    # 计算最大回撤
    from season_label import compute_max_draw_down
    tmp_dt = test_dt
    tmp_dt.index = tmp_dt['日期']
    mdd = compute_max_draw_down(test_dt)
    print('mdd: {}'.format(mdd))

    # 计算收益率
    from season_label import compute_interva_profit
    itv_pft, annual_pft = compute_interva_profit(value_list[0], value_list[-1], len(time_list))
    print('itv_pft: {}'.format(itv_pft))
    print('annual_pft: {}'.format(annual_pft))
    return

def main_test():
    # 测试波动率
    # test_standard_deviation()

    # 测试最大回测
    # test_max_draw_down()

    # 测试绝对收益
    # test_interval_profit()

    # 测试杠杆率
    # test_leverage_ration()

    # 测试超额收益alpha
    test_compute_alpha_categroy()

    # 测试下行波动率
    # test_down_std()

    # 测试夏普率
    # test_sharp()

    # 测试信息比率
    # test_info_ratio()

    # 测试数据接口
    # test_dt_input()

if __name__ == '__main__':
    main_test()
