# -*- coding: utf-8 -*-
# coding: utf-8

import make_dataset
import os
import talib as ta
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import argparse
import multiprocessing as mp
import time
import tools
"""
Created on Tue Dec  8 21:59:37 2015

@author: wada
"""


#現在の資金で株をどのくらい買えるかを計算
def calcstocks(money, price):
    i = 0
    _sum = 0
    while _sum <= money:
        i = i + 1
        _sum = 100 * price * i
        
    return 100 * (i - 1)
    
    
def getStrategy_RSI(start_trading_day,end_trading_day,_time,_close):
    point = []
    iday = _time.index(start_trading_day)
    eday = _time.index(end_trading_day)
    rsi = ta.RSI(np.array(_close,dtype='f8'),timeperiod=14)
    rsi = rsi[iday:eday]
    point.append(0)
    for i in range(1,len(rsi)):
        if (rsi[i] <= 30) and (rsi[i - 1] > 30):
            point.append(1)
        elif (rsi[i] >= 50) and (rsi[i - 1] < 50):
            point.append(-1)
        else:
            point.append(0)
            
    return point
    
def getStrategy_MACD(start_trading_day,end_trading_day,_time,_close):
    point = []
    iday = _time.index(start_trading_day)
    eday = _time.index(end_trading_day)
    macd, signal,hist = ta.MACD(np.array(_close,dtype='f8'),fastperiod=12,slowperiod=26,signalperiod=9)
    macd = macd[iday:eday]
    signal = signal[iday:eday]
    point.append(0)
    for i in range(1,len(macd)):
        if (macd[i-1] <= signal[i-1]) and (macd[i] >= signal[i]):
            point.append(1)
        elif (macd[i-1] >= signal[i-1]) and (macd[i] <= signal[i]):
            point.append(-1)
        else:
            point.append(0)
    return point
    
def getStrategy_GD(start_trading_day,end_trading_day,_time,_close):
    point = []
    iday = _time.index(start_trading_day)
    eday = _time.index(end_trading_day)
    short_ema = ta.EMA(np.array(_close,dtype='f8'),timeperiod=10)
    long_ema = ta.EMA(np.array(_close,dtype='f8'),timeperiod=25)
    short_ema = short_ema[iday:eday]
    long_ema = long_ema[iday:eday]
    point.append(0)
    for i in range(1,len(short_ema)):
        if (short_ema[i-1] <= long_ema[i-1]) and (short_ema[i] >= long_ema[i]):
            point.append(1)
        elif (short_ema[i-1] >= long_ema[i-1]) and (short_ema[i] <= long_ema[i]):
            point.append(-1)
        else:
            point.append(0)
    return point
    
def getStrategy_STOCH(start_trading_day,end_trading_day,_time,_close,_max,_min):
    point = []
    iday = _time.index(start_trading_day)
    eday = _time.index(end_trading_day)
    slowk,slowd = ta.STOCH(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), fastk_period = 5,slowk_period=3,slowd_period=3)
    slowk = slowk[iday:eday]
    slowd = slowd[iday:eday]
    point.append(0)
    for i in range(1,len(slowk)):
        if (slowk[i-1] <= slowd[i-1]) and (slowk[i] >= slowd[i]) and (slowk[i] <= 30):
            point.append(1)
        elif (slowk[i-1] <= 50) and (slowk[i] >= 50):
            point.append(-1)
        else:
            point.append(0)
    return point
    
def trading(money,point,price):
    proper = []
    order = []
    stocks = []
    stock = 0
    buyprice = 0
    havestock = 0#株を持っている：１，持っていない：０
    trading_count = 0#取引回数
    #一日目は飛ばす
    start_p = money#初期総資産
    proper.append(start_p)
    order.append(0)
    stocks.append(0)
    
    
    #trading loop
    for i in range(1,len(point)):
        if point[i] == 1:#buy_pointのとき
            s = calcstocks(money, price[i])#現在の所持金で買える株数を計算
            
            if s > 0:#現在の所持金で株が買えるなら
                havestock = 1
                order.append(1)#買う
                stock += s
                buyprice = price[i]
                money = money - s * buyprice
            else:
                order.append(0)#買わない
                
        elif point[i] == -1:#sell_pointのとき
            if havestock == 1:#株を持っているなら
                order.append(-1)#売る
                money = money + stock * price[i]
                trading_count += 1
                stock = 0
                havestock = 0
            else:#株を持っていないなら
                order.append(0)#何もしない
                
        else:#no_operationのとき
            order.append(0)
        
        _property = stock * price[i] + money
        proper.append(_property)
        stocks.append(stock)
        end_p = _property#最終総資産
        
    profit_ratio = float((end_p - start_p) / start_p) * 100
    
    return profit_ratio, proper, order, stocks
    
def trading_file(filename):
    start_trading_day = 20000104
    end_trading_day = 20081230
    _property = 0#総資産
    money = 1000000#所持金
    
    filepath = args.data_folder +'/'+ filename
    #株価データの読み込み
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = make_dataset.readfile(filepath)
    try:
        iday = _time.index(start_trading_day)
        eday = _time.index(end_trading_day)
    except:
        return np.nan
        
    point_rsi = getStrategy_RSI(start_trading_day,end_trading_day,_time,_close)
    #売買開始日からスライス
    _time = _time[iday:eday]
    _close = _close[iday:eday]
    
    profit_ratio = trading(money,point_rsi,_close)[0]
    
    return profit_ratio
    


    
def trading_parallel(files, proc):
    pool = mp.Pool(proc)
    callback = pool.map(trading_file, files)
    
    return np.nanmean(np.array(callback))
    
def trading_parallel_async(files,proc):
    pool = mp.Pool(proc)
    callback = [pool.apply_async(trading_file,args=(i,)) for i in files]
    return np.nanmean(np.array([p.get() for p in callback]))

def main(files):
    start_trading_day = 20000104
    end_trading_day = 20081230

    #start_trading_day = 20090105
    #end_trading_day = 20101229


    meigara_count = 0


    sum_profit_ratio_rsi = 0
    sum_profit_ratio_macd = 0
    sum_profit_ratio_gd = 0
    sum_profit_ratio_stoch = 0
    sum_profit_ratio_rsi = 0
    sum_bh_profit_ratio = 0
        

    for f in tqdm(files):
        #print f
        
        _property = 0#総資産
        money = 1000000#所持金
        
        filepath = args.data_folder +'/'+ f
        #株価データの読み込み
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = make_dataset.readfile(filepath)
        try:
            iday = _time.index(start_trading_day)
            eday = _time.index(end_trading_day)
        except:
            #print "can't find start_test_day"
            continue#start_trading_dayが見つからなければ次のファイルへ   
            
        point_rsi = getStrategy_RSI(start_trading_day,end_trading_day,_time,_close)
        point_macd = getStrategy_MACD(start_trading_day,end_trading_day,_time,_close)
        point_gd = getStrategy_GD(start_trading_day,end_trading_day,_time,_close)
        point_stoch = getStrategy_STOCH(start_trading_day,end_trading_day,_time,_close,_max,_min)
        
        #売買開始日からスライス
        _time = _time[iday:eday]
        _close = _close[iday:eday]
            
        #buy&holdの利益率を計算
        bh_profit_ratio = float((_close[-1] - _close[0]) / _close[0]) * 100
        sum_bh_profit_ratio += bh_profit_ratio
        profit_ratio = trading(money,point_rsi,_close)
        #print "RSI profit of %s is %f " % (f, profit_ratio[0])
        sum_profit_ratio_rsi += profit_ratio[0]
        
        profit_ratio = trading(money,point_macd,_close)
        #print "MACD profit of %s is %f " % (f, profit_ratio[0])
        sum_profit_ratio_macd += profit_ratio[0]
        
        profit_ratio = trading(money,point_gd,_close)
        #print "GD profit of %s is %f " % (f, profit_ratio[0])
        sum_profit_ratio_gd += profit_ratio[0]
        
        profit_ratio = trading(money,point_stoch,_close)
        #print "STOCH profit of %s is %f " % (f, profit_ratio[0])
        sum_profit_ratio_stoch += profit_ratio[0]
        
        meigara_count += 1
        #print meigara_count
        
        
        
    print "RSI profit average is = %f" % (sum_profit_ratio_rsi / meigara_count)
    print "MACD profit average is = %f" % (sum_profit_ratio_macd / meigara_count)
    print "GD profit average is = %f" % (sum_profit_ratio_gd / meigara_count)
    print "STOCH profit average is = %f" % (sum_profit_ratio_stoch / meigara_count)
    print 'buy&hold profit = %f' % (sum_bh_profit_ratio / meigara_count)
    print "all meigara is %d" % meigara_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--data_folder', '-f', type=str, default='./stockdata',
                        help='stock data folder')
    args = parser.parse_args()
    
    files = os.listdir(args.data_folder)
    #starttime = time.time()
    #main(files)
    #print time.time() - starttime
    #starttime = time.time()
    print trading_parallel(files,proc=10)
    #print time.time() - starttime
    #starttime  = time.time()
    print trading_parallel_async(files,proc=10)
    #print time.time() - starttime