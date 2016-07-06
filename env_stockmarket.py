# -*- coding: utf-8 -*-

import dqn_agent_nature
import make_dataset

class Stock_agent:
    
    def __init__(self):
        self.stock = 0
        self.havestock = 0
        self.money = 1000000
        self.property =0
        self.buyprice = 0
        
    def observation():
        pass
        
    def get_agent():
        pass
        
    def get_reward():
        pass
        
    def update_agent():
        pass
        
class StockMarket():
    
    def __init__(self,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):
        self.u_vol = u_vol
        self.u_ema = u_ema
        self.u_rsi = u_rsi
        self.u_macd = u_macd
        self.u_stoch = u_stoch
        self.u_wil = u_wil
        
    def get_trainData(filename,end_train_day,input_num,stride=1):
        pass
        all_data = []
        traindata = []
        
        #print tech_name
        filepath = "./stockdata/%s" % filename
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)

        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(end_train_day)
        except:
            print "can't find start_test_day"
            #start_test_dayが見つからなければ次のファイルへ
            return -1,-1
        
        cutpoint = iday - input_num + 1
        
        rec = copy.copy(_close)
        price_min = min(_close)
        price_max = max(_close)
        normalizationArray(rec,price_min,price_max)
        all_data.append(rec)
        
        
        if u_vol == True:
            vol_list = _volume
            t_min = min(vol_list[:cutpoint])
            t_max = max(vol_list[:cutpoint])
            normalizationArray(vol_list,t_min,t_max)
            all_data.append(vol_list)
            
        if u_ema == True:
            ema_list1 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 10)
            ema_list2 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 25)
            ema_list1 = np.ndarray.tolist(ema_list1)
            ema_list2 = np.ndarray.tolist(ema_list2)
            t_min = min(_close[:cutpoint])
            t_max = max(_close[:cutpoint])
            
            normalizationArray(ema_list1,t_min,t_max)
            normalizationArray(ema_list2,t_min,t_max)
            all_data.append(ema_list1)
            all_data.append(ema_list2)
            
        if  u_rsi == True:
            rsi_list = ta.RSI(np.array(_close, dtype='f8'), timeperiod = 14)
            rsi_list = np.ndarray.tolist(rsi_list)
            
            normalizationArray(rsi_list,0,100)
            all_data.append(rsi_list)
            
        if u_macd == True:
            macd_list,signal,hist = ta.MACD(np.array(_close, dtype='f8'), fastperiod = 12, slowperiod = 26, signalperiod = 9)
            macd_list = np.ndarray.tolist(macd_list)
            signal = np.ndarray.tolist(signal)
            
            t_min = np.nanmin(macd_list[:cutpoint])
            t_max = np.nanmax(macd_list[:cutpoint])
            if (t_min == np.nan) or (t_max == np.nan):
                return -1,-1
            normalizationArray2(macd_list,t_min,t_max)
            normalizationArray2(signal,t_min,t_max)
            all_data.append(macd_list)
            all_data.append(signal)
            
        if u_stoch == True:
            slowk,slowd = ta.STOCH(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), fastk_period = 5,slowk_period=3,slowd_period=3)
            slowk = np.ndarray.tolist(slowk)
            slowd = np.ndarray.tolist(slowd)
            normalizationArray(slowk,0,100)
            normalizationArray(slowd,0,100)
            all_data.append(slowk)
            all_data.append(slowd)
            
        if u_wil == True:
            will = ta.WILLR(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), timeperiod = 14)
            will = np.ndarray.tolist(will)
            normalizationArray(will,-100,0)
            all_data.append(will)
        
        all_data = np.array(all_data)
        
        traindata = all_data[:,:cutpoint]
        trainprice = _close[:cutpoint]
        
        #テクニカル指標のパラメータ日数分最初を切る
        traindata = traindata[:,30:]
        trainprice = trainprice[30:]
        
        return traindata,trainprice
    