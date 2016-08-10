# -*- coding: utf-8 -*-

import dqn_agent_nature
import make_dataset
import copy
import csv
import numpy as np
import talib as ta

class Stock_agent():
    
    def __init__(self,agent):
    
        #agent = dqn_agent_nature.dqn_agent()
        self.Agent = agent
        self.stock = 0
        self.havestock = 0
        self.action = 0
        self.money = 1000000
        self.property =0
        self.buyprice = 0
        

    def get_reward(self, last_action, nowprice ,buyprice):
        
        if (last_action == 0) or (last_action == 1):

            return 0
            
        elif last_action == -1:
            
            return float(nowprice - buyprice) / buyprice
        
    def get_prospect_profit(self,havestock, nowprice, buyprice):
        #株を持っている場合の見込み利益を返す
        if havestock == 0:
            return 0
        elif havestock == 1:
            
            return float(nowprice - buyprice) / buyprice
        

    def calcstocks(self,money, price):
        i = 0
        _sum = 0
        while _sum <= money:
            i = i + 1
            _sum = 100 * price * i
            
        return 100 * (i - 1)
        
    def trading(self,term, price, traindata):
        #print term
        #print traindata
        #print traindata.shape
        start_p = self.money
        end_p = 0
        if price == -1:
            return 'error'
            
        for i in xrange(term - 1,len(price)):
            #print i,i-term
            observation = copy.deepcopy(traindata[:,i-term+1:i+1])
            #print observation
            prospect_profit = self.get_prospect_profit(self.havestock,price[i],self.buyprice)
            agent_status = np.array([self.havestock,prospect_profit])
            observation = observation.reshape(1,-1)#一次元配列に変形
            observation = np.array([np.r_[observation[0], agent_status]])
            #print observation
            reward = self.get_reward(self.action, price[i-1], self.buyprice)
            
            if i == (term - 1):
                print 'agent start!'
                Q_action = self.Agent.agent_start(observation)
            elif i == (len(price) - 1):
                print 'agent end!'
                Q_action = self.Agent.agent_end(reward)
            else:
                Q_action = self.Agent.agent_step(reward, observation)
                
            
            if Q_action == 1:#buy_pointのとき
                s = self.calcstocks(self.money, price[i])#現在の所持金で買える株数を計算
                
                if s > 0:#現在の所持金で株が買えるなら
                    self.havestock = 1
                    self.action = 1
                    #order.append(1)#買う
                    self.stock += s
                    self.buyprice = price[i]
                    self.money = self.money - s * self.buyprice
                else:
                    #order.append(0)#買わない
                    self.action = 0
                    
            elif Q_action == -1:#sell_pointのとき
                if self.havestock == 1:#株を持っているなら
                    self.action = -1
                    #order.append(-1)#売る
                    self.money = self.money + self.stock * price[i]
                    self.stock = 0
                    self.havestock = 0
                    #self.buyprice = 0
                else:#株を持っていないなら
                    #order.append(0)#何もしない
                    self.action = 0
                    
                    
            else:#no_operationのとき
                #order.append(0)
                self.action = 0
                
                
            self.property = self.stock * price[i] + self.money
            #proper.append(self.property)
            #stocks.append(self.stock)
            end_p = self.property#最終総資産
            
        profit_ratio = float((end_p - start_p) / start_p) * 100
        
        return profit_ratio
            
class StockMarket():
    
    def __init__(self,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):
    
        
        
        self.u_vol = u_vol
        self.u_ema = u_ema
        self.u_rsi = u_rsi
        self.u_macd = u_macd
        self.u_stoch = u_stoch
        self.u_wil = u_wil
        
    def get_trainData(self,filename,end_train_day,input_num,stride=1):
        
        all_data = []
        traindata = []
        
        #print tech_name
        filepath = "./stockdata/%s" % filename
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = self.readfile(filepath)

        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(end_train_day)
        except:
            print "can't find start_test_day"
            #start_test_dayが見つからなければ次のファイルへ
            raise Exception('cannot find start_test_day')            
        
        cutpoint = iday - input_num + 1
        
        rec = copy.copy(_close)
        price_min = min(_close)
        price_max = max(_close)
        make_dataset.normalizationArray(rec,price_min,price_max)
        all_data.append(rec)
        
        
        if self.u_vol == True:
            vol_list = _volume
            t_min = min(vol_list[:cutpoint])
            t_max = max(vol_list[:cutpoint])
            make_dataset.normalizationArray(vol_list,t_min,t_max)
            all_data.append(vol_list)
            
        if self.u_ema == True:
            ema_list1 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 10)
            ema_list2 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 25)
            ema_list1 = np.ndarray.tolist(ema_list1)
            ema_list2 = np.ndarray.tolist(ema_list2)
            t_min = min(_close[:cutpoint])
            t_max = max(_close[:cutpoint])
            
            make_dataset.normalizationArray(ema_list1,t_min,t_max)
            make_dataset.normalizationArray(ema_list2,t_min,t_max)
            all_data.append(ema_list1)
            all_data.append(ema_list2)
            
        if  self.u_rsi == True:
            rsi_list = ta.RSI(np.array(_close, dtype='f8'), timeperiod = 14)
            rsi_list = np.ndarray.tolist(rsi_list)
            
            make_dataset.normalizationArray(rsi_list,0,100)
            all_data.append(rsi_list)
            
        if self.u_macd == True:
            macd_list,signal,hist = ta.MACD(np.array(_close, dtype='f8'), fastperiod = 12, slowperiod = 26, signalperiod = 9)
            macd_list = np.ndarray.tolist(macd_list)
            signal = np.ndarray.tolist(signal)
            
            t_min = np.nanmin(macd_list[:cutpoint])
            t_max = np.nanmax(macd_list[:cutpoint])
            if (t_min == np.nan) or (t_max == np.nan):
                print 'np.nan error'
                raise Exception('np.nan error')
            make_dataset.normalizationArray(macd_list,t_min,t_max)
            make_dataset.normalizationArray(signal,t_min,t_max)
            all_data.append(macd_list)
            all_data.append(signal)
            
        if self.u_stoch == True:
            slowk,slowd = ta.STOCH(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), fastk_period = 5,slowk_period=3,slowd_period=3)
            slowk = np.ndarray.tolist(slowk)
            slowd = np.ndarray.tolist(slowd)
            make_dataset.normalizationArray(slowk,0,100)
            make_dataset.normalizationArray(slowd,0,100)
            all_data.append(slowk)
            all_data.append(slowd)
            
        if self.u_wil == True:
            will = ta.WILLR(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), timeperiod = 14)
            will = np.ndarray.tolist(will)
            make_dataset.normalizationArray(will,-100,0)
            all_data.append(will)
        
        all_data = np.array(all_data)
        
        traindata = all_data[:,:cutpoint]
        trainprice = _close[:cutpoint]
        
        #テクニカル指標のパラメータ日数分最初を切る
        traindata = traindata[:,30:]
        trainprice = trainprice[30:]
        
        return traindata,trainprice
    
    def get_testData(self,filename,start_test_day,input_num,stride=1):
        
        all_data = []
        testdata = []
        
        #print tech_name
        filepath = "./stockdata/%s" % filename
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = self.readfile(filepath)

        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            #start_test_dayが見つからなければ次のファイルへ
            raise Exception('cannot find start_test_day')            
        
        cutpoint = iday - input_num + 1
        
        rec = copy.copy(_close)
        price_min = min(_close[:cutpoint])
        price_max = max(_close[:cutpoint])
        make_dataset.normalizationArray(rec,price_min,price_max)
        all_data.append(rec)
        
        
        if self.u_vol == True:
            vol_list = _volume
            t_min = min(vol_list[:cutpoint])
            t_max = max(vol_list[:cutpoint])
            make_dataset.normalizationArray(vol_list,t_min,t_max)
            all_data.append(vol_list)
            
        if self.u_ema == True:
            ema_list1 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 10)
            ema_list2 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = 25)
            ema_list1 = np.ndarray.tolist(ema_list1)
            ema_list2 = np.ndarray.tolist(ema_list2)
            t_min = min(_close[:cutpoint])
            t_max = max(_close[:cutpoint])
            
            make_dataset.normalizationArray(ema_list1,t_min,t_max)
            make_dataset.normalizationArray(ema_list2,t_min,t_max)
            all_data.append(ema_list1)
            all_data.append(ema_list2)
            
        if  self.u_rsi == True:
            rsi_list = ta.RSI(np.array(_close, dtype='f8'), timeperiod = 14)
            rsi_list = np.ndarray.tolist(rsi_list)
            
            make_dataset.normalizationArray(rsi_list,0,100)
            all_data.append(rsi_list)
            
        if self.u_macd == True:
            macd_list,signal,hist = ta.MACD(np.array(_close, dtype='f8'), fastperiod = 12, slowperiod = 26, signalperiod = 9)
            macd_list = np.ndarray.tolist(macd_list)
            signal = np.ndarray.tolist(signal)
            
            t_min = np.nanmin(macd_list[:cutpoint])
            t_max = np.nanmax(macd_list[:cutpoint])
            if (t_min == np.nan) or (t_max == np.nan):
                print 'np.nan error'
                raise Exception('np.nan error')
            make_dataset.normalizationArray(macd_list,t_min,t_max)
            make_dataset.normalizationArray(signal,t_min,t_max)
            all_data.append(macd_list)
            all_data.append(signal)
            
        if self.u_stoch == True:
            slowk,slowd = ta.STOCH(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), fastk_period = 5,slowk_period=3,slowd_period=3)
            slowk = np.ndarray.tolist(slowk)
            slowd = np.ndarray.tolist(slowd)
            make_dataset.normalizationArray(slowk,0,100)
            make_dataset.normalizationArray(slowd,0,100)
            all_data.append(slowk)
            all_data.append(slowd)
            
        if self.u_wil == True:
            will = ta.WILLR(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), timeperiod = 14)
            will = np.ndarray.tolist(will)
            make_dataset.normalizationArray(will,-100,0)
            all_data.append(will)
        
        all_data = np.array(all_data)
        
        testdata = all_data[:,cutpoint:]
        testprice = _close[cutpoint:]
        
        return testdata,testprice
    
    def readfile(self,filename):
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        f = open(filename,'rb')
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            #print row
            #print row[0]
            _time.append(float(row[0]))
            _open.append(float(row[1])*float(row[6]))
            _max.append(float(row[2])*float(row[6]))
            _min.append(float(row[3])*float(row[6]))
            _close.append(float(row[4])*float(row[6]))
            _volume.append(float(row[5])*float(row[6]))
            _keisu.append(float(row[6]))
            _shihon.append(float(row[7]))
        
        f.close()   
        return _time,_open,_max,_min,_close,_volume,_keisu,_shihon
