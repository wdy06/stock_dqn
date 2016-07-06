# -*- coding: utf-8 -*-
# coding: utf-8
import csv
import os
import numpy as np
import talib as ta
import time
import copy

"""
Created on Tue Dec  8 17:48:50 2015

@author: wada
"""
t_folder = './teacher_data/'

def readfile(filename):
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

#processing array index to 0~1
def normalizationArray(array,amin,amax):
    amin = float(amin)
    amax = float(amax)
    if amin != amax:
        for i,element in enumerate(array):
            if element > amax:
                array[i] = 1
            elif element < amin:
                array[i] = 0
            elif element == np.nan:
                array[i] = np.nan
            else:
                ret = (float(element) - amin) / (amax - amin)
                array[i] = ret
    #期間の最大最小が等しい場合はすべての要素を0.5とする
    elif amin == amax:
        for i,element in enumerate(array):
            array[i] = float(0.5)

def normalizationArray2(array,amin,amax):
    #processing array index to -1~1
    amin = float(amin)
    amax = float(amax)
    if amin != amax:
        for i,element in enumerate(array):
            if element > amax:
                array[i] = 1
            elif element < amin:
                array[i] = -1
            elif element == np.nan:
                array[i] = np.nan
            else:
                ret = -1 + (2*(float(element) - amin) / (amax - amin))
                array[i] = ret
    #期間の最大最小が等しい場合はすべての要素を0.5とする
    elif amin == amax:
        for i,element in enumerate(array):
            array[i] = float(0)
            
def denormalizationArray(array,amin,amax):
    amin = float(amin)
    amax = float(amax)
    if amin != amax:
        for i,element in enumerate(array):
            if element == 1:
                array[i] = amax
            elif element == 0:
                array[i] = amin
            else:
                ret = amin + float(element)*(amax - amin)
                array[i] = ret
    
def data_completion():#欠損値を前日の価格で補完
    files = os.listdir("./ori_stockdata")
    for f in files:    
        print f
        filepath = "./ori_stockdata/%s" % f
        fr = open(filepath,'rb')
        outfilepath = "./stockdata/%s" % f
        fw = open(outfilepath,'w')
        reader = csv.reader(fr)
        writer = csv.writer(fw)        
        #writer.writerow(next(reader))
        #last = []
        for i, row in enumerate(reader):
            #last = row[:]
            if i == 0:
                writer.writerow(row)
            else:
                if int(row[1]) != 0:
                    #print "completion!"
                    writer.writerow(row)
                    last = row[:]
                elif int(row[1]) == 0:
                    writer.writerow(last)
        fr.close()
        fw.close()

def arrange_train_num(inputfile, outputfile):
    print "start arrange..."
    start_time = time.clock()
    data = []
    c_buy = 0
    c_sell = 0
    c_no = 0
    print 'time:%d[s]' % (time.clock() - start_time)
    print 'open ' + inputfile
    icsvdata = open(t_folder + inputfile,'rb')
    print 'open ' + outputfile
    ocsvdata = open(t_folder + outputfile, 'w')
    reader = csv.reader(icsvdata)
    writer = csv.writer(ocsvdata)
    print 'start no_ope_data appending...'
    count = 0
    for row in reader:
        label = row[-3]
        count += 1
        if int(label) == 0:
            c_buy +=1
            writer.writerow(row)
        elif int(label) == 1:
            c_sell +=1
            writer.writerow(row)
        elif int(label) == 2:
            c_no += 1
            if count % 2 == 0:
                continue
            data.append(row)
            
    
    print "buy %d, sell %d, no %d" % (c_buy, c_sell, c_no)
    target_num = int((c_buy + c_sell) / 2)
    #print target_num
    print 'array shuffling...'
    print 'time:%d[s]' % (time.clock() - start_time)
    data = np.random.permutation(data)
    data = data[:target_num]
    print "arrange to"
    print "buy %d, sell %d, no %d" % (c_buy, c_sell, len(data))
    print 'no ope data writing...'
    writer.writerows(data)
    print 'time:%d[s]' % (time.clock() - start_time)
    
    
    
    icsvdata.close()
    ocsvdata.close()
    print "end arrange"
    
def arrange_train_num2(inputfile, outputfile):
    #regression teacher data arrange
    print "start arrange..."
    start_time = time.clock()
    data = []
    c_buy = 0
    c_sell = 0
    c_no = 0
    print 'time:%d[s]' % (time.clock() - start_time)
    print 'open ' + inputfile
    icsvdata = open(t_folder + inputfile,'rb')
    print 'open ' + outputfile
    ocsvdata = open(t_folder + outputfile, 'w')
    reader = csv.reader(icsvdata)
    writer = csv.writer(ocsvdata)
    print 'start no_ope_data appending...'
    count = 0
    for row in reader:
        target = row[-3]
        count += 1
        if float(target) >= 0.05:
            if count % 10 == 0:
                continue
            c_buy +=1
            writer.writerow(row)
        elif float(target) <= -0.05:
            if count % 10 == 0:
                continue
            c_sell +=1
            writer.writerow(row)
        else:
            if count % 2 == 0:
                continue
            data.append(row)
            c_no += 1
    
    print "buy %d, sell %d, no %d" % (c_buy, c_sell, c_no)
    target_num = int((c_buy + c_sell) / 2)
    #print target_num
    print 'array shuffling...'
    print 'time:%d[s]' % (time.clock() - start_time)
    data = np.random.permutation(data)
    data = data[:target_num]
    print "arrange to"
    print "buy %d, sell %d, no %d" % (c_buy, c_sell, len(data))
    print 'no ope data writing...'
    writer.writerows(data)
    print 'time:%d[s]' % (time.clock() - start_time)
    
    
    
    icsvdata.close()
    ocsvdata.close()
    print "end arrange"

def getMaxChangePrice(price_list):
    #リスト先頭の価格を基準にリスト内の価格で最も変動率が大きい価格を返す
    now_price = price_list[0]
    rec = [abs(x - now_price) for x in price_list]
    predic_price = price_list[rec.index(max(rec))]
    
    return predic_price
    
def getMaxPrice(price_list):
    return max(price_list)
def getTeacherData(filename,start_test_day,next_day,input_num):
    traindata = []
    testdata = []

    _time = []
    _open = []
    _max = []
    _min = []
    _close = []
    _volume = []
    _keisu = []
    _shihon = []
    filepath = "./stockdata/%s" % filename
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)

    #start_test_dayでデータセットを分割
    try:
        iday = _time.index(start_test_day)
    except:
        print "can't find start_test_day"
        #start_test_dayが見つからなければ次のファイルへ
        return -1
        
    cutpoint = iday - input_num + 1
        
    trainprice = _close[:cutpoint]
    testprice = _close[cutpoint:]
      
    if len(trainprice) < input_num or len(testprice) < input_num:
        return -1
    
    price_min = min(trainprice)
    price_max = max(trainprice)
    
    datalist = trainprice
    
    for i, price in enumerate(datalist):
        """
        if i % 2 == 0:
            #全部は多すぎるので半分
            continue
        """
        inputlist = copy.copy(datalist[i:i + input_num])
        
        
        try:
            now_price = datalist[i + input_num - 1]
            #predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            term_prices = datalist[i + input_num:i + input_num + next_day -1]
            rec = [abs(x - now_price) for x in term_prices]
            predic_price = term_prices[rec.index(max(rec))]
        except:
            continue#datalistが短すぎる場合は飛ばす
        outputlist = []
        outputlist.append((predic_price - now_price) / now_price)
        outputlist.append(price_min)
        outputlist.append(price_max)
        

        normalizationArray(inputlist,price_min,price_max)
        
        traindata.append(inputlist + outputlist)
        
        
        if i + input_num + next_day == len(datalist):
            break
        
    
    
    datalist = testprice
    
    for i, price in enumerate(datalist):
        """
        if i % 2 == 0:
            #全部は多すぎるので半分
            continue
        """
        inputlist = copy.copy(datalist[i:i + input_num])
        
        try:
            now_price = datalist[i + input_num - 1]
            #predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            term_prices = datalist[i + input_num:i + input_num + next_day -1]
            rec = [abs(x - now_price) for x in term_prices]
            predic_price = term_prices[rec.index(max(rec))]
        except:
            continue#datalistが短すぎる場合は飛ばす
        outputlist = []
        outputlist.append((predic_price - now_price) / now_price)
        outputlist.append(price_min)
        outputlist.append(price_max)
        
        normalizationArray(inputlist,price_min,price_max)
        
        testdata.append(inputlist + outputlist)
        if i + input_num + next_day == len(datalist):
            break
            
    return traindata, testdata
    
def getTeacherDataTech(filename,start_test_day,next_day,input_num, tech_name = None, param1 = None, param2 = None, param3 = None):
    #株価とテクニカル指標の教師データを作成し、そのリストを返す
    traindata = []
    testdata = []
    #print tech_name
    _time = []
    _open = []
    _max = []
    _min = []
    _close = []
    _volume = []
    _keisu = []
    _shihon = []
    filepath = "./stockdata/%s" % filename
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)

    #start_test_dayでデータセットを分割
    try:
        iday = _time.index(start_test_day)
    except:
        print "can't find start_test_day"
        #start_test_dayが見つからなければ次のファイルへ
        return -1
        
    if tech_name == "EMA":
        tech1 = ta.EMA(np.array(_close, dtype='f8'), timeperiod = param1)
        tech1 = np.ndarray.tolist(tech1)
    elif tech_name == "RSI":
        tech1 = ta.RSI(np.array(_close, dtype='f8'), timeperiod = param1)
        tech1 = np.ndarray.tolist(tech1)
    elif tech_name == "MACD":
        tech1,tech2,gomi = ta.MACD(np.array(_close, dtype='f8'), fastperiod = param1, slowperiod = param2, signalperiod = param3)
        tech1 = np.ndarray.tolist(tech1)
        tech2 = np.ndarray.tolist(tech2)
    elif tech_name == "STOCH":
        tech1,tech2 = ta.STOCH(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), fastk_period = param1,slowk_period=param2,slowd_period=param3)
        tech1 = np.ndarray.tolist(tech1)
        tech2 = np.ndarray.tolist(tech2)
    elif tech_name == "WILLR":
        tech1 = ta.WILLR(np.array(_max, dtype='f8'),np.array(_min, dtype='f8'),np.array(_close, dtype='f8'), timeperiod = param1)
        tech1 = np.ndarray.tolist(tech1)
    elif tech_name == "VOL":
        #print 'vol1'
        tech1 = _volume
    #print tech_name
    #print 'cc'
    cutpoint = iday - input_num + 1
    #print param1
    #print tech_name
    #_close = _close[2*param1:]
    trainprice = _close[:cutpoint]
    testprice = _close[cutpoint:]
    trainprice = trainprice[2*param1:]
    
    
    #tech1 = tech1[2*param1:]
    traintech1 = tech1[:cutpoint]
    testtech1 = tech1[cutpoint:]
    traintech1 = traintech1[2*param1:]
    
    
    
    if tech_name in ("MACD", "STOCH"):
        
        #tech2 = tech2[2*param1:]
        traintech2 = tech2[:cutpoint]
        testtech2 = tech2[cutpoint:]
        traintech2 = traintech2[2*param1:]
    
    
    
    
    if len(trainprice) < input_num or len(testprice) < input_num:
        return -1
    
    price_min = min(trainprice)
    price_max = max(trainprice)
    if tech_name in ("EMA", "MACD"):
        tech_min = min(trainprice)
        tech_max = max(testprice)
    elif tech_name in ("RSI", "STOCH"):
        tech_min = 0
        tech_max = 100
    elif tech_name == "WILLR":
        tech_min = -100
        tech_max = 0
    elif tech_name == "VOL":
        tech_min = min(traintech1)
        tech_max = max(traintech1)
    
    datalist = trainprice
    datalist_tech1 = traintech1
    if tech_name in ("MACD", "STOCH"):
        datalist_tech2 = traintech2
    
    print tech_name, input_num
    for i, price in enumerate(datalist):
        
        if i % 2 == 0:
            #全部は多すぎるので半分
            continue
        
        inputlist = copy.copy(datalist[i:i + input_num])
        inputlist_tech1 = copy.copy(datalist_tech1[i:i + input_num])
        if tech_name in ("STOCH", "MACD"):
            inputlist_tech2 = copy.copy(datalist_tech2[i:i + input_num])
        
        try:
            now_price = datalist[i + input_num - 1]
            #predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            term_prices = datalist[i + input_num:i + input_num + next_day -1]
            rec = [abs(x - now_price) for x in term_prices]
            predic_price = term_prices[rec.index(max(rec))]
        except:
            continue#datalistが短すぎる場合は飛ばす
        outputlist = []
        outputlist.append((predic_price - now_price) / now_price)
        outputlist.append(price_min)
        outputlist.append(price_max)
        

        normalizationArray(inputlist,price_min,price_max)
        normalizationArray(inputlist_tech1,tech_min,tech_max)
        
        if tech_name in ("STOCH", "MACD"):
            normalizationArray(inputlist_tech2,tech_min,tech_max)
            traindata.append(inputlist + inputlist_tech1 + inputlist_tech2 + outputlist)#train.csvに書き込み
        else:
            #print 'append'
            traindata.append(inputlist + inputlist_tech1 + outputlist)#train.csvに書き込み
        
        
        if i + input_num + next_day == len(datalist):
            break
        
    
    
    datalist = testprice
    datalist_tech1 = testtech1
    if tech_name in ("STOCH", "MACD"):
        datalist_tech2 = testtech2
    
    for i, price in enumerate(datalist):
        
        if i % 2 == 0:
            #全部は多すぎるので半分
            continue
        
        inputlist = copy.copy(datalist[i:i + input_num])
        inputlist_tech1 = copy.copy(datalist_tech1[i:i + input_num])
        if tech_name in ("STOCH", "MACD"):
            inputlist_tech2 = copy.copy(datalist_tech2[i:i + input_num])
        
        try:
            now_price = datalist[i + input_num - 1]
            #predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            term_prices = datalist[i + input_num:i + input_num + next_day -1]
            rec = [abs(x - now_price) for x in term_prices]
            predic_price = term_prices[rec.index(max(rec))]
        except:
            continue#datalistが短すぎる場合は飛ばす
        outputlist = []
        outputlist.append((predic_price - now_price) / now_price)
        outputlist.append(price_min)
        outputlist.append(price_max)
        
        normalizationArray(inputlist,price_min,price_max)
        normalizationArray(inputlist_tech1,tech_min,tech_max)
        
        if tech_name in ("STOCH", "MACD"):
            normalizationArray(inputlist_tech2,tech_min,tech_max)
            testdata.append(inputlist + inputlist_tech1 + inputlist_tech2 + outputlist)#train.csvに書き込み
        else:
            testdata.append(inputlist + inputlist_tech1 + outputlist)#train.csvに書き込み
        
        if i + input_num + next_day == len(datalist):
            break
            
    return traindata, testdata
    
def getTeacherDataMultiTech(filename,start_test_day,next_day,input_num,stride=1,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):
    
    #株価と複数のテクニカル指標の教師データを作成し、そのリストを返す
    all_data = []
    traindata = []
    testdata = []
    #print tech_name
    filepath = "./stockdata/%s" % filename
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)

    #start_test_dayでデータセットを分割
    try:
        iday = _time.index(start_test_day)
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
    testdata = all_data[:,cutpoint:]
    
    trainprice = _close[:cutpoint]
    testprice = _close[cutpoint:]
    
    #テクニカル指標のパラメータ日数分最初を切る
    traindata = traindata[:,30:]
    trainprice = trainprice[30:]
    
    if (len(traindata[0]) < input_num) or (len(testdata[0]) < input_num):
        return -1,-1
    
    train_output = []
    trainprice = trainprice[input_num - 1:]
    for i,price in enumerate(trainprice):
        now_price = price
        term_prices = trainprice[i:i + next_day]
        if len(term_prices) != next_day:
            break
        #print term_prices
        predic_price = getMaxChangePrice(term_prices)
        train_output.append((predic_price - now_price) / now_price)
    #raw_input()
    test_output = []
    testprice = testprice[input_num - 1:]
    for i,price in enumerate(testprice):
        now_price = price
        term_prices = testprice[i:i + next_day]
        if len(term_prices) != next_day:
            break
        predic_price = getMaxChangePrice(term_prices)
        test_output.append((predic_price - now_price) / now_price)
        
    f_traindata = []
    
    for i in range(0,len(traindata[0]),stride):
        if i >= len(train_output):
            break
        rec = np.reshape(traindata[:,i:i+input_num],(1,-1))[0]
        
        rec = np.ndarray.tolist(rec)
        f_traindata.append(rec + [train_output[i]] + [price_min] + [price_max])
        
    #print np.array(f_traindata).shape
    
    f_testdata = []
    for i in range(0,len(testdata[0]),stride):
        if i >= len(test_output):
            break
        rec = np.reshape(testdata[:,i:i+input_num],(1,-1))[0]
        
        rec = np.ndarray.tolist(rec)
        f_testdata.append(rec + [test_output[i]] + [price_min] + [price_max])
    #print np.array(f_testdata).shape
    #raw_input()
    return f_traindata,f_testdata
    
def getTeacherDataMultiTech_label(filename,start_test_day,next_day,input_num,stride=1,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):
    
    #株価と複数のテクニカル指標の教師データを作成し、そのリストを返す
    all_data = []
    traindata = []
    testdata = []
    #print tech_name
    filepath = "./stockdata/%s" % filename
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)

    #start_test_dayでデータセットを分割
    try:
        iday = _time.index(start_test_day)
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
    testdata = all_data[:,cutpoint:]
    
    trainprice = _close[:cutpoint]
    testprice = _close[cutpoint:]
    
    #テクニカル指標のパラメータ日数分最初を切る
    traindata = traindata[:,30:]
    trainprice = trainprice[30:]
    
    if (len(traindata[0]) < input_num) or (len(testdata[0]) < input_num):
        return -1,-1
    
    train_output = []
    trainprice = trainprice[input_num - 1:]
    for i,price in enumerate(trainprice):
        now_price = price
        term_prices = trainprice[i:i + next_day]
        if len(term_prices) != next_day:
            break
        #print term_prices
        predic_price = getMaxChangePrice(term_prices)
        predic_ratio = (predic_price - now_price) / now_price
        if predic_ratio > 0.05:
            train_output.append(0)
        elif predic_ratio < -0.05:
            train_output.append(1)
        else:
            train_output.append(2)
    #raw_input()
    test_output = []
    testprice = testprice[input_num - 1:]
    for i,price in enumerate(testprice):
        now_price = price
        term_prices = testprice[i:i + next_day]
        if len(term_prices) != next_day:
            break
        predic_price = getMaxChangePrice(term_prices)
        predic_ratio = (predic_price - now_price) / now_price
        if predic_ratio > 0.05:
            test_output.append(0)
        elif predic_ratio < -0.05:
            test_output.append(1)
        else:
            test_output.append(2)
        
    f_traindata = []
    
    for i in range(0,len(traindata[0]),stride):
        if i >= len(train_output):
            break
        
        rec = np.reshape(traindata[:,i:i+input_num],(1,-1))[0]
        
        rec = np.ndarray.tolist(rec)
        
        f_traindata.append(rec + [train_output[i]] + [price_min] + [price_max])
        
    #print np.array(f_traindata).shape
    count = 0
    f_testdata = []
    for i in range(0,len(testdata[0]),stride):
        if i >= len(test_output):
            break
        
        rec = np.reshape(testdata[:,i:i+input_num],(1,-1))[0]
        
        rec = np.ndarray.tolist(rec)
        f_testdata.append(rec + [test_output[i]] + [price_min] + [price_max])
    #print np.array(f_testdata).shape
    #raw_input()
    return f_traindata,f_testdata
#------------------------------------------
   
def make_dataset_1():#一定期間の株価から翌日の株価を回帰予測    
    start_test_day = 20090105 
    input_num = 20
    output_num = 1     
    
    train_count = 0
    test_count = 0
    
    fw1 = open(t_folder + 'train.csv', 'w')
    fw2 = open(t_folder + 'test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for f in files:
        print f
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        del _open
        del _max
        del _min
        del _volume
        del _keisu
        del _shihon
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        trainlist = _close[:iday]
        testlist = _close[iday:]
        #train data
        #x_train = []
        #y_train = []
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            outputlist = copy.copy(datalist[input_num + i:input_num + i + output_num])
            norm_min = min(datalist[i:input_num + i + output_num])
            norm_max = max(datalist[i:input_num + i + output_num])
            normalizationArray(inputlist,norm_min,norm_max)
            normalizationArray(outputlist,norm_min,norm_max)
            #x_train.append(inputlist)
            #y_train.append(outputlist)
            writer1.writerow(inputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + output_num == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            outputlist = copy.copy(datalist[input_num + i:input_num + i + output_num])
            norm_min = min(inputlist + outputlist)
            norm_max = max(inputlist + outputlist)
            normalizationArray(inputlist,norm_min,norm_max)
            normalizationArray(outputlist,norm_min,norm_max)
            #x_test.append(inputlist)
            #y_test.append(outputlist)
            writer2.writerow(inputlist + outputlist)#test.csvに書き込み
            test_count = test_count + 1
            if i + input_num + output_num == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print 'finished!!'
def make_dataset_code(code,input_num,output_num):#一定期間の株価から翌日の株価を回帰予測    
    START_TEST_DAY = 20090105 
    train_count = 0
    test_count = 0
    
    filename = "stock(" + str(code) + ").CSV"
    _time = []
    _open = []
    _max = []
    _min = []
    _close = []
    _volume = []
    _keisu = []
    _shihon = []
    filepath = "./stockdata/" + filename
    _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
    
    #start_test_dayでデータセットを分割
    try:
        iday = _time.index(START_TEST_DAY)
    except:
        print "can't find start_test_day"
        
    trainlist = _close[:iday]
    testlist = _close[iday:]
    
    min_price = min(trainlist)
    max_price = max(trainlist)
    #train data
    fw = open(t_folder + 'train(' + str(code) +').csv', 'w')
    writer = csv.writer(fw, lineterminator='\n')
    
    datalist = trainlist
    for i, price in enumerate(datalist):
        inputlist = copy.copy(datalist[i:i + input_num])
        outputlist = copy.copy(datalist[input_num + i:input_num + i + output_num])
        normalizationArray(inputlist,min_price,max_price)
        normalizationArray(outputlist,min_price,max_price)
        
        outputlist.append(min_price)
        outputlist.append(max_price)
        
        writer.writerow(inputlist + outputlist)#train.csvに書き込み
        train_count = train_count + 1
        if i + input_num + output_num == len(datalist):
            break
            
    fw.close()
    
    #test data
    
    fw = open(t_folder + 'test(' + str(code) + ').csv', 'w')
    writer = csv.writer(fw, lineterminator='\n')
    
    datalist = testlist
    for i, price in enumerate(datalist):
        inputlist = copy.copy(datalist[i:i + input_num])
        outputlist = copy.copy(datalist[input_num + i:input_num + i + output_num])
        
        normalizationArray(inputlist,min_price,max_price)
        normalizationArray(outputlist,min_price,max_price)
        
        outputlist.append(min_price)
        outputlist.append(max_price)
        
        writer.writerow(inputlist + outputlist)#test.csvに書き込み
        test_count = test_count + 1
        if i + input_num + output_num == len(datalist):
            break
    fw.close()
            
    
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print 'finished!!'
    
def make_dataset_2():#一定期間の株価から数日後の株価の値上がり率から売買シグナルを出力    
    start_test_day = 20090105 
    input_num = 70   
    next_day = 5#何日後の値上がり率で判断するか
    up_ratio = 5
    down_ratio = -5    
    
    train_count = 0
    test_count = 0
    buy = 0
    sell = 0
    no_ope = 0
    fw1 = open(t_folder + 'tmp_train.csv', 'w')
    fw2 = open(t_folder + 'tmp_test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        del _open
        del _max
        del _min
        del _volume
        del _keisu
        del _shihon
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        trainlist = _close[:iday]
        testlist = _close[iday:]
        if len(trainlist) < input_num or len(testlist) < input_num:
            continue
        
        #train data
        #x_train = []
        #y_train = []
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
                buy += 1
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
                sell += 1
            else:
                if i % 2 ==0:
                    outputlist.append(2)#no_operation
                    no_ope += 1
                else:
                    continue
            

            normalizationArray(inputlist,min(inputlist),max(inputlist))
            
            writer1.writerow(inputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + next_day == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
            else:
                if i % 2 ==0:
                    outputlist.append(2)#no_operation
                    no_ope += 1
                else:
                    continue
            

            normalizationArray(inputlist,min(inputlist),max(inputlist))
            
            writer2.writerow(inputlist + outputlist)#train.csvに書き込み
            test_count = test_count + 1
            if i + input_num + next_day == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print "label in train buy %d, sell %d, no_ope %d" % (buy, sell, no_ope)
    print 'finished!!'
        
def make_dataset_3():#make_dataset2のテクニカル指標入力版    
    start_test_day = 20090105 
    input_num = 50
    tech_input_num = 50
    #all_input_num = input_num + tech_input_num
    param = 14#テクニカル指標のパラメータ
    next_day = 5#何日後の値上がり率で判断するか
    up_ratio = 5
    down_ratio = -5    
    
    train_count = 0
    test_count = 0
    buy = 0
    sell = 0
    no_ope = 0
    fw1 = open(t_folder + 'tmp_tech_train.csv', 'w')
    fw2 = open(t_folder + 'tmp_tech_test.csv', 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは初期化
        del _open
        del _max
        del _min
        del _volume
        del _keisu
        del _shihon
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        rsi = ta.RSI(np.array(_close, dtype='f8'), timeperiod = param)
        rsi = np.ndarray.tolist(rsi)
        #最初の数日はRSIが計算できないのでスライス
        rsi = rsi[param:]
        _close = _close[param:]
        trainlist = _close[:iday]
        techtrainlist = rsi[:iday]
        testlist = _close[iday:]
        techtestlist = rsi[iday:]
        if len(trainlist) < input_num or len(testlist) < input_num:
            continue
        #train data
        #x_train = []
        #y_train = []
        datalist = trainlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            techinputlist = copy.copy(techtrainlist[i:i + tech_input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
                buy += 1
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
                sell += 1
            else:
                #no_operationのデータは多すぎるので２日に一回
                if i % 2 ==0:
                    outputlist.append(2)#no_operation
                    no_ope += 1
                else:
                    continue
            
            
            normalizationArray(inputlist,min(inputlist),max(inputlist))
            normalizationArray(techinputlist,min(techinputlist),max(techinputlist))
            writer1.writerow(inputlist + techinputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + next_day == len(datalist):
                break
            
       
        #test data
        #x_test = []
        #y_test = []
        datalist = testlist
        for i, price in enumerate(datalist):
            inputlist = copy.copy(datalist[i:i + input_num])
            techinputlist = copy.copy(techtestlist[i:i + tech_input_num])
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = datalist[i + input_num + next_day - 1]
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            if (float(predic_price / now_price) - 1) * 100 >=up_ratio:
                outputlist.append(0)#buy
            elif (float(predic_price / now_price) - 1) * 100 <=down_ratio:
                outputlist.append(1)#sell
            else:
                if i % 2 == 0:
                    outputlist.append(2)#no_operation
                else:
                    continue
            
            normalizationArray(inputlist,min(inputlist),max(inputlist))
            normalizationArray(techinputlist,min(techinputlist),max(techinputlist))
            writer2.writerow(inputlist + techinputlist + outputlist)#train.csvに書き込み
            test_count = test_count + 1
            if i + input_num + next_day == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print "label in train buy %d, sell %d, no_ope %d" % (buy, sell, no_ope)
    print 'finished!!'
        
def make_dataset_4(inputnum):#一定期間の株価から数日後の株価の最大値を回帰 
    print 'make_dataset_4'
    start_test_day = 20090105 
    input_num = inputnum   
    next_day = 5#何日後の値上がり率で判断するか
    #up_ratio = 5
    #down_ratio = -5    
    
    train_count = 0
    test_count = 0
    fpath1 = t_folder + 'train' + str(input_num) + '.csv'
    fpath2 = t_folder + 'test' + str(input_num) + '.csv'
    fw1 = open(fpath1, 'w')
    fw2 = open(fpath2, 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        _time = []
        _open = []
        _max = []
        _min = []
        _close = []
        _volume = []
        _keisu = []
        _shihon = []
        filepath = "./stockdata/%s" % f
        _time,_open,_max,_min,_close,_volume,_keisu,_shihon = readfile(filepath)
        #使わないリストは削除
        del _open
        del _max
        del _min
        del _volume
        del _keisu
        del _shihon
        #start_test_dayでデータセットを分割
        try:
            iday = _time.index(start_test_day)
        except:
            print "can't find start_test_day"
            continue#start_test_dayが見つからなければ次のファイルへ
        trainlist = _close[:iday]
        testlist = _close[iday:]
        if len(trainlist) < input_num or len(testlist) < input_num:
            continue
        
        norm_min = min(trainlist)
        norm_max = max(trainlist)
        
        datalist = trainlist
        for i, price in enumerate(datalist):
            if i % 2 == 0:
                #全部は多すぎるので半分
                continue
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            except:
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            outputlist.append((predic_price - now_price) / now_price)
            outputlist.append(norm_min)
            outputlist.append(norm_max)
            

            normalizationArray(inputlist,norm_min,norm_max)
            
            
            writer1.writerow(inputlist + outputlist)#train.csvに書き込み
            train_count = train_count + 1
            if i + input_num + next_day == len(datalist):
                break
            
       
        
        datalist = testlist
        for i, price in enumerate(datalist):
            if i % 2 == 0:
                #全部は多すぎるので半分
                continue
            inputlist = copy.copy(datalist[i:i + input_num])
            
            try:
                now_price = datalist[i + input_num - 1]
                predic_price = max(datalist[i + input_num:i + input_num + next_day -1])
            except:
                #print 'datalist too short'
                continue#datalistが短すぎる場合は飛ばす
            outputlist = []
            outputlist.append((predic_price - now_price) / now_price)
            outputlist.append(norm_min)
            outputlist.append(norm_max)
            
            normalizationArray(inputlist,norm_min,norm_max)
            
            writer2.writerow(inputlist + outputlist)#test.csvに書き込み
            test_count = test_count + 1
            if i + input_num + next_day == len(datalist):
                break
                
            
    fw1.close()
    fw2.close()
    print 'save ' + str(fpath1)
    print 'save ' + str(fpath2)
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print 'finished!!'
    
def make_dataset_5(inputnum, tech_name = None, param1 = None, param2 = None, param3 = None):#一定期間の株価,テクニカル指標から数日後の株価の最大値を回帰 
    print 'make_dataset_5'
    start_test_day = 20090105 
    input_num = inputnum
    next_day = 5#何日後の値上がり率で判断するか
    
    train_count = 0
    test_count = 0
    fpath1 = t_folder + 'train' + tech_name + str(input_num) + '.csv'
    fpath2 = t_folder + 'test' + tech_name + str(input_num) + '.csv'
    fw1 = open(fpath1, 'w')
    fw2 = open(fpath2, 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        try:
            train, test = getTeacherDataTech(f,start_test_day,next_day,input_num,tech_name, param1, param2, param3)
        except:
            continue
        writer1.writerows(train)
        writer2.writerows(test)
            
    fw1.close()
    fw2.close()
    print 'save ' + str(fpath1)
    print 'save ' + str(fpath2)
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print 'finished!!'
    
    
def make_dataset_6(fname,inputnum,next_day=5,stride=2,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):#一定期間の株価,テクニカル指標から数日後の株価の最大値を回帰 
    print 'make_dataset_6'
    start_test_day = 20090105 
    input_num = inputnum
    #next_day = 5#何日後の値上がり率で判断するか
    
    train_count = 0
    test_count = 0
    fpath1 = t_folder + 'train_' + str(fname) + str(input_num) + '.csv'
    fpath2 = t_folder + 'test_' + str(fname) +  str(input_num) + '.csv'
    fw1 = open(fpath1, 'w')
    fw2 = open(fpath2, 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        
        train, test = getTeacherDataMultiTech(f,start_test_day,next_day,input_num,stride=stride,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)
        if (train == -1) or (test == -1):
            print 'skip',f
            continue
        
        test = test[:int(len(test)/2)]
        writer1.writerows(train)
        writer2.writerows(test)
        train_count += len(train)
        test_count += len(test)
    fw1.close()
    fw2.close()
    print 'save ' + str(fpath1)
    print 'save ' + str(fpath2)
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    print 'finished!!'
    
def make_dataset_7(fname,inputnum,next_day=5,stride=2,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):#一定期間の株価,テクニカル指標から数日後の株価の最大値でクラス分類
    print 'make_dataset_7'
    start_test_day = 20100104
    input_num = inputnum
    #next_day = 5#何日後の値上がり率で判断するか
    
    train_count = 0
    test_count = 0
    fpath1 = t_folder + "tmp_train.csv"
    fpath2 = t_folder + "tmp_test.csv"
    fw1 = open(fpath1, 'w')
    fw2 = open(fpath2, 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        
        train, test = getTeacherDataMultiTech_label(f,start_test_day,next_day,input_num,stride=stride,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)
        if (train == -1) or (test == -1):
            print 'skip',f
            continue
            
        writer1.writerows(train)
        writer2.writerows(test)
            
    fw1.close()
    fw2.close()
    print 'save ' + str(fpath1)
    print 'save ' + str(fpath2)
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    
    arrange_train_num("tmp_train.csv", 'train_' + str(fname) + str(input_num) + '_class.csv')
    arrange_train_num("tmp_test.csv", 'test_' + str(fname) +  str(input_num) + '_class.csv') 
    
    print 'finished!!'
    
def make_dataset_8(fname,inputnum,next_day=5,stride=2,u_vol=False,u_ema=False,u_rsi=False,u_macd=False,u_stoch=False,u_wil=False):#一定期間の株価,テクニカル指標から数日後の株価の最大値を回帰
    #make_dataset_6の学習データ数調整版
    print 'make_dataset_8'
    start_test_day = 20100104
    input_num = inputnum
    #next_day = 5#何日後の値上がり率で判断するか
    
    train_count = 0
    test_count = 0
    fpath1 = t_folder + "tmp_train.csv"
    fpath2 = t_folder + "tmp_test.csv"
    fw1 = open(fpath1, 'w')
    fw2 = open(fpath2, 'w')
    writer1 = csv.writer(fw1, lineterminator='\n')
    writer2 = csv.writer(fw2, lineterminator='\n')
    
    files = os.listdir("./stockdata")
    for k, f in enumerate(files):
        print f, k
        
        train, test = getTeacherDataMultiTech(f,start_test_day,next_day,input_num,stride=stride,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)
        if (train == -1) or (test == -1):
            print 'skip',f
            continue
            
        writer1.writerows(train)
        writer2.writerows(test)
        train_count += len(train)
        test_count += len(test)
    fw1.close()
    fw2.close()
    print 'save ' + str(fpath1)
    print 'save ' + str(fpath2)
    print "train_count = %d" % train_count
    print "test_count = %d" % test_count
    
    arrange_train_num2("tmp_train.csv", 'train_' + str(fname) + str(input_num) + '_reg.csv')
    arrange_train_num2("tmp_test.csv", 'test_' + str(fname) +  str(input_num) + '_reg.csv') 
    
    print 'finished!!'
    
if __name__ == '__main__':
    print "start make dataset"
    #getTeacherDataTech('stock(9984).CSV',20090105,5,10,'EMA',10)
    #print "end!"
    #raw_input()
    #make_dataset_6('volemarsistoch_n10_',30,next_day=10,u_vol=True,u_ema=True,u_rsi=True,u_stoch=True)
    #make_dataset_6('vol2ema_n5_',30,stride=4,next_day=5,u_vol=True,u_ema=True)
    #make_dataset_6('volRsiStoch_n5_',30,stride=4,next_day=5,u_vol=True,u_rsi=True,u_stoch=True)
    make_dataset_7('priceonly_n5_',30,next_day=5)
    #make_dataset_7('vol2Ema_n5_',30,next_day=5,u_vol=True,u_ema=True)
    #make_dataset_6('volRsiStoch_m_n5_',30,next_day=5,u_vol=True,u_rsi=True,u_stoch=True)
    #make_dataset_6('macdtest',30,u_macd=True)
    #make_dataset_6('ematest',30,u_ema=True)
    #make_dataset_6('ocirator',30,u_vol=True,u_rsi=True,u_stoch=True,u_wil=True)
    #make_dataset_6('volRsiStoch',30,u_vol=True,u_rsi=True,u_stoch=True)
    print 'finished!!!!!!!!!!!!!!'
    """
    raw_input()
    for i in xrange(10,101,10):
        make_dataset_5(i,tech_name="VOL",param1=0)
        make_dataset_5(i,"RSI",param1 = 14)
        make_dataset_5(i,"EMA",param1 = 10)
        make_dataset_5(i,"MACD",param1 = 12, param2 = 26, param3 = 9)
        make_dataset_5(i,"WILLR",param1 = 14)
    """
    #arrange_train_num("tmp_tech_train.csv", "train70.csv")
    #arrange_train_num("tmp_tech_test.csv", "test70.csv") 
    
    #data_completion()
    print "finished make dataset"
    
        
        
        
        
        
        
        