# -*- coding: utf-8 -*-
import glob
import os
import shutil
import numpy as np
import csv

def get_nikkei255_file():
    
    f = open('nikkei255.txt','r')
    for row in f:
        
        code = row[:4]
        print code
        file_name_list = glob.glob('./stockdata/*'+str(code)+'*')
        print file_name_list
        if len(file_name_list) == 0:
            print 'no file code', code
        else:
            file_name = file_name_list[0]
            print file_name,' copy'
            shutil.copy(file_name,'./nikkei255')
        #raw_input()
        
    f.close()
    
def listToCsv(filename,*args):
    data = []
    for i in range(len(args)):
        data.append(args[i])
    
    
    data = np.array(data).transpose()
    #print data
    fw = open(filename, 'w')
    writer = csv.writer(fw)
    writer.writerows(data)
    fw.close()
    
    print 'saved ' + str(filename)
    
def order2buysell(order,price):
    buy_point = []
    sell_point = []
    for i,o in enumerate(order):
        if o == 1:
            buy_point.append(price[i])
            sell_point.append(np.nan)
        elif o == -1:
            buy_point.append(np.nan)
            sell_point.append(price[i])
        elif o == 0:
            buy_point.append(np.nan)
            sell_point.append(np.nan)
            
    return buy_point, sell_point
    
if __name__ == "__main__":
    
    get_nikkei255_file()