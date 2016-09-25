# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle
import env_stockmarket
import dqn_agent_nature
import tools
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda

def trading_files(files):
    pass
    

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('model',help='path of using agent')
parser.add_argument('--input_num', '-in', default=60, type=int,
                    help='input node number')
parser.add_argument('--channel', '-c', default=1, type=int,
                    help='data channel')
parser.add_argument('--experiment_name', '-n', default='experiment',type=str,help='experiment name')
parser.add_argument('--u_vol', '-vol',type=int,default=1,
                    help='use vol or no')
parser.add_argument('--u_ema', '-ema',type=int,default=1,
                    help='use ema or no')
parser.add_argument('--u_rsi', '-rsi',type=int,default=1,
                    help='use rsi or no')
parser.add_argument('--u_macd', '-macd',type=int,default=0,
                    help='use macd or no')
parser.add_argument('--u_stoch', '-stoch',type=int,default=1,
                    help='use stoch or no')
parser.add_argument('--u_wil', '-wil',type=int,default=1,
                    help='use wil or no')
                    
args = parser.parse_args()

if args.u_vol == 0: u_vol = False
elif args.u_vol == 1: u_vol = True
if args.u_ema == 0: u_ema = False
elif args.u_ema == 1: u_ema = True
if args.u_rsi == 0: u_rsi = False
elif args.u_rsi == 1: u_rsi = True
if args.u_macd == 0: u_macd = False
elif args.u_macd == 1: u_macd = True
if args.u_stoch == 0: u_stoch = False
elif args.u_stoch == 1: u_stoch = True
if args.u_wil == 0: u_wil = False
elif args.u_wil == 1: u_wil = True


if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"

folder = './test_result/' + args.experiment_name + '/'
if os.path.isdir(folder) == True:
    print 'this experiment name is existed'
    print 'please change experiment name'
    raw_input()
else:
    print 'make experiment folder'
    os.makedirs(folder)
    
    
END_TRAIN_DAY = 20081230
#START_TEST_DAY = 20090105
START_TEST_DAY = 20100104

#モデルの読み込み
Agent = dqn_agent_nature.dqn_agent(state_dimention=args.input_num * args.channel + 2,)
Agent.agent_init()
Agent.DQN.load_model(args.model)
Agent.policyFrozen = True
    
market = env_stockmarket.StockMarket(END_TRAIN_DAY,START_TEST_DAY,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)

files = os.listdir("./nikkei100")

Agent.init_max_Q_list()
Agent.init_reward_list()
profit_list = []


for f in files:
    print f
    stock_agent = env_stockmarket.Stock_agent(Agent)
    
    try:
        testdata,testprice = market.get_testData(f,args.input_num)
        #testdata, testprice = market.get_trainData(f,END_TRAIN_DAY,args.input_num)
    except:
        print 'skip',f
        continue
        
    profit_ratio, proper, order, stocks, price = stock_agent.trading_test(args.input_num,testprice,testdata)
    profit_list.append(profit_ratio)
    

    tools.listToCsv(folder+str(f).replace(".CSV", "")+'.csv', price, proper, order,stocks)
    
    buy_order, sell_order = tools.order2buysell(order,price)

    #2軸使用
    fig, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.set_ylabel('price')
    axis1.set_ylabel('buy')
    axis1.set_ylabel('sell')
    axis2.set_ylabel('property')
    axis1.plot(price, label = "price")
    axis1.plot(buy_order,'o',label='buy')
    axis1.plot(sell_order,'^',label='sell')
    axis1.legend(loc = 'upper left')
    axis2.plot(proper, label = 'property', color = 'g')
    axis2.legend()
    filename = folder + str(f).replace(".CSV", "") + ".png"
    plt.savefig(filename)
    plt.close()
    
print 'average profit:', sum(profit_list)/len(profit_list)