# -*- coding: utf-8 -*-

import os
import time
import argparse
import env_stockmarket
import dqn_agent_nature
import tools
import numpy as np
from chainer import cuda

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input_num', '-in', default=60, type=int,
                    help='input node number')
parser.add_argument('--channel', '-c', default=1, type=int,
                    help='data channel')
parser.add_argument('--experiment_name', '-n', default='experiment',type=str,help='experiment name')
parser.add_argument('--u_vol', '-vol',type=int,default=0,
                    help='use vol or no')
parser.add_argument('--u_ema', '-ema',type=int,default=0,
                    help='use ema or no')
parser.add_argument('--u_rsi', '-rsi',type=int,default=0,
                    help='use rsi or no')
parser.add_argument('--u_macd', '-macd',type=int,default=0,
                    help='use macd or no')
parser.add_argument('--u_stoch', '-stoch',type=int,default=0,
                    help='use stoch or no')
parser.add_argument('--u_wil', '-wil',type=int,default=0,
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

folder = './train_result/' + args.experiment_name + '/'
if os.path.isdir(folder) == True:
    print 'this experiment name is existed'
    print 'please change experiment name'
    raw_input()
else:
    print 'make experiment folder'
    os.makedirs(folder)


END_TRADING_DAY = 20081230
n_epoch = 1000

start_time = time.clock()

Agent = dqn_agent_nature.dqn_agent(state_dimention=args.input_num * args.channel + 2)
Agent.agent_init()

market = env_stockmarket.StockMarket(u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)
ave_Q = []
ave_reward = []
ave_profit = []
var_Q = []
var_reward = []
var_profit = []

print 'epoch:', n_epoch
files = os.listdir("./nikkei10")
for epoch in range(1,n_epoch + 1):
    profit_list = []
    print('epoch', epoch),
    print 'time:%d[s]' % (time.clock() - start_time)
    print 'epoch!!!', epoch
    
    for f in files:
        print f
        stock_agent = env_stockmarket.Stock_agent(Agent)
        
        try:
            traindata,trainprice = market.get_trainData(f,END_TRADING_DAY,args.input_num)
        except:
            print 'skip',f
            continue
            
        profit_ratio = stock_agent.trading(args.input_num,trainprice,traindata)
        profit_list.append(profit_ratio)
        
    ave_Q.append(Agent.get_average_Q())
    ave_reward.append(Agent.get_average_reward())
    ave_profit.append(sum(profit_list)/len(profit_list))
    var_Q.append(Agent.get_variance_Q())
    var_reward.append(Agent.get_varance_reward())
    var_profit.append(np.var(np.array(profit_list)))
    
    tools.listToCsv(folder+'log.csv', ave_Q, var_Q, ave_reward, var_reward, ave_profit, var_profit)
    
    if epoch % 10 == 0:
        Agent.DQN.save_model(folder, epoch)