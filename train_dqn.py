# -*- coding: utf-8 -*-

import os
import time
import argparse
import env_stockmarket
import dqn_agent_nature
import tools

from chainer import cuda

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input_num', '-in', default=60, type=int,
                    help='input node number')
parser.add_argument('--channel', '-c', default=1, type=int,
                    help='data channel')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"


END_TRADING_DAY = 20081230
n_epoch = 1000

start_time = time.clock()

Agent = dqn_agent_nature.dqn_agent(state_dimention=args.input_num * args.channel + 2)
Agent.agent_init()

market = env_stockmarket.StockMarket()
ave_Q = []
ave_reward = []
ave_profit = []

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
    
    tools.listToCsv('log.csv', ave_Q, ave_reward,ave_profit)
    
    if epoch % 10 == 0:
        Agent.DQN.save_model(epoch)