# -*- coding: utf-8 -*-

import os
import time
import argparse
import env_stockmarket
import dqn_agent_nature

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
n_epoch = 1

start_time = time.clock()

Agent = dqn_agent_nature.dqn_agent(state_dimention=args.input_num*args.channel)
Agent.agent_init()

market = env_stockmarket.StockMarket()

print 'epoch:', n_epoch
files = os.listdir("./stockdata")
for epoch in range(1,n_epoch + 1):
    print('epoch', epoch),
    print 'time:%d[s]' % (time.clock() - start_time)
    print 'epoch!!!', epoch
    raw_input()
    for f in files:
        print f
        stock_agent = env_stockmarket.Stock_agent(Agent)
        traindata,trainprice = market.get_trainData(f,END_TRADING_DAY,args.input_num)
        stock_agent.trading(args.input_num,trainprice,traindata)
        