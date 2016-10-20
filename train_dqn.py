# -*- coding: utf-8 -*-

import os
import time
import argparse
import env_stockmarket
import dqn_agent_nature
import tools
import numpy as np
from chainer import cuda
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import evaluation_performance



parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--data_folder', '-f', type=str, default='./nikkei100',
                    help='data size of history')
parser.add_argument('--input_num', '-in', default=60, type=int,
                    help='input node number')
parser.add_argument('--channel', '-c', default=8, type=int,
                    help='data channel')
parser.add_argument('--experiment_name', '-n', default='experiment',type=str,help='experiment name')
parser.add_argument('--batchsize', '-B', type=int, default=100,
                    help='replay size')
parser.add_argument('--historysize', '-D', type=int, default=10**5,
                    help='data size of history')
parser.add_argument('--epsilon_discount_size', '-eds', type=int, default=10**6,
                    help='data size of history')
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
parser.add_argument('--targetFlag', '-tf',type=int,default=1,
                    help='target flag')
                    
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

if args.targetFlag == 0: targetFlag = False
elif args.targetFlag == 1: targetFlag = True


if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    print "use gpu"

folder = './train_result/' + args.experiment_name + '/'
if os.path.isdir(folder) == True:
    print 'this experiment name is existed'
    print 'please change experiment name'
    raw_input()
else:
    print 'make experiment folder'
    os.makedirs(folder)


END_TRAIN_DAY = 20081230
START_TEST_DAY = 20090105
n_epoch = 1000

start_time = time.clock()

Agent = dqn_agent_nature.dqn_agent(gpu_id=args.gpu,state_dimention=args.input_num * args.channel + 2,batchsize=args.batchsize,historysize=args.historysize,epsilon_discount_size=args.epsilon_discount_size,targetFlag = targetFlag)
Agent.agent_init()

market = env_stockmarket.StockMarket(END_TRAIN_DAY,START_TEST_DAY,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)

evaluater = evaluation_performance.Evaluation(args.gpu,market,args.data_folder,folder,args.input_num)



print 'epoch:', n_epoch

with open(folder + 'settings.txt', 'wb') as o:
    o.write('epoch:' + str(n_epoch) + '\n')
    o.write('data_folder:' + str(args.data_folder) + '\n')
    o.write('input:' + str(args.input_num) + '\n')
    o.write('channel:' + str(args.channel) + '\n')
    #o.write('hidden:' + str(model.hidden_num) + '\n')
    #o.write('layer_num:' + str(model.layer_num) + '\n')
    o.write('batchsize:' + str(args.batchsize) + '\n')
    o.write('historysize:' + str(args.historysize) + '\n')
    o.write('epsilon_discount_size:' + str(args.epsilon_discount_size) + '\n')
    o.write('targetFlag:' + str(targetFlag) + '\n')
    
    
files = os.listdir(args.data_folder)
for epoch in tqdm(range(1,n_epoch + 1)):
    Agent.init_max_Q_list()
    Agent.init_reward_list()


    #ファイルの順をシャッフル
    random.shuffle(files)
    #train_loop
    Agent.policyFrozen = False
    for f in tqdm(files):

        stock_agent = env_stockmarket.Stock_agent(Agent)
        
        try:
            traindata,trainprice = market.get_trainData(f,args.input_num)
        except:
            continue
            
        profit_ratio = stock_agent.trading(args.input_num,trainprice,traindata)

    #model evaluation
    eval_model = Agent.DQN.get_model_copy()
    evaluater.eval_performance(eval_model)
    evaluater.get_epsilon(Agent.epsilon)
    evaluater.save_eval_result()

    if epoch % 1 == 0:
        
        Agent.DQN.save_model(folder,epoch)