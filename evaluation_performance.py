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

def eval_performance(market,model,target_folder,input_num):

    print 'start evaluating...'
    Agent = dqn_agent_nature.dqn_agent()
    Agent.agent_init()
    Agent.DQN.load_model(model)
    Agent.policyFrozen = True
    
    
    profit_list = []
    test_profit_list = []
    files = os.listdir(target_folder)
    
    #train term evaluation
    for f in files:
        
        stock_agent = env_stockmarket.Stock_agent(Agent)
        try:
            traindata,trainprice = market.get_trainData(f,input_num)
        except:
            continue
            
        profit_ratio = stock_agent.trading(args.input_num,trainprice,traindata)
        profit_list.append(profit_ratio)
        
    train_ave = np.mean(np.array(profit_list))
    train_ave_Q = Agent.get_average_Q()
    train_ave_reward = Agent.get_average_reward()
    
    #test term evaluation
    for f in files:
        
        stock_agent = env_stockmarket.Stock_agent(Agent)
        try:
            traindata,trainprice = market.get_testData(f,input_num)
        except:
            continue
            
        profit_ratio = stock_agent.trading(args.input_num,trainprice,traindata)
        test_profit_list.append(profit_ratio)
        
    
    test_ave = np.mean(np.array(test_profit_list))
    
    return train_ave, test_ave, train_ave_Q, train_ave_reward