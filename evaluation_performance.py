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
import talib as ta

class Evaluation():

    def __init__(self,gpu_id,market, target_folder,result_folder, input_num):
        self.gpu_id = gpu_id
        self.market = market
        self.target_folder = target_folder
        self.input_num = input_num
        self.result_folder = result_folder
        
        self.train_ave_profit_list = []
        self.test_ave_profit_list = []
        self.ave_Q_list = []
        self.ave_reward_list = []
        self.epsilon_list = []
        
        
    def eval_performance(self,model):

        print 'start evaluating...'
        Agent = dqn_agent_nature.dqn_agent(gpu_id = self.gpu_id,state_dimention=1)
        Agent.agent_init()
        Agent.DQN.model = model
        Agent.DQN.model_to_gpu()
        Agent.policyFrozen = True
        
        
        profit_list = []
        test_profit_list = []
        files = os.listdir(self.target_folder)
        
        #train term evaluation
        for f in files:
            
            stock_agent = env_stockmarket.Stock_agent(Agent)
            try:
                traindata,trainprice = self.market.get_trainData(f,self.input_num)
            except:
                continue
                
            profit_ratio = stock_agent.trading(self.input_num,trainprice,traindata)
            profit_list.append(profit_ratio)
            
        train_ave = np.mean(np.array(profit_list))
        train_ave_Q = Agent.get_average_Q()
        train_ave_reward = Agent.get_average_reward()
        
        #test term evaluation
        for f in files:
            
            stock_agent = env_stockmarket.Stock_agent(Agent)
            try:
                traindata,trainprice = self.market.get_testData(f,self.input_num)
            except:
                continue
                
            profit_ratio = stock_agent.trading(self.input_num,trainprice,traindata)
            test_profit_list.append(profit_ratio)
            
        
        test_ave = np.mean(np.array(test_profit_list))
        
        self.train_ave_profit_list.append(train_ave)
        self.test_ave_profit_list.append(test_ave)
        self.ave_Q_list.append(train_ave_Q)
        self.ave_reward_list.append(train_ave_reward)
        
        return train_ave, test_ave, train_ave_Q, train_ave_reward
        
    def save_eval_result(self):
        tools.listToCsv(self.result_folder+'log.csv',self.train_ave_profit_list,self.ave_Q_list,self.ave_reward_list,self.test_ave_profit_list,self.epsilon_list)
        
        #2軸使用
        fig, axis1 = plt.subplots()
        axis2 = axis1.twinx()
        axis1.set_ylabel('ave_max_Q')
        axis2.set_ylabel('epsilon')
        axis1.plot(self.ave_Q_list, label = "ave_max_Q")
        axis1.legend(loc = 'upper left')
        axis2.plot(self.epsilon_list, label = 'epsilon', color = 'g')
        axis2.legend()
        filename = self.result_folder + "log_ave_max_Q.png"
        plt.grid(which='major')
        plt.savefig(filename)
        plt.close()
        
        #2軸使用
        fig, axis1 = plt.subplots()
        axis2 = axis1.twinx()
        axis1.set_ylabel('ave_reward')
        axis2.set_ylabel('epsilon')
        axis1.plot(self.ave_reward_list, label = "ave_reward")
        reward_ema = ta.EMA(np.array(self.ave_reward_list,dtype='f8'),timeperiod=10)
        axis1.plot(reward_ema, color = 'r')
        axis1.legend(loc = 'upper left')
        axis2.plot(self.epsilon_list, label = 'epsilon', color = 'g')
        axis2.legend()
        filename = self.result_folder + "log_ave_reward.png"
        plt.grid(which='major')
        plt.savefig(filename)
        plt.close()
        
        #2軸使用
        fig, axis1 = plt.subplots()
        axis2 = axis1.twinx()
        axis1.set_ylabel('ave_train_profit')
        axis2.set_ylabel('epsilon')
        axis1.plot(self.train_ave_profit_list, label = "ave_train_profit")
        train_ema = ta.EMA(np.array(self.train_ave_profit_list,dtype='f8'),timeperiod=10)
        axis1.plot(train_ema, color = 'r')
        axis1.legend(loc = 'upper left')
        axis2.plot(self.epsilon_list, label = 'epsilon', color = 'g')
        axis2.legend()
        filename = self.result_folder + "log_ave_train_profit.png"
        plt.grid(which='major')
        plt.savefig(filename)
        plt.close()
        
        #2軸使用
        fig, axis1 = plt.subplots()
        axis2 = axis1.twinx()
        axis1.set_ylabel('ave_test_profit')
        axis2.set_ylabel('epsilon')
        axis1.plot(self.test_ave_profit_list, label = "ave_test_profit")
        test_ema = ta.EMA(np.array(self.test_ave_profit_list,dtype='f8'),timeperiod=10)
        axis1.plot(test_ema, color = 'r')
        axis1.legend(loc = 'upper left')
        axis2.plot(self.epsilon_list, label = 'epsilon', color = 'g')
        axis2.legend()
        filename = self.result_folder + "log_ave_test_profit.png"
        plt.savefig(filename)
        plt.close()
        
    def get_epsilon(self, epsilon):
        self.epsilon_list.append(epsilon)
        