# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 01:26:48 2019

@author: 佟斌
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time 
from sklearn import preprocessing



np.random.seed()
np.set_printoptions(precision=2,threshold=20000)
df=pd.read_csv("SP500_Return.csv",index_col=1)
pp=10#就选前面10个的股票把
rand=np.random.choice(df.columns[1:-6],pp,0)
#stock_names=['GOOG_price','AAPL_price','IBM_price']+rand.tolist()
stock_names=[]+rand.tolist()

while 1:
    if len(set(stock_names))!=pp:
        rand=np.random.choice(df.columns[1:-6],pp-3,0)
        #stock_names=['GOOG_price','AAPL_price','IBM_price']+rand.tolist()
        stock_names=[]+rand.tolist()
    else:
        break
df['AAPL_price']=df.index
df.index=range(df.shape[0])
#['SLG_price', 'JWN_price', 'UNP_price']

stocks=stock_names+['Mkt-RF',	'SMB','HML','RMW','CMA','RF']
df=df.loc[:,['date']+stocks]

df_normal=pd.DataFrame.copy(df)
df_normal.iloc[:,1::]=preprocessing.scale(df.iloc[:,1::])

class game(): #each episode to create a game object
    
    def __init__(self,df,begin,steps,ini_val=1,pp=2,df_af_norm=df_normal): #begin is the starting point in dataframe， step is the rolling window period of each episode
        
        self.df_FF=df.loc[begin:begin+steps,:]  #fama-french 3 factor model
        
        self.norm_df=df_af_norm
        
        
        
        self.makov_df=df.loc[begin-steps:begin+steps,:] # 
        #print(self.df_FF.iloc[:,1])
        self.df=self.df_FF.iloc[:,0:pp]  #前3个市场的return

        
        self.mdp_rf=np.mean(self.df_FF['RF'])
        self.mkv_rf=np.mean(self.makov_df['RF'])
        self.indx=pp
        
        self.begin=begin
        self.steps=steps
        
        
        
 
        
      
        
        
        self.record=dict()
        self.record['step']=0
        self.record['cum']=ini_val #initially our porttfolio is 1
        self.record['current_states']=self.df.iloc[self.record['step']+1,1::]

        
        #self.input_state=[1/(pp-1)]*(pp-1)+self.record['current_states'].tolist()+[1,1,0]+self.df_FF.iloc[self.record['step'],self.indx::].tolist() #只输入fama frech model
        self.input_state=[1/(pp-1)]*(pp-1)+self.norm_df.iloc[self.record['step']+1,1:self.indx].tolist()+[1,1,0]+self.norm_df.iloc[self.record['step'],self.indx::].tolist() #只输入fama frech model
        
        
        self.single_stocks=np.ones(pp-1)
        
        
        
        
        
        self.input_collect=[self.input_state]
        #记录
        self.makoviz_record={}
        self.makoviz_record['return']=[]
        self.makoviz_record['weights']=[]
        self.mdp_record={}
        self.mdp_record['return']=[]
        self.mdp_record['weights']=[]
        #记录
        
        self.position=np.array([0.5,0.5])
        

        
        self.vol=dict()
        self.vol['mdp']=[]
        self.vol['mkv']=[]
        
        self.sharp=dict()
        self.sharp['mdp']=[]
        self.sharp['mkv']=[]
        
        self.max_draw=dict()
        self.max_draw['mdp']=[]
        self.max_draw['mkv']=[]
        
        
        
        
        
        
        self.record['next_states']=self.df.iloc[self.record['step']+1,1::] #next state
    
        
        self.record['markovitz rate']=1  #the control group that we invest 33.33% in each stock everyday instead of MDP 
        self.cum_collect=[1]
        self.mkv_collect=[1]
        
        self.reward_collect=[]
        

      
    def take_step(self,action): #simulate OpenAI 's evironment.step()  to return reward and change to the next state

        #calculate markovitz model:
        tp=self.makov_df.iloc[self.record['step']:self.record['step']+self.steps:,1:self.indx]

        cov_matrix=np.cov(tp.T)
        
        
        inv=np.linalg.inv(cov_matrix)
        vec1=np.ones(self.indx-1).reshape(self.indx-1,1)
        markoviz_weights=inv@vec1/(vec1.T@inv@vec1)
        self.makoviz_record['weights'].append(markoviz_weights.ravel())
        self.makoviz_record['return'].append(self.record['next_states'].tolist())
        self.mdp_record['weights'].append(self.record['next_states'])
        self.mdp_record['return'].append(action)

        
        #MDP model
        
        self.record['step']+=1
        
        
        #self.position+=np.array(action)
        #reward=np.dot(self.record['cum']*self.position,self.record['next_states'].tolist()) #our model's true gain 
        reward=np.dot(self.record['cum']*np.array(action),self.record['next_states'].tolist()) #our model's true gain 
        self.mdp_record['weights'].append(action)
        self.mdp_record['return'].append(self.record['next_states'].tolist())
        

        markovitz_reward=np.dot(self.record['markovitz rate']*markoviz_weights.ravel(),self.record['next_states'].tolist())  #evenly allocated gain
        #self.record['market rate']+=even_reward
        self.record['cum']+=reward
        self.record['markovitz rate']+=markovitz_reward
        
        
        self.single_stocks=(self.single_stocks*(1+self.record['next_states']))
        
        
        
        
        self.record['current_states']=self.record['next_states']
        self.record['next_states']=self.df.iloc[self.record['step']+1,1::]
        
        
        
        #print(self.input_state)

        
        self.reward_collect.append(reward)
        #final_reward=reward-markovitz_reward #our MDP model reward minus evenly investment reward
        
        
        reward-=markovitz_reward
        #self.input_state=(self.position.tolist()+[self.record['cum']]+[self.record['markovitz rate']]+[reward]+self.df_FF.iloc[self.record['step'],3::].tolist())
        
        

        self.input_state=(action+self.norm_df.iloc[self.record['step']+1,1:self.indx].tolist()+[self.record['cum']]+[self.record['markovitz rate']]+[reward]+self.norm_df.iloc[self.record['step'],self.indx::].tolist())
        self.input_collect.append(self.input_state)
  
        #print(len(self.input_state))
        #进行归一化
# =============================================================================
#         if len(self.input_collect)>=5:
#             
#             k=preprocessing.scale(np.array(self.input_collect))[-1,6:9].tolist()
#         
#         #print("这是之前input  ",self.input_state)
#             self.input_state=(action+self.record['current_states'].tolist()+k+self.df_FF.iloc[self.record['step'],self.indx::].tolist())
#         #print("这是之后input  ",self.input_state)
# =============================================================================
        
        
        self.cum_collect.append(self.record['cum']) #record each day's position (original is 1)
        self.mkv_collect.append(self.record['markovitz rate'])
        
        
        
        #数据统计：
        self.vol['mdp'].append(reward)
        self.vol['mkv'].append(markovitz_reward)
        
        
        mdp_max,mdp_min=1,-100
        
        for j in range(len(self.cum_collect)):
            mdp_max=np.max([mdp_max,self.cum_collect[j]])
            mdp_min=np.max([mdp_max-self.cum_collect[j],mdp_min]) 
        
 



        util=0
        #aim=np.max([np.max(self.single_stocks),1.3**(self.record['step']/self.steps)])
        aim=np.sort(self.single_stocks)[-1]
        aim2=np.sort(self.single_stocks)[-3]
        
        if reward>0:

          
            util=0.1
            if self.record['cum']>aim and self.record['step']>10:
               
                util+=(0.1*(aim**(self.record['step']/self.steps)))
            
            if self.record['cum']<aim2 and self.record['step']>10:
     
                util=0.02
        else:
           
            util=-0.05
            if self.record['cum']<aim2 and self.record['step']>10:
                util-=(0.1*(aim**(self.record['step']/self.steps)))
       
                
        if mdp_min>0.5:
            util-=0.2
            















       
  
#以下是复现的
            
# =============================================================================
#             
#         util=0
#         aim=1.5
#         aim2=1.1
#         if reward>0:
# 
#           
#             util=0.1
#             if self.record['cum']>aim**(self.record['step']/self.steps) and self.record['step']>50:
#                 util=0.2*(aim**(self.record['step']/self.steps))
#         else:
#            
#             util=-0.05
#             if self.record['cum']<aim2**(self.record['step']/self.steps) and self.record['step']>50:
#                 util=-0.2*(aim**(self.record['step']/self.steps))
# =============================================================================
        #print(util)
        return [reward/100,util]

    
    def is_done(self): # whether or not reach the maximum steps of each episode
        if self.record['step']!=self.steps-1:
            return 0
        else:
          
            return 1

        
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.998,
            e_greedy=0.95,
            replace_target_iter=300,
            memory_size=500,
            batch_size=30,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer =tf.constant_initializer(0.1), tf.constant_initializer(0.1)
        #tf.random_normal_initializer(0., 0.3)
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 100, tf.nn.tanh, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e2')
            #e3 = tf.layers.dense(e2, 50, tf.nn.tanh, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e3')
            #e4 = tf.layers.dense(e3, 25, tf.nn.tanh, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e4')
            self.q_eval = tf.layers.dense(e2, self.n_actions,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 100, tf.nn.tanh, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t2')
            #t3 = tf.layers.dense(t2, 50, tf.nn.tanh, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t3')
            #t4 = tf.layers.dense(t3, 25, tf.nn.tanh, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t4')
            self.q_next = tf.layers.dense(t2,self.n_actions,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t5')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdagradOptimizer(0.0001).minimize(self.loss)
            #self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
            # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
            #self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
            #self._train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        
        
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
       
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        
        observation = np.array(observation).reshape(1,pp*2+9)
        #print(observation)
        actions_value=None
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            #print(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return [action,actions_value]

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size,replace=0)
            #print(self.memory_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

        
ALL_ACTION=[]  # enumerate action space
for i in np.arange(0,1.01,0.2):
    for j in np.arange(0,1-i+0.01,0.2):
        ALL_ACTION.append([i,j,1-i-j])  
        
ALL_ACTION=[i.tolist() for i in np.eye(pp)]
ALL_ACTION.append([0]*pp)

num_ba=200

        
RL = DeepQNetwork(n_actions=len(ALL_ACTION),
                  n_features=pp*2+6+3,
                  learning_rate=0.003, e_greedy=0.99,
                  replace_target_iter=5, memory_size=num_ba,batch_size=num_ba,
                  e_greedy_increment=0.0003,)      

total_steps = 0
MAX_EP_STEPS=252*2 # abotu one year investment period



reward_collect=[]  #
market_collect=[]  #evenly allocation's return collection
true_collect=[]    #deep Q learning's return cllection

cost_collect=[]
record=pd.DataFrame()
your_collect,market_collect=[],[]

game_info_colelct=dict()
game_info_colelct['cum']=[]
game_info_colelct['each']=[]

game_mdp_vol=[]
game_mkv_vol=[]
game_mdp_sharp=[]
game_mkv_sharp=[]
game_mdp_draw=[]
game_mkv_draw=[]
start=[]
end=[]


action_collect=[]



stock_collect=[]

for i_episode in range(MAX_EP_STEPS,3103-MAX_EP_STEPS,1):

    youxi=game(df,i_episode,MAX_EP_STEPS,1,pp+1,df_normal)
    observation =youxi.input_state
    ep_r = 0
    while True:

        kaka = RL.choose_action(observation)
        action=kaka[0]
        reward=youxi.take_step(ALL_ACTION[action])[1]
        observation_=youxi.input_state
        done=youxi.is_done()

        RL.store_transition(observation, action, reward, observation_)
        action_collect.append(action)
        ep_r += (reward*(0.998**youxi.record['step']))
        if total_steps > 200: #这一部分是过了20就每次rolling window一样的learn
            RL.learn()
            
            

        if done:
            #计算其他数据：
            stock_collect.append(youxi.single_stocks)
            game_mdp_vol.append(np.std(youxi.vol['mdp']))
            game_mkv_vol.append(np.std(youxi.vol['mkv']))
            game_mdp_sharp.append(((youxi.record['cum']-1)/252-youxi.mdp_rf/252)/np.std(youxi.vol['mdp']))
            game_mkv_sharp.append(((youxi.record['markovitz rate']-1)/252-youxi.mkv_rf/252)/np.std(youxi.vol['mkv']))
            
            mdp_max,mdp_min=1,-100
            mkv_max,mkv_min=1,-100
            
            for j in range(len(youxi.cum_collect)):
                mdp_max=np.max([mdp_max,youxi.cum_collect[j]])
                mdp_min=np.max([mdp_max-youxi.cum_collect[j],mdp_min])  #这个测试的是max drawdown
                mkv_max=np.max([mkv_max,youxi.mkv_collect[j]])
                mkv_min=np.max([mkv_max-youxi.mkv_collect[j],mkv_min]) 
            game_mdp_draw.append(mdp_min)
            game_mkv_draw.append(mkv_min)
            
            start.append(youxi.df.loc[youxi.begin,'date'])
            end.append(youxi.df.loc[youxi.begin+252,'date'])
           
            
            plt.figure(figsize=(10,10))
            plt.plot(your_collect)
            plt.plot(market_collect)
            plt.legend(['your','markov'])
            plt.show()
            
            souji=[]
            for k in range(len(ALL_ACTION)):
                souji.append(sum(np.array(action_collect[-504::])==k))

            plt.figure()
            plt.plot(souji)
            print(str(i_episode))
            plt.legend(str(i_episode))
            plt.show()
            
            
            
            
            
            
            
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 4),
                  ' cost: ', round(np.mean(RL.cost_his), 4)," market return: ",round(youxi.record['markovitz rate']-1,4)," your return: ",round(youxi.record['cum']-1,4),'mdp draw: ',game_mdp_draw[-1],'mkv draw: ',game_mkv_draw[-1])
            break

        observation = observation_
        total_steps += 1
        
    

    
    
    market_collect.append(round(youxi.record['markovitz rate']-1,3))
    your_collect.append(round(youxi.record['cum']-1,3))
    reward_collect.append(round(ep_r, 2))
    cost_collect.append(np.mean(RL.cost_his))

record['start_date']=start
record['end_date']=end
record['your_collect']=your_collect
record['market_collect']=market_collect
record['reward_collect']=reward_collect
record['cost_collect']=cost_collect
record['mdp_vol']=game_mdp_vol
record['mkv_vol']=game_mkv_vol
record['mdp_sharp']=game_mdp_sharp
record['mkv_sharp']=game_mkv_sharp
record['mdp_draw']=game_mdp_draw
record['mkv_draw']=game_mkv_draw
record[stock_names[0]]=np.array(stock_collect)[:,0]
record[stock_names[1]]=np.array(stock_collect)[:,1]
record[stock_names[2]]=np.array(stock_collect)[:,2]
record[stock_names[3]]=np.array(stock_collect)[:,3]
record[stock_names[4]]=np.array(stock_collect)[:,4]
record[stock_names[5]]=np.array(stock_collect)[:,5]
record[stock_names[6]]=np.array(stock_collect)[:,6]
record[stock_names[7]]=np.array(stock_collect)[:,7]
record[stock_names[8]]=np.array(stock_collect)[:,8]
record[stock_names[9]]=np.array(stock_collect)[:,9]



#record['stocks name']=stock_names+['None']*(len(record['cost_collect'])-len(stock_names))

record.to_csv("DQN__4月7日早上随机股票 10个股2年66batch.csv")


RL.plot_cost()
#tf.reset_default_graph()

xi=pd.DataFrame(action_collect)
xi['count']=np.ones(xi.shape[0])
xi.columns=['jilu','count']
xi.groupby('jilu').count()

        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        