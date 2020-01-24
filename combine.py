import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import environment as environ


class DiscreteAgent(ABC):
    @abstractmethod
    def __init__(self,env):pass
    
    @abstractmethod
    def update_equations(self):pass

    @abstractmethod
    def get_action(self, state):pass

class Confused_Agent(environ.Grid_env,DiscreteAgent):
    def __init__(self,env):
        super().__init__()
        self.policy= np.zeros([self.n_states, self.n_actions], dtype = float)

    
    def update_equations(self):pass
    
    def get_action(self,state):
        a = np.arange(4)
        np.random.shuffle(a)
        return [a[0]]
    
    def step(self,action,current_state,V):
        # Bellman update equation
        next_state=current_state
        if current_state in self.terminal_state:
            temp=0
        elif(current_state <=7   and action == 2):
            temp=self.probability[current_state,action]*( self.reward[current_state] + V[next_state])
        elif(current_state%8 ==0 and action == 0):
            temp=self.probability[current_state,action]*( self.reward[current_state] + V[next_state])
        elif( (current_state+1)%8 ==0 and action == 1):
            temp=self.probability[current_state,action]*( self.reward[current_state] + V[next_state])
        elif(current_state >= 56  and action == 3):
            temp=self.probability[current_state,action]*( self.reward[current_state] + V[next_state])
        elif(action == 0 ):
            next_state-=1
        elif(action == 1 ):
            next_state+=1
        elif(action == 2 ):
            next_state-=8
        elif(action == 3 ):
            next_state+=8
        temp= self.probability[current_state,action]*( self.reward[current_state] + V[next_state])
        return temp

    def update_policy(self,st_v):
        # Policy Improvement / Finding 
        p_old=copy.deepcopy(self.policy)
        for s in range(self.n_states):
            temp=-10000
            ind=-1
            if(s%8 != 0 and temp  <  st_v[s-1] ):
                ind=0
                temp=st_v[s-1]
            if( (s+1)%8 !=0 and temp  <  st_v[s+1] ):
                ind=1
                temp=st_v[s+1]
            if(s >7 and temp  <  st_v[s-8] ):
                ind=2
                temp=st_v[s-8]
            if(s < 56 and temp  <  st_v[s+8] ):
                ind=3
                temp=st_v[s+8]
            self.policy[s,:]=0
            self.policy[s,ind]=1
        return (all(p_old.flatten() == self.policy.flatten()))
    

    def print_policy(self,p):
        # Printing Policy
        final= np.argmax(self.policy, axis=1)
        final_policy=[]
        for i in range(self.n_states):
            if(i==63):
                final_policy.append(" ")
            elif(final[i]==0):
                final_policy.append("l")
            elif(final[i]==1):
                final_policy.append("r")
            elif(final[i]==2):
                final_policy.append("u")
            elif(final[i]==3):
                final_policy.append("d")
        if p:
            print(np.reshape(final_policy, (self.shape)))
        return final




class Agent1(environ.Grid_env,DiscreteAgent):
    def __init__(self,env,gamma):
        super().__init__()
        # agent's policy
        self.policy= np.zeros([self.n_states, self.n_actions], dtype = float)
        self.policy[:,:]=0.25
        # Discount factor
        self.gamma=gamma
        
    def update_equations(self):pass

    def get_action(self, state):
        return self.action

    def step(self,action,current_state,V):
        # Bellman update equation
        next_state=current_state
        if current_state in self.terminal_state:
            temp=0
        elif(current_state <=7   and action == 2):
            temp=self.probability[current_state,action]*( self.reward[current_state] + self.gamma*V[next_state])
        elif(current_state%8 ==0 and action == 0):
            temp=self.probability[current_state,action]*( self.reward[current_state] + self.gamma*V[next_state])
        elif( (current_state+1)%8 ==0 and action == 1):
            temp=self.probability[current_state,action]*( self.reward[current_state] + self.gamma*V[next_state])
        elif(current_state >= 56  and action == 3):
            temp=self.probability[current_state,action]*( self.reward[current_state] + self.gamma*V[next_state])
        elif(action == 0 ):
            next_state-=1
        elif(action == 1 ):
            next_state+=1
        elif(action == 2 ):
            next_state-=8
        elif(action == 3 ):
            next_state+=8
        temp= self.probability[current_state,action]*( self.reward[current_state] + self.gamma*V[next_state])
        return temp

    def update_policy(self,st_v):
        # Policy Improvement / Finding 
        p_old=copy.deepcopy(self.policy)
        for s in range(self.n_states):
            temp=-10000
            ind=-1
            if(s%8 != 0 and temp  <  st_v[s-1] ):
                ind=0
                temp=st_v[s-1]
            if( (s+1)%8 !=0 and temp  <  st_v[s+1] ):
                ind=1
                temp=st_v[s+1]
            if(s >7 and temp  <  st_v[s-8] ):
                ind=2
                temp=st_v[s-8]
            if(s < 56 and temp  <  st_v[s+8] ):
                ind=3
                temp=st_v[s+8]
            self.policy[s,:]=0
            self.policy[s,ind]=1
        return (all(p_old.flatten() == self.policy.flatten()))
    

    def print_policy(self,p):
        # Printing Policy
        final= np.argmax(self.policy, axis=1)
        final_policy=[]
        for i in range(self.n_states):
            if(i==63):
                final_policy.append(" ")
            elif(final[i]==0):
                final_policy.append("l")
            elif(final[i]==1):
                final_policy.append("r")
            elif(final[i]==2):
                final_policy.append("u")
            elif(final[i]==3):
                final_policy.append("d")
        if p:
            print(np.reshape(final_policy, (self.shape)))
            print("gamma=",self.gamma)
        return final


class Agent2(environ.Gambler_env,DiscreteAgent):
    def __init__(self,env,gamma,p_head):
        super().__init__()
        self.policy= np.zeros([self.n_states], dtype = float)
        self.policy[:]=0
        # discount
        self.gamma=gamma
        # probability of winning 
        self.p_head=p_head
        
    def update_equations(self):pass

    def get_action(self, state):
        # actions as given in question
        return np.arange(min(state, self.goal - state) + 1)

    def step(self,action,current_state,V):
        # Bellman update eqaution 
        # reward will be zero for all transition except the one which agent reaches to goal 
        if current_state is self.goal:
            # agent will get no reward after reaching terminal state
            temp=0
        else:
            temp=self.p_head * ( self.reward[current_state+action] + V[current_state + action])  + ( 1-self.p_head )* ( self.reward[current_state-action] +  V[current_state - action])
        return temp
    
    def update_policy(self,V):
        for states in range(1,self.goal):
            actions=self.get_action(states)
            Q=np.zeros(len(actions),dtype = float)
            for i in range(len(actions)):
                Q[i]=self.step(actions[i],states,V)
            self.policy[states] = actions[np.argmax( np.round( Q[1:] , 5)) + 1]
        # print(self.policy)
        print("Probibility of head=",self.p_head)
        return self.policy
    
    def print_policy(self,p):
        return self.policy

class PolicyIteration(DiscreteAgent):
    def __init__(self,env,agent,epsilon,printd=True):
        super().__init__(env)
        self.agent=agent
        # cutoff or stopping condition
        self.epsilon = epsilon
        # State value initialize with zero 
        self.state_value=np.zeros(self.agent.n_states)
        self.iter=0
        self.value_history=[]
        self.print=printd

    def get_action(self, state):
        return self.agent.get_action(state)

    def update_equations(self):
        Stable_Policy=False
        while(Stable_Policy!=True):
            self.state_value=self.evaluate_policy()
            Stable_Policy=self.agent.update_policy(self.state_value)
            self.iter+=1
        # print(self.state_value.reshape(self.agent.shape))
        final_p = self.agent.print_policy(self.print)
        self.value_history.append(self.state_value)
        if self.print:
            print("epsilon=",self.epsilon)
            print("Stopping condition meets")
            print("No of iteration=",self.iter)
        return self.value_history,self.state_value,final_p

    def evaluate_policy(self):
        # Policy evaluation
        self.value_history.append(copy.deepcopy(self.state_value))
        while(1):
            delta = 0
            for s in range(self.agent.n_states):
                A = self.agent.get_action(s)
                Q = np.zeros(self.agent.n_actions)
                v= self.state_value[s]
                temp_sum=0
                for i in range(4):
                    temp_sum += self.agent.policy[s,i] * self.agent.step(A[i],s,self.state_value)        
                self.state_value[s]=temp_sum
                delta = max(delta, np.abs(self.state_value[s]- v))
            if delta < self.epsilon:  
                return self.state_value


class ValueIteration(DiscreteAgent):
    def __init__(self,env,agent,epsilon,max_iteration=1000,printd=True):
        super().__init__(env)
        self.agent=agent
        # State value initialize with zero 
        self.state_value=np.zeros(self.agent.n_states)
        # cutoff or stopping condition
        self.epsilon =epsilon
        self.iter=0
        # storing history for each sweep
        self.value_history=[]
        self.max_iteration=max_iteration
        self.print=printd

    def get_action(self, state):
        pass

    def update_equations(self):
        while True:
            V = copy.deepcopy(self.state_value)
            self.value_history.append(V)
            delta = 0

            for s in range(self.agent.n_states):
                A = self.agent.get_action(s)
                Q=np.zeros(len(A),dtype=float)
                for i in range(len(A)):
                    Q[i] = self.agent.step(A[i],s,self.state_value)
                self.state_value[s] = max(Q)
                delta = max(delta, np.abs(self.state_value[s]- V[s]))
            self.iter+=1
            # print(self.iter)
            if delta < self.epsilon or self.iter >= self.max_iteration:
                self.value_history.append(self.state_value)   
                # print(self.state_value.reshape(self.agent.shape))
                # policy finding
                self.agent.update_policy(self.state_value)
                final = self.agent.print_policy(self.print)
                if self.print:
                    print("epsilon=",self.epsilon)
                    print("Stopping condition meets")
                    print("No of iteration=",self.iter)
                return self.value_history,self.state_value,final

if __name__ == "__main__":
    env_gam=environ.Grid_env()
    agent=Confused_Agent(env_gam)
    VI= ValueIteration(env_gam,agent,1e-7,100) 
    history,val_fun,f_policy = VI.update_equations()
    print(val_fun)
    print(f_policy)

