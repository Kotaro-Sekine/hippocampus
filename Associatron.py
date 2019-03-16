#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
class hebb:
    memory = []
    memory_size = 0
    
    def __init__(self, size):
        self.memory_size = size
        self.memory = np.reshape([0]*size**2, (size, size))
        
    def learn(self, activity):
        if self.memory_size != len(activity):
            return False

        for y in range(self.memory_size):
            for x in range(self.memory_size):
                if x == y:
                    self.memory[x,y] = 0
                elif activity[x] == 1 and activity[y] == 1:
                    self.memory[x,y] += 1
                elif activity[x] == -1 and activity[y] == -1:
                    pass  #このケースはpassでなく+1とすることも多々(今回は神経回路のモデル化を意識してpassにした)
                else:
                    self.memory[x,y] -= 1
        #print(self.memory)#途中経過表示
        return True

    def remember(self, activity):
        if self.memory_size != len(activity):
            return False

        activity = np.reshape(activity, (self.memory_size, 1))
        reminder = np.dot(self.memory,activity)
        #print(reminder)#途中経過表示
        return np.reshape(reminder / abs(reminder), (1, self.memory_size)).flatten()


# In[2]:


def learn(m):
    if h.learn(m):
        print("I learn",m)
    else:
        print("I can't learn",m)

def remember(question):
    answer = h.remember(question)
    if any(answer):
        print("Question:",question,"Answer:",answer)
    else:
        print("I can't remember",question)

h = hebb(3) #the number of neurons

#########################
print("Counts")
n=int(input()) #何回学習・思いだす作業を行うか The total number of tasks of "learn" and "remember"
for i in range(n):
    print("0=learn,1=remember")
    a=int(input()) #0=learn,1=remember
    if a==0:
        b=list(map(int,input().split()))
        learn(b)
    else:
        b=list(map(int,input().split()))
        remember(b)
#########################
#重み行列,入力配列を事前決定する場合は上の##内をコメントアウトし,こっちを使う
#If you don't want to use "input", you can use this.
#learn([1,1,-1])
#learn([1,1,-1])
#learn([1,1,-1])
#remember([1,1,-1])
#remember([1,1,0])
#learn([1,-1,1])
#learn([1,-1,1])
#remember([1,1,-1])
#remember([1,-1,1])
#########################

