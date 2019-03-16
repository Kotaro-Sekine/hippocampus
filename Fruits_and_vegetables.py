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
        #print(reminder) #途中経過
######################閾値設定(option)#########################################
        #for i in range(self.memory_size):
         #   if reminder[i]>3:
          #      pass
           # else: #閾値を下回ったら負に(活性なしと判断)
            #    reminder[i]=-abs(reminder[i]
######################################################################
        reminder2 = np.reshape(reminder / abs(reminder), (1, self.memory_size)).flatten()
        reminder2[np.isnan(reminder2)] = -1
        reminder3 = [int(s) for s in reminder2]
        return reminder3


# In[2]:


def learn(m):
    if h.learn(m):
        print("I learn",m)
    else:
        print("I can't learn",m)

def remember(question):
    answer = h.remember(question)
    if any(answer):
        if answer==[1,-1,1,-1]:
            print("apple")
        elif answer==[1,-1,-1,1]:
            print("tomato")
        elif answer==[-1,1,1,-1]:
            print("banana")
        else:
            print("I can not answer")
    else:
        print("I can not answer")
##activity[0]=red,[1]=yellow,[2]=fruit,[3]=vegetable##
def keyword(k):
    if k=="apple":
        return [1,-1,1,-1]
    elif k=="tomato":
        return [1,-1,-1,1]
    elif k=="banana":
        return [-1,1,1,-1]
    elif k=="red":#個々の数値は要検討
        return [1,-1,0,0]
    elif k=="yellow":
        return [-1,1,0,0]
    elif k=="fruit":
        return [0,0,1,-1]
    elif k=="vegetable":
        return [0,0,-1,1]
    else:
        print("I can not remember this")

h = hebb(4) #the number of neurons

#########################
print("Counts")
n=int(input()) #何回学習・思いだす作業を行うか The total number of tasks of "learn" and "remember"
for i in range(n):
    print("0=learn,1=remember")
    a=int(input()) #0=learn,1=remember
    if a==0:
        b=input() #input fruits/vegetables
        if b=="apple" or "tomato" or "banana":
            learn(keyword(b))
        else:
            print("I can not remeber this")
    else:
        b=input()  #input keyword
        remember(keyword(b))
#########################
#重み行列,入力配列を事前決定する場合は上の##内をコメントアウトし,こっちを使う
#If you don't want to use "input", you can use this.
#learn(keyword("apple"))
#learn(keyword("apple"))
#learn(keyword("tomato"))
#learn(keyword("apple"))
#remember(keyword("red"))
#remember(keyword("yellow"))
#########################

