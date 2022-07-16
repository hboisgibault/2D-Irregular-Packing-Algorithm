import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from tools.geofunc import GeoFunc
from tools.show import PltFunc
from tools.nfp import NFP
from tools.data import getData
from tools.packing import PackingUtil,NFPAssistant,PolyListProcessor,Poly
from TOPOS import TOPOS
from bottom_left_fill import BottomLeftFill
import json
from shapely.geometry import Polygon,mapping
from shapely import affinity
import csv
import time
import multiprocessing
import datetime
import random
import copy

def packingLength(poly_list,history_index_list,history_length_list,width,**kw):
    polys=PolyListProcessor.getPolysVertices(poly_list)
    index_list=PolyListProcessor.getPolyListIndex(poly_list)
    length=0
    check_index=PolyListProcessor.getIndex(index_list,history_index_list)
    if check_index>=0:
        length=history_length_list[check_index]
    else:
        try:
            if 'NFPAssistant' in kw:
                blf=BottomLeftFill(width,polys,NFPAssistant=kw['NFPAssistant'])
                # blf.showAll()
                length=blf.contain_length
            else:
                length=BottomLeftFill(width,polys).contain_length
        except:
            print('出现Self-intersection')
            length=99999
        history_index_list.append(index_list)
        history_length_list.append(length)
    return length

class GA(object):
    '''
    参考文献：A 2-exchange heuristic for nesting problems 2002
    '''
    def __init__(self,width,poly_list,nfp_asst=None,generations=10,pop_size=20):
        self.width=width
        self.minimal_rotation=360 # 最小的旋转角度
        self.poly_list=poly_list

        self.ga_multi=False # 开了多进程反而更慢
        if self.ga_multi:
            multiprocessing.set_start_method('spawn',True) 

        self.elite_size=10 # 每一代选择个数
        self.mutate_rate=0.1 # 变异概率
        self.generations=generations # 代数
        self.pop_size=pop_size # 每一代的个数

        self.history_index_list=[]
        self.history_length_list=[]
        
        if nfp_asst:
            self.NFPAssistant=nfp_asst
        else:
            self.NFPAssistant=NFPAssistant(PolyListProcessor.getPolysVertices(poly_list),get_all_nfp=True)

        self.geneticAlgorithm()

        self.plotRecord()

    # GA算法核心步骤
    def geneticAlgorithm(self):
        self.pop = [] # 种群记录
        self.length_record = [] # 记录高度
        self.lowest_length_record = [] # 记录全局高度
        self.global_best_sequence = [] # 全局最优序列
        self.global_lowest_length = 9999999999 # 全局最低高度
        
        # 初步的随机数组
        for i in range(0, self.pop_size):
            _list=copy.deepcopy(self.poly_list)
            random.shuffle(_list)
            self.pop.append(_list)

        # 持续获得下一代
        for i in range(0, self.generations):
            print("############################ Compute the ",i+1,"th generation #######################################")
            self.getLengthRanked() # 高度排列
            self.getNextGeneration() # 获得下一代

            # 高度记录与最低高度处理
            self.length_record.append(self.fitness_ranked[0][1])
            if self.fitness_ranked[0][1]<self.global_lowest_length:
                self.global_lowest_length=self.fitness_ranked[0][1]
                self.global_best_sequence=self.pop[self.fitness_ranked[0][0]]
            self.lowest_length_record.append(self.global_lowest_length)
            # print(self.global_lowest_length)

        # print("Final length: " + str(self.global_lowest_length))

        blf=BottomLeftFill(self.width,PolyListProcessor.getPolysVertices(self.global_best_sequence),NFPAssistant=self.NFPAssistant)
        blf.showAll()


    def plotRecord(self):
        plt.plot(self.lowest_length_record)
        plt.ylabel('Length')
        plt.xlabel('Generation')
        plt.show()

    # 对序列进行排序
    def getLengthRanked(self):
        length_results = []
        self.fitness_sum = 0
        
        if self.ga_multi==True:
            tasks=[[pop] for pop in self.pop]
            pool=multiprocessing.Pool()
            results=pool.starmap(self.getLength,tasks)
            for i in range(0,len(self.pop)):
                # print("length:",results[i])
                self.fitness_sum+=1000/results[i]
                length_results.append([i,results[i],1000/results[i],PolyListProcessor.getPolyListIndex(self.pop[i])])
        else:
            for i in range(0,len(self.pop)):
                length=self.getLength(self.pop[i])
                self.fitness_sum+=1000/length
                length_results.append([i,length,1000/length,PolyListProcessor.getPolyListIndex(self.pop[i])])

        self.fitness_ranked=sorted(length_results, key = operator.itemgetter(1)) # 排序，包含index

    def getLength(self,poly_list):
        length=packingLength(poly_list,self.history_index_list,self.history_length_list,self.width,NFPAssistant=self.NFPAssistant)
        return length
    
    # 根据排序选择序列
    def getNextGeneration(self):
        # mating_pool = self.rouletteWheelSelection() # 轮盘赌方法获得足够的后代并打乱，效果不佳
        mating_pool = self.eliteSelection() # 精英选择策略
        children=mating_pool
        for i in range(0, self.pop_size - self.elite_size):
            children.append(self.breed(children[random.randint(0,self.elite_size-1)], children[random.randint(0,self.elite_size-1)]))

        # 逐一进行突变处理获得新种群
        self.pop=[]
        for item in children:
            self.pop.append(self.mutate(item))
    
    # 精英选择策略
    def eliteSelection(self):
        mating_pool=[]
        for i in range(0, self.elite_size):
            mating_pool.append(self.pop[self.fitness_ranked[i][0]])
        return mating_pool
    
    # 参考：https://github.com/mangwang/PythonForFun/blob/master/rouletteWheelSelection.py
    def rouletteWheelSelection(self):
        mating_pool=[]
        for i in range(0, self.elite_size):
            rndPoint = random.uniform(0, self.fitness_sum)
            accumulator = 0.0
            for index, item in enumerate(self.fitness_ranked):
                accumulator += item[2]
                if accumulator >= rndPoint:
                    mating_pool.append(self.pop[item[0]])
        return mating_pool

    # 序列交配修改顺序（不修改方向）
    def breed(self,parent1, parent2):
        geneA,geneB = random.randint(0,len(parent1)-1), random.randint(0,len(parent1)-1)
        start_gene,end_gene = min(geneA, geneB),max(geneA, geneB)
        
        parent1_index = PolyListProcessor.getPolyListIndex(parent1)
        parent2_index = PolyListProcessor.getPolyListIndex(parent2)

        child1_index = parent1_index[start_gene:end_gene] # 截取一部分
        child2_index = [item for item in parent2_index if item not in child1_index] # 截取剩余部分

        return PolyListProcessor.getPolysByIndex(child1_index,self.poly_list) + PolyListProcessor.getPolysByIndex(child2_index,self.poly_list)

    # 个体突变，随机交换或方向改变
    def mutate(self,individual):
        for swapped in range(len(individual)):
            if(random.random() < self.mutate_rate):
                # 首先是交换位置
                if random.random()<=0.5:
                    individual=PolyListProcessor.randomSwap(individual,swapped)
                else:
                    individual=PolyListProcessor.randomRotate(individual,self.minimal_rotation,swapped)
        return individual

if __name__=='__main__':
    starttime = datetime.datetime.now()

    polys = getData(0)
    all_rotation = [0] # 禁止旋转
    poly_list = PolyListProcessor.getPolyObjectList(polys, all_rotation)

    nfp_assistant=NFPAssistant(polys, store_nfp=False, get_all_nfp=True, load_history=True)

    GA(760,poly_list,nfp_asst=nfp_assistant)

    endtime = datetime.datetime.now()
    print (endtime - starttime)
