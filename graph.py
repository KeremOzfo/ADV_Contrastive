import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

def table(dic,save_csv = False):
    panda = pd.DataFrame(dic)
    print(panda)
    if save_csv:
        panda.to_csv('results.csv')

def load_results(file):
    dic = {'Alpha':[],'Temp':[],'Beta1':[],'Beta2':[],'Norm':[],
           'Clean last':[],'Adv last':[],'Clean best':[],'Adv best':[],}
    results_clean = []
    results_adv =[]
    keylist = dic.keys()
    for result in listdir(file):
        res_path = path.join(file,result)
        details = result.split('-')
        for detail in details:
            info = detail.split('_')
            if info[0] in keylist:
                dic[info[0]].append(info[1])
        res_adv, res_clean = compile_results(res_path)
        results_clean.append(res_clean)
        results_adv.append(res_adv)
        dic['Clean last'].append(np.round(res_clean[-1],3))
        dic['Adv last'].append(np.round(res_adv[-1], 3))
        dic['Clean best'].append(np.round(max(res_clean), 3))
        dic['Adv best'].append(np.round(max(res_adv), 3))
        for key in dic:
            if len(dic[key]) < len(dic['Clean last']):
                dic[key].append('-')
    table(dic,True)
    return dic, results_clean,results_adv

def compile_results(adress):
    results_adv = None
    results_clean = None
    for dir in listdir(adress):
        if dir[0] == 'C' or dir[0] == 'c':
            pat = path.join(adress, dir)
            results_clean = np.load(pat)
        elif dir[0] == 'A' or dir[0] == 'a':
            pat = path.join(adress, dir)
            results_adv = np.load(pat)
    return results_adv, results_clean


def legend_maker(dic):
    legends = []
    total = len(dic['Clean last'])
    for i in range(total):
        Temp = dic['Temp'][i]
        alfa = dic['Alpha'][i]
        beta1 = dic['Beta1'][i]
        beta2 = dic['Beta2'][i]
        norm = dic['Norm'][i]
        leg = 'T={} \u03B1={} \u03B2-1={}, \u03B2-2={}'.format(Temp,alfa,beta1,beta2,norm)
        legends.append(leg)
    return legends


def graph(adv,clean, dic,interval=1):
    marker = ['s', 'v', '+', 'o', '*']
    color = ['r', 'c', 'b', 'g','y','b','k']
    linestyle =['--', '-.', ':']
    linecycler = cycle(linestyle)
    colorcycler = cycle(color)
    markercycler = cycle(marker)
    legends = legend_maker(dic)
    for d_clean,d_adv,legend in zip(clean,adv,legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        c = next(colorcycler)
        for i in range(0,len(d_clean)):
            x_axis.append(i*interval)
        plt.plot(x_axis,d_adv, color=c,marker=m ,linestyle =l ,markersize=2, label=legend,linewidth=2)
        plt.plot(x_axis, d_clean,color=c,linewidth=2)
    #plt.axis([0, 30,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([290, 300, 90, 95])
    #plt.ylim([50,80])
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Test Classification Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()






loc = 'Results/'
types = ['benchmark','timeCorrelated','topk']
NNs = ['simplecifar']

locations = []
labels =[]
for tpye in types:
    for nn in NNs:
        locations.append(loc + tpye +'/'+nn)
        labels.append(tpye +'--'+ nn)

dic,results_clean,results_adv = load_results('Results/trades2')
graph(results_adv,results_clean,dic)