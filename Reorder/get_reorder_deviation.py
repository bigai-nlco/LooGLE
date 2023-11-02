import json
import numpy as np
import re
import itertools

def location_square_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if lst_2 !=[] and n == len(lst_2):
            for i in range(n): 
                try:
                    lst[lst_1.index(lst_2[i])] = i
                except:
                    break

    try:
        s = 0
        for i in range(n):
            s += (lst[i]-i) ** 2
        s /= n
        return s
        
    except:
        return "None"

def location_mean_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if lst_2 !=[] and n == len(lst_2): 
            for i in range(n):
                try:
                    lst[lst_1.index(lst_2[i])] = i
                except:
                    break
    try:
        s = 0
        for i in range(n):
            s += abs(lst[i]-i)
        s /= n
        return s
    except:
        return "None"


def swap_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None: 
        if lst_2 !=[] and n == len(lst_2):
            for i in range(n):
                try:
                    lst[lst_1.index(lst_2[i])] = i
                except:
                    break
    try:    
        count = 0	
        for i in range(n):
            if lst[i] == -1:
                continue
            p = i
            while lst[p] != -1:
                q = lst[p]
                lst[p] = -1
                p = q
            count += 1
        return n - count 
    except:
        return "None"


def swap_distance_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if lst_2 !=[] and n == len(lst_2):
            for i in range(n):
                try:
                    lst[lst_1.index(lst_2[i])] = i
                except:
                    break
    try:
        swap_lst = []
        weight = 0
        while location_mean_deviation(lst) != 0:
            r_best = 0	
            i_best = 0
            j_best = 0
            for i in range(n):
                for j in range(i+1, n):	
                    r = ((abs(lst[i]-i)+abs(lst[j]-j)) - (abs(lst[j]-i)+abs(lst[i]-j)))/(j-i)
                    if r > r_best:
                        r_best = r
                        i_best = i
                        j_best = j
            lst[i_best], lst[j_best] = lst[j_best], lst[i_best]
            weight += (j_best-i_best)
            swap_lst.append((i_best, j_best))
        return weight
    except:
        return "None"

