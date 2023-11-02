import json
import numpy as np
import re
import itertools
from get_reorder_deviation import (
    location_square_deviation,
    location_mean_deviation,
    swap_deviation,
    swap_distance_deviation
)

def get_max_location_square_deviation(n):
    inp = list(range(1,n+1))
    permutations_lst = list(itertools.permutations(inp))
    
    tmp = -1
    for i in permutations_lst:
        if inp != list(i):
            dis = location_square_deviation(inp,i)
            if dis >= tmp:
                tmp = dis
    return tmp


def get_max_location_mean_deviation(n):
    inp = list(range(1,n+1))
    permutations_lst = list(itertools.permutations(inp))
    
    tmp = -1
    for i in permutations_lst:
        if inp != list(i):
            dis = location_mean_deviation(inp,i)
            if dis >= tmp:
                tmp = dis
    return tmp


def get_max_swap_deviation(n):
    inp = list(range(1,n+1))
    permutations_lst = list(itertools.permutations(inp))
    
    tmp = -1
    for i in permutations_lst:
        if inp != list(i):
            dis = swap_deviation(inp,i)
            if dis >= tmp:
                tmp = dis
    return tmp


def get_max_swap_distance_deviation(n):
    inp = list(range(1,n+1))
    permutations_lst = list(itertools.permutations(inp))
    
    tmp = -1
    for i in permutations_lst:
        if inp != list(i):
            dis = swap_distance_deviation(inp,i)
            if dis >= tmp:
                tmp = dis
    return tmp