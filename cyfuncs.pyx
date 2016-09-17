# -*- coding: utf-8 -*-
import numpy as np

#processing array index to 0~1
def normalizationArray(array,_min,_max):
    cdef float amin, amax, element
    cdef int i
    
    amin = float(_min)
    amax = float(_max)
    
    if amin != amax:
        for i,element in enumerate(array):
            if element > amax:
                array[i] = 1
            elif element < amin:
                array[i] = 0
            elif element == np.nan:
                array[i] = np.nan
            else:
                ret = (float(element) - amin) / (amax - amin)
                array[i] = ret
    #期間の最大最小が等しい場合はすべての要素を0.5とする
    elif amin == amax:
        for i,element in enumerate(array):
            array[i] = float(0.5)