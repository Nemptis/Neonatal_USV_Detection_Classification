#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:28:01 2023

@author: lgranzow
"""

import numpy as np
#import copy

'''
def delete_short_runs(mask,num):
  #take a binary array list (mask), and set a run of 1s to 0s if it is shorter than num
  counter = 0
  for i in range(len(mask)):  
    if mask[i] == 1:
      counter += 1
    else:
      if (counter > 0) and (counter < num):
        mask[i-counter:i] = np.zeros(counter)
      counter = 0
  return mask
'''

def delete_short_runs(a, n):
    #does the same as above, but more efficient; from https://stackoverflow.com/questions/24407406/removing-runs-from-a-2d-numpy-array
    #that is: take a binary array list a (i.e. long lis of 0s and 1s) and set a run of 1s to 0s if it is shorter than num
    #timeit: 3.9ms +- 0.7ms (new) vs. 101ms +- 34.7ms (old)
    match = a[:-n].copy()
    for i in range(1, n): # find 1s that have n-1 1s following
        match &= a[i:i-n]
    matchall = match.copy()
    matchall.resize(match.size + n)
    for i in range(1, n): # make the following n-1 1s as well
        matchall[i:i-n] |= match
    #b = a.copy()
    #b ^= matchall # xor into the original data; replace by + to make 2s
    return matchall

def smooth_mask_callsfirst(mask,Nd,Ng):
  #take a binary array list mask, consisting of a long list of 0s ans 1s.
  #first delete short runs of 1s
  mask = delete_short_runs(mask,Nd)
  #then delete short runs of 0s
  mask = 1-delete_short_runs(1-mask,Ng)
  return mask

def smooth_mask_gapsfirst(mask,Nd,Ng):
  #take a binary array list mask, consisting of a long list of 0s ans 1s.
  #first delete short runs of 0s
  mask = 1-delete_short_runs(1-mask,Ng)
  #then delete short runs of 1s
  mask = delete_short_runs(mask,Nd)
  return mask

def detect_entropyratio(S,f,Nd,Ng,entropythreshold,ratiothreshold):
    #split high and low part
    idx_f_high = ((f>40000) & (f<110000))
    idx_f_low = (f<40000)
    S_high = S[idx_f_high,:]
    S_low = S[idx_f_low,:]
    
    #ratio of energy in 40-110kHz part and energy in 0-40kHz part
    ratio = np.divide(np.mean(S_high,axis=0),np.mean(S_low,axis=0))
    
    #entropy of 40-110kHz part
    S_high_normalized = S_high/np.sum(S_high,axis=0)
    entropy_high = -np.sum(S_high_normalized*np.log(S_high_normalized),axis=0)
    
    #do detectio: entropy and ratio threshold must be met, then deleting gaps and too short calls
    detection = (entropy_high < entropythreshold) & (ratio > ratiothreshold).astype(int)
    detection = smooth_mask_gapsfirst(detection,Nd=Nd,Ng=Ng)
    
    start_idcs = np.where(np.diff(detection)==1)[0]
    print(f'detected {len(start_idcs)} calls')
    
    #returning detection, and ratio and entropy as well if I want plot them
    return detection, ratio, entropy_high