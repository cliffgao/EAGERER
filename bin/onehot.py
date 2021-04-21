#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os,sys
import numpy as np
def one_hot_seq(seq):
    ### A=10000000000000000000
    aseq=seq.upper()
    AAdb={'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,
          'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'U':20,'X':20}
    #AAseq="ACDEFGHIKLMNPQRSTVWY"
    coded=[] #""
    LL=len(aseq)
    aa=np.zeros((LL,21),dtype=int)
    for k in range(len(aseq)):
        item=aseq[k]
        try:
            j=AAdb[item]
        except KeyError:
            j=20
        #if j<=20:
        aa[k][j]=1
    #print all item in aa by row
    for k in range(len(aa)):
        coded.append(aa[k].tolist())
    return(coded)
    
#seq="ACD"
#print(one_hot_seq(seq))
def one_hot_num(seq):
    ## one hot for number  0=0000000000 1=0100000000
    aseq=seq.upper()
    coded=""
    LL=len(aseq)
    aa=np.zeros((LL,10),dtype=int)
    for k in range(len(aseq)):
        j=int(aseq[k])
        aa[k][j]=1
    #print all item in aa by row
    for x in np.nditer(aa):
        coded+="%d," %x
    return(coded)
    
#seq="115"
#print(one_hot_num(seq))
if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage seq")
    else:
        seq=sys.argv[1]
        aa=one_hot_seq(seq)
        print(aa)
