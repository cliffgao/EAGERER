#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:05:47 2020
Using  SPROf model
To Improve Protein Sequence Profile Prediction through Image Captioning on Pairwise Residue Distance Map
@author: Cliff
"""
import os,sys
from my_distmat_class import DISTMAT
from my_dssp_class import DSSP
from onehot import one_hot_seq
## reading the 
#from align import align_two_seq_needle,align_two_seq_refaa4dist
from align4mat import align_two_seq_needle,align_two_seq_refaa4dist2d
import numpy as np 
def get_model(amatfn,adsspfn,proid,checkFlag=False):
    """ using the matrix 2d features   sequence 1d features  """
    ## read in amat fn adsspfn and fwn
    distcls=DISTMAT(amatfn)
    aseq=distcls.get_seq()
    xamat=distcls.get_mat()
    #xamat=distcls.get_mat_hists()    
    dsspcls=DSSP(adsspfn)
    bseq=dsspcls.get_seq()
    xrsas=dsspcls.get_rsa()
   
    commSeq="" 
    #fw=open(fwn,'w')
    if len(aseq)!=len(bseq):
        #print("Length different,aligned sequence ")
        refaa1d,refaa3d=align_two_seq_needle(aseq,bseq)
        #print(refaa1d)
        #print(refaa3d)
        amat,rsas,commSeq=align_two_seq_refaa4dist2d(xamat,xrsas,refaa1d,refaa3d)
        print(len(amat),len(rsas),"aligned",proid)
        if checkFlag:
            print(refaa1d)
            print(refaa3d)
    elif aseq==bseq:
        amat=xamat
        rsas=xrsas
        commSeq=aseq
    else:
        print("!!!!   checked %s" %amatfn)
        print(aseq)
        print(bseq)
        amat=xamat
        rsas=xrsas
        commSeq=aseq
    ###  
    aHot=one_hot_seq(commSeq)  #0,1,0,...0,
    #NN=len(commSeq)
    ### write to three different files 
    #proid=os.path.basename(amatfn).split('.')[0]
    f1d="./outs/d1/%s.npy" %proid
    f2d="./outs/d2/%s.npy" %proid
    f3d="./outs/lbs/%s.npy" %proid
    np.save(f1d,aHot)
    np.save(f2d,amat)
    np.save(f3d,rsas)
    
    
    return(0)

def run4oneProd(pdbfile):
    proid=os.path.splitext(os.path.basename(pdbfile))[0]
    amatfn="./%s.mat" %proid
    adsspfn="./%s.dssp" %proid
    if not os.path.isfile(amatfn):
        ### compute the distance
        cmd1="./bin/calDist.py %s" %pdbfile
        os.system(cmd1)
    if not os.path.isfile(adsspfn):
        ### compute the dssp for labels
        #adsspfn="./%s.dssp" %proid
        cmd2="./bin/runDSSP.py %s F" %pdbfile
        os.system(cmd2)
        #fwn="./outs/%s.mdl" %item
    checkFlag=False #True
    get_model(amatfn,adsspfn,proid,checkFlag)
    print("%s" %proid) 
     
    return((amatfn,adsspfn))
def run4batch(listfn):
    fr=open(listfn,'r')
    lines=[line.strip() for line in fr.readlines()]
    fr.close()
    checkFlag=False #True
    for item in lines:
        ### compute the distance
        pdbfn="%s" %item 
        proid=os.path.splitext(os.path.basename(pdbfile))[0]
        amatfn,adsspfn=run4oneProd(pdbfn)
        #proid=item
        get_model(amatfn,adsspfn,proid,checkFlag)
        print(item)
        
    
if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage  pdbfile batchFlag(T/F)")
    else:
        #listfn=sys.argv[1]
        #main(listfn)
        pdbfile=sys.argv[1]
        if sys.argv[2]=='T':
            run4batch(pdbfile)
        else:
            run4oneProd(pdbfile)
    
    
    
    
    
