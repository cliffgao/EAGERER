#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:05:47 2020
Using  SPROf model
To Improve Protein Sequence Profile Prediction through Image Captioning on Pairwise Residue Distance Map
@author: Cliff
"""
import os,sys
from calDist import calc_dist_main
from onehot import one_hot_seq
import numpy as np 

def get_model(pdbfn):
    """ using the matrix 2d features   sequence 1d features  """
    #proid=os.path.splitext(os.path.basename(pdbfn))[0]
    proid=os.path.basename(pdbfn)
    ## read in amat fn adsspfn and fwn
    amat,aseq,resseqs=calc_dist_main(pdbfn)
    #print(np.shape(amat),aseq)
    #one-hot encode 
    aHot=one_hot_seq(aseq)  #0,1,0,...0,
    #print(aHot)
    #NN=len(commSeq)
    rsas=resseqs #[-1]*len(amat)
    #rsas=[]
    N1=len(aseq)
    N2=len(resseqs)
    if N1!=N2:
        print("different length seq: %d  resseq: %d" %(N1,N2))
    #for k in range(N1):
    #    rsas.append("%d,%s" %(resseqs[k],aseq[k]))
    ### write to three different files 
    #proid=os.path.basename(amatfn).split('.')[0]
    f1d="./features/d1/%s.npy" %proid
    f2d="./features/d2/%s.npy" %proid
    f3d="./features/lbs/%s.npy" %proid
    np.save(f1d,aHot)
    np.save(f2d,amat)
    np.save(f3d,rsas)
    
    #amino acid; residue sequence number 
    return(aseq,resseqs)
def main(listfn):
    fr=open(listfn,'r')
    lines=[line.strip() for line in fr.readlines()]
    fr.close()
    for item in lines:
        ### for each pdb file
        ### get the features: distance +on_hot 
        pdbfn=item
        proid=os.path.splitext(os.path.basename(pdbfn))[0]
        if os.path.isfile(pdbfn):
            aseq,aresnum=get_model(pdbfn)
        else:
            print("not find %s" %item)
        ### run for prediction
        cmd="./bin/test.py --test_list %s" %listfn 
        os.system(cmd)
        ### parepare the final outputs
        Eagererfn="EAGERER.ckpt_lb_pd.csv"
        if os.path.isfile(Eagererfn):
            fr=open(Eagererfn,'r')
            lines=[line.strip() for line in fr.readlines()]
            fr.close()
            fwn="%s.out" %proid
            fw=open(fwn,'w')
            fw.write("#Position,AminoAcid,RelativeSolventArea\n")
            for k in range(len(lines)):
                if len(lines[k])>0:
                    aline=lines[k].split(",")
                    posit=aresnum[k]  #int(aline[0])
                    rsaval=float(aline[1])
                    fw.write("%5d,%2s,%9.6f\n" %(posit,aseq[k],rsaval))
            fw.close()

        print("%s is done,results are in same folder of pdbfile " %pdbfn)
if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage  listfn ")
    else:
        listfn=sys.argv[1]
        main(listfn)
    
    
    
    
    
