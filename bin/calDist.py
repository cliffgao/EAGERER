#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:38:16 2020

@author: Cliff
"""

### Calculate the distance of pdb using biopython
import os,sys
import warnings
warnings.filterwarnings("ignore")
from Bio.PDB.PDBParser import PDBParser
from Bio import PDB
import numpy as np


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    try:
        x=residue_one["CA"].coord 
        y=residue_two["CA"].coord
    except KeyError:
        #print("x",residue_one)
        #print("y",residue_two)
        x=None #'X' #-1
        y=None #'Y' #-1
    #diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    if (x is not None) and (y is not None):
        diff_vector  = x-y
        adist=np.sqrt(np.sum(diff_vector * diff_vector))
    else:
        adist=-1.0
    return (adist) 
#np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            #if 
            adist= calc_residue_dist(residue_one, residue_two)
            #if adist>=0:
            answer[row, col] = adist #calc_residue_dist(residue_one, residue_two)
    return answer
def get_seq_from_pdb(model):
    # get the proseqs from pdb
    #using C-N
    ppb=PDB.PPBuilder()
    #using CA-CA
    #ppb=PDB.CaPPBuilder()
    seqs=[]
    for pp in ppb.build_peptides(model):
        aseq=pp.get_sequence()
        #print(aseq,"*")
        seqs.append(str(aseq))
    return(seqs)
def get_first_chainID(xpdb_frn):
    #2021-04-01
    chainID=" "
    fr=open(xpdb_frn,'r')
    for line in fr.readlines():
        if len(line)>=22 and line[0:4]=='ATOM':
            chainID=line[21]
            break
    fr.close()

    return(chainID)

def calc_dist_main(xpdb_frn,fwn=None):
    """ read in pdb file"""
    #pdb_code=xpdb_frn[0:4] #"1a1x"
    #chain_id=xpdb_frn[4] #""
    #pdb_filename="./pdb/%s%s" %(pdb_code,chain_id)
    pdb_filename=xpdb_frn
    pdb_code=os.path.splitext(os.path.basename(pdb_filename))[0]
    #chain_id=" "
    chain_id=get_first_chainID(xpdb_frn)
    structure = PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]
    #print(model[chain_id])
    chainOne=model[chain_id]
    chainTwo=model[chain_id]
    #print(chainOne)
    ###  get resseq 
    resseqs=[]
    for xx in chainOne:
        resseqs.append(int(xx.__repr__().split("=")[2].split()[0])) #['resseq'])
    ###  2021-04-20 up
    dist_mat=calc_dist_matrix(chainOne, chainTwo)
    #print("min distance," ,np.min(dist_mat))
    #print("max distance,",np.max(dist_mat))
    #print(np.shape(dist_mat)) 
    seqs=get_seq_from_pdb(model) #may be multi sequences using the first one
    #print(len(seqs))
    #print(len(seqs[0]))
    #print(seqs)
    # some pdb may be seperated by disorder, so connect amino acid 
    aseq="".join(seqs)
    #print(aseq)
    if fwn is not None:
        fw=open(fwn,'w')
        fw.write("%s\n%s\n" %(xpdb_frn,aseq))
        row,col=np.shape(dist_mat)
        for i in range(row):
            for j in range(col):
                fw.write("%f " %dist_mat[i][j])
            fw.write("\n")
        fw.close()
    #distance;seq;pdfile_name
    return(dist_mat,aseq,resseqs)

def main(frn):
    #frn="tr.5id"
    fr=open(frn,'r')
    lines=[line.strip() for line in fr.readlines()]
    fr.close()
    for item in lines:
        if len(item)>0:
            xpdb_frn=item #"1a1xA"
            fwn="./dists/%s.mat" %xpdb_frn
            calc_dist_main(xpdb_frn,fwn)
            print(item)
    return(0)
def run4one(pdbfile,fwn):
    # pdb distance matrix 
    calc_dist_main(pdbfile,fwn)
#frn="ts.5id" #"tr.5id"
#main(frn)
if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage pdbfile")
    else:
        pdbfile=sys.argv[1]
        afile=os.path.splitext(os.path.basename(pdbfile))[0]
        
        fwn="%s.mat" %afile
        #run4one(pdbfile,fwn)
        amat,aseq,resseqs=calc_dist_main(pdbfile)
        print(amat)
        print(aseq)
        print(resseqs)

