#!/usr/bin/env python
import torch
from dataloader import *
from model import *
import argparse
import os
import numpy as np  #from scipy.stats import pearsonr 
import pandas as pd 
#torch.cuda.set_device(0)
#BATCHSIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--sequential_features', type=str, default='./features/d1', help='The full path of sequential features of test data.')
parser.add_argument('--pairwise_features', type=str, default='./features/d2', help='The full path of pairwise features of test data.')
parser.add_argument('--target_output', type=str, default='./features/lbs', help='The full path of the target outputs.')
parser.add_argument('--test_list', type=str, default='test_list', help='The full path of the test list.')
parser.add_argument('--models_path', type=str, default='./models', help='path list of models.')
config = parser.parse_args()

if __name__ == '__main__':
    myname = config.models_path
    nameList=os.listdir(myname)#models' names
    #fwn0="sprofpred"
    #print("#modelname overall_pcc average_pcc overall_mae average_mae")
    for modelname in nameList:
        net = SPROF(ResidualBlock).eval()
        net = net.to(device)
        net.load_state_dict(torch.load(f'{config.models_path}/{modelname}'))
        data_loader = get_loader(config.sequential_features, config.pairwise_features, config.target_output,
                                 config.test_list, batch_size=1, shuffle=False, num_workers=0)
        #lll = []#result record
        #f = open('sprofpred4each.txt', 'w')#result record
        #f2 = open('sprofpred_lb_pd.txt', 'w')#result record
        #fwn0="%s_4each.csv" %(modelname)
        #fwn="%s_lb_pd.csv"  %(modelname)
        #lbs_pds=[]  # store lbs and predicts 
        with torch.no_grad():
            for i, (test1d, test2d, targetdata, name) in enumerate(data_loader):
                test1d = test1d.to(device)
                test2d = test2d.to(device)
                targetdata=targetdata.float()  #cliff 2020-05-31 
                #print(targetdata)
                targetdata = targetdata.to(device)
                output = net(test1d, test2d)  # model output
                output = output.to(device)
                #softm = torch.nn.Softmax(dim=0)
                #score = softm(output[0])
                #scoreMax = torch.max(score, 0)[1]
                # print(name)
                # print(targetdata[0])
                # print(scoreMax)
                #ac = float(torch.sum(scoreMax == targetdata[0])) / float(len(targetdata[0]))#accuracy calc
                #print(targetdata)
                #print(output)
                alb=targetdata[0].cpu().numpy()
                apd=output[0].cpu().numpy()[0]
                #acor=np.corrcoef(alb,apd)[0,1] #cor p-value 
                #amae=np.mean(np.abs(alb-apd)) # 2020-08-31
                #print(alb.shape,apd.shape)
                #print(alb)
                #print(apd)
                #print(name)
                alb_pd=np.column_stack((alb,apd))
                #df=pd.DataFrame(alb_pd,columns=["No","RSA"],dtype={"No":np.int64,"RSA":np.float64})
                df=pd.DataFrame(alb_pd,columns=["No","RSA"])
                df["No"]=df["No"].astype(np.int64) #,dtype={"No":np.int64,"RSA":np.float64})
                #print(name)
                outfn="%s.out" %os.path.splitext(name[0])[0]
                df.to_csv(outfn,index=False,header=True,float_format="%10.6f")
                #if i==0:
                #    lbs_pds=alb_pd
                #else:
                #    lbs_pds=np.concatenate((lbs_pds,alb_pd))
                #print(ac)
                #lll.append(acor)
                #f.write("%s %s %.3f\n" %(modelname,name[i],acor))
                #lll.append([modelname,name[0],acor,amae])
                #name[0] + ' ' + '%.3f' % acor + '\n')#result record
                #print("%s %s is done" %(modelname,i))
            ### --for all models in test
            #df=pd.DataFrame(lbs_pds)
            #df.to_csv(fwn,index=False,header=False)
            ### --
            #print(lbs_pds.shape)
            #overall_pcc=np.corrcoef(lbs_pds[:,0],lbs_pds[:,1])[0,1]
            #overall_mae=np.mean(np.abs(lbs_pds[:,0]-lbs_pds[:,1]))  #2020-08-31
            #print("*** overall_pcc: %.3f "%overall_pcc)
            ### 
            #df2=pd.DataFrame(lll)
            #df2.to_csv(fwn0,index=False,header=False)
            #avgpcc=df2.iloc[:,2].mean()
            #avgmae=df2.iloc[:,3].mean()  #2020-08-31
            #print("*** avg_pcc    : %.3f" %avgpcc)
            #print("%20s %8.3f  %8.3f %8.3f %8.3f" %(modelname,overall_pcc,avgpcc,overall_mae,avgmae))
            #lll = np.array(lll)
            #np.save('sprof.npy', lll)#result record
    #f.close()
