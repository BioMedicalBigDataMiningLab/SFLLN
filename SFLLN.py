from sklearn.cross_validation import KFold
import numpy as np
import math
from copy import deepcopy
from math import*
from pylab import *
from compiler.ast import flatten
import xlrd
import ast
import copy


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve

 

import random
from numpy import linalg as LA



class data:
    drug_list = []
    protein_list_target = []
    protein_list_enzy = []
    protein_list_path = []
    protein_list_structure = []
    all_message_dict = {}
    drug_drug_interaction_matrix = ''
    enzy_matrix = ''
    structure_matrix = ''
    target_matrix = ''
    path_matrix = ''
    wrong_list = []

    def get_drug_drug_interaction_data(self):
        file_object = open('interaction matrix.txt')
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        x = ast.literal_eval(file_context)
        for i in sorted(self.wrong_list,reverse=True):
            del x[i]
            for line in x:
                del line[i]
        self.drug_drug_interaction_matrix = np.matrix(x)


    def get_data(self):
        book = xlrd.open_workbook('feature matrix.xls')
        sh = book.sheet_by_index(0)
        self.get_drug_drug_interaction_data()
        interaction_matrix = self.drug_drug_interaction_matrix
        sum_matrix = np.sum(interaction_matrix,axis=0)
        for r in range(0,interaction_matrix.shape[0]):
            if sum_matrix[0,r]==0:
                self.wrong_list.append(r)

        not_input = True
        index = -1
        for rx in range(sh.nrows):
            if not_input:
                not_input = False
                continue
            index = index + 1
            if index in self.wrong_list:
                continue
            content_list = sh.row(rx)
            if self.drug_list.count(content_list[0].value)==0:
                self.drug_list.append(content_list[0].value)
            target_str = content_list[3].value
            target_list = target_str.split('|')
            target_list.pop()

            for u in target_list:
                if u!='':
                    if self.protein_list_target.count(u)==0:
                        self.protein_list_target.append(u)
            enzy_str = content_list[4].value
            enzy_list = enzy_str.split('|')
            enzy_list.pop()
            for u in enzy_list:
                if u != '':
                    if self.protein_list_enzy.count(u) == 0:
                        self.protein_list_enzy.append(u)
            path_str = content_list[5].value
            path_list = path_str.split('|')
            path_list.pop()
            for u in path_list:
                if u != '':
                    if self.protein_list_path.count(u) == 0:
                        self.protein_list_path.append(u)

            structure_str = content_list[2].value
            structure_list = structure_str.lstrip('[').rstrip(']').split(',')
            structure_list_int = [int(x) for x in structure_list]
            for u in structure_list_int:
                if self.protein_list_structure.count(u)==0:
                    self.protein_list_structure.append(u)
            self.all_message_dict[content_list[0].value] = {"target_list":target_list,"enzy_list":enzy_list,"path_list":path_list,"structure_list":structure_list_int}

        for drug in self.drug_list:
            dict = self.all_message_dict[drug]
            m=[]
            for i in self.protein_list_target:
                if i not in dict["target_list"]:
                    m.append(0)
                else:m.append(1)
            dict["target_list_value"] = m
            n = []
            for i in self.protein_list_enzy:
                if i not in dict["enzy_list"]:
                    n.append(0)
                else:n.append(1)
            dict["enzy_list_value"] = n
            k = []
            for i in self.protein_list_path:
                if i not in dict["path_list"]:
                    k.append(0)
                else:k.append(1)
            dict["path_list_value"]=k
            s = []
            for i in self.protein_list_structure:
                if i not in dict["structure_list"]:
                    s.append(0)
                else:
                    s.append(1)
            dict["structure_list_value"] = s
        self.get_drug_drug_interaction_data()

    def get_all_message_dict(self):
        return self.all_message_dict

    def get_drug_list(self):
        return self.drug_list

    def data_prepare(self):
        self.get_drug_drug_interaction_data()
        self.get_data()

    def get_drug_drug_interactive_matrix(self):
        return self.drug_drug_interaction_matrix

    def divide_drug_interaction_matrix_into_train_test(self,test_divide,seed):
        m = self.drug_drug_interaction_matrix
        sum_link = np.where(m==1)
        a_list = []
        b_list = []
        for long in range(0,sum_link[0].__len__()):
            if sum_link[0][long]>sum_link[1][long]:
                a_list.append(sum_link[0][long])
                b_list.append(sum_link[1][long])
        sum_list_index = range(0,a_list.__len__())
        random.seed(seed)
        random.shuffle(sum_list_index)
        test_link_index = sum_list_index[0:sum_list_index.__len__()/test_divide]
        test_matrix = np.zeros(m.shape)
        for i in test_link_index:
            test_matrix[a_list[i],b_list[i]]=1
            test_matrix[b_list[i],a_list[i]]=1
        return [m-test_matrix,test_matrix]


    def form_matrix(self):
        target_list = []
        enzy_list = []
        path_list = []
        structure_list = []
        for i in self.drug_list:
            dict = self.all_message_dict[i]
            target_list.append(dict["target_list_value"])
            enzy_list.append(dict["enzy_list_value"])
            path_list.append(dict["path_list_value"])
            structure_list.append(dict["structure_list_value"])

        self.target_matrix = np.matrix(target_list)
        self.enzy_matrix = np.matrix(enzy_list)
        self.path_matrix = np.matrix(path_list)
        self.structure_matrix = np.matrix(structure_list)

def cross_validation(sim_list,drug_drug_interaction_matrix,target_matrix,enzy_matrix,path_matrix,structure_matrix,num_CV,seed,mu,delta,lam,max_iter,eps,method='sflln'):
    folds=list(KFold(len(drug_drug_interaction_matrix),n_folds=num_CV,shuffle=True,random_state=np.random.RandomState(seed)))
    metrics=[]
    for i in range(0,num_CV):
        train_index=folds[i][0]
        test_index=folds[i][1]
        Xs=[np.transpose(target_matrix[train_index]),np.transpose(enzy_matrix[train_index]),np.transpose(path_matrix[train_index]),np.transpose(structure_matrix[train_index])]
        X_test=[np.transpose(target_matrix[test_index]),np.transpose(enzy_matrix[test_index]),np.transpose(path_matrix[test_index]),np.transpose(structure_matrix[test_index])]
#        global X_test,Xs
        train_tar_sim=cal_Jaccard_sim(Xs[0])
        train_tar_sim=matrix_normalize(train_tar_sim)
        train_enzy_sim=cal_Jaccard_sim(Xs[1])
        train_enzy_sim=matrix_normalize(train_enzy_sim)
        train_path_sim=cal_Jaccard_sim(Xs[2])
        train_path_sim=matrix_normalize(train_path_sim)
        train_str_sim=cal_Jaccard_sim(Xs[3])
        train_str_sim=matrix_normalize(train_str_sim)
        new_sim_list=[]
        for i in range(len(sim_list)):
            new_sim_list.append(sim_list[i][test_index][:,train_index])
        if method=='sflln':
            y_pred=SFLLN_pred(sim_list,Xs,X_test,drug_drug_interaction_matrix,train_index,test_index,mu,delta,lam,max_iter,eps)
            pred_edge=[]
            test_edge=[]
            test_matrix=drug_drug_interaction_matrix[test_index][:,train_index]
            for i in range(len(test_index)):
                for j in range(len(train_index)):
                    pred_edge.append(y_pred[i,j])
                    test_edge.append(test_matrix[i,j])
            metrics.append(model_eval_vec(pred_edge,test_edge))
            print i,metrics
    return metrics

def cross_valid_semi_sup(sim_list,drug_drug_interaction_matrix,target_matrix,enzy_matrix,path_matrix,structure_matrix,num_CV,seed,mu,delta,lam,max_iter,eps,method='sflln'):
    row_num=drug_drug_interaction_matrix.shape[0]
    col_num=drug_drug_interaction_matrix.shape[1]
    link_position=[]
    Xs=[np.transpose(target_matrix),np.transpose(enzy_matrix),np.transpose(path_matrix),np.transpose(structure_matrix)]
    Ls = []
    metrics=[]
    for i in range(row_num):
        for j in range(i+1,col_num):
            if drug_drug_interaction_matrix[i,j]==1:
                link_position.append([i,j])
    link_position=array(link_position)
    folds=list(KFold(len(link_position),n_folds=num_CV,shuffle=True,random_state=np.random.RandomState(seed)))
    for i in range(num_CV):
        train_index=folds[i][0]
        test_index=folds[i][1]
        train_d_d_interaction=deepcopy(drug_drug_interaction_matrix)
        for position in link_position[test_index]:
            train_d_d_interaction[position[0],position[1]]=0
            train_d_d_interaction[position[1],position[0]]=0
        if method=='sflln':
            nieghbor_sim=fast_calculate(train_d_d_interaction,np.int(floor(606)))
            Ls.append(delta*(np.eye(len(drug_drug_interaction_matrix))-nieghbor_sim))
            result=SFLLN(Xs,Ls,train_d_d_interaction,mu,delta,lam,max_iter,eps)
            construct_matrix=result[1]
            construct_matrix=construct_matrix+construct_matrix.T
            Ls=Ls[0:1]
        pred_lost_edge=[]
        test_lost_edge=[]
        for row in range(row_num):
            for col in range(col_num):
                if train_d_d_interaction[row,col]==0:

                    pred_lost_edge.append(construct_matrix[row,col])
                    test_lost_edge.append(drug_drug_interaction_matrix[row,col])
        print sum(test_lost_edge)
        print sum(pred_lost_edge)
        print sum(drug_drug_interaction_matrix)
        metrics.append(model_eval_vec(pred_lost_edge,test_lost_edge))
        print i,metrics
    return metrics

def model_eval(result_matrix,test_matrix):
    real_labels=np.matrix(np.reshape(test_matrix,test_matrix.shape[0]*test_matrix.shape[1],1))
    pred_labels=np.matrix(np.reshape(result_matrix,result_matrix.shape[0]*result_matrix.shape[1],1))
    precision, recall, pr_thresholds = precision_recall_curve(real_labels.A[0], pred_labels.A[0])
    aupr_score = auc(recall, precision)
    fpr, tpr, auc_thresholds = roc_curve(real_labels.A[0], pred_labels.A[0])
    auc_score = auc(fpr, tpr)
    all_F_measure=np.zeros(len(pr_thresholds))
    for k in range(0,len(pr_thresholds)):
        if (precision[k]+precision[k])>0:
            all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
        else:
            all_F_measure[k]=0
    max_index=all_F_measure.argmax()
    threshold=pr_thresholds[max_index]
#    print real_labels
    predicted_score=np.zeros(len(real_labels.A[0]))
#    global predicted_score
#    print pred_labels.A[0]
    for i in range(len(pred_labels.A[0])):
        if pred_labels.A[0][i]>threshold:
            predicted_score[i]=1
#    predicted_score[pred_labels.A[0]>threshold]=1
    f=f1_score(real_labels.A[0],predicted_score)
    accuracy=accuracy_score(real_labels.A[0],predicted_score)
    precision=precision_score(real_labels.A[0],predicted_score)
    recall=recall_score(real_labels.A[0],predicted_score)
    print aupr_score,auc_score,f,accuracy,precision,recall
    return [aupr_score,auc_score,f,accuracy,precision,recall]

def model_eval_vec(pred_labels,real_labels):
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, pred_labels)
    print len(precision),len(recall),len(pr_thresholds)
    aupr_score = auc(recall, precision)
    fpr, tpr, auc_thresholds = roc_curve(real_labels, pred_labels)
    auc_score = auc(fpr, tpr)
    all_F_measure=np.zeros(len(pr_thresholds))
    for k in range(0,len(pr_thresholds)):
        if (precision[k]>0 and recall[k]>0):
            all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
        else:
            all_F_measure[k]=0
    print precision,recall,all_F_measure,pr_thresholds
    max_index=all_F_measure.argmax()
#    max_index=1
    threshold=pr_thresholds[max_index]
#    print real_labels
    predicted_score=np.zeros(len(real_labels))
#    global predicted_score
#    print pred_labels.A[0]
    for i in range(len(pred_labels)):
        if pred_labels[i]>threshold:
            predicted_score[i]=1
#    predicted_score[pred_labels.A[0]>threshold]=1
    f=f1_score(real_labels,predicted_score)
    accuracy=accuracy_score(real_labels,predicted_score)
    precision=precision_score(real_labels,predicted_score)
    recall=recall_score(real_labels,predicted_score)
    print aupr_score,auc_score,f,accuracy,precision,recall
    return [aupr_score,auc_score,f,accuracy,precision,recall]

def get_random_matrix(seed,i,long):
    np.random.seed(seed)
    a = np.matrix(np.random.rand(i.shape[0], long), dtype=float)
    if a.all():
        return a
    else:
        a = a +0.00001

  
def get_pos_part(u):
    return (np.abs(u)+u)/2

def get_neg_part_as_pos(u):
    return (np.abs(u)-u)/2


def cal_Jaccard_sim(matrix):
    numerator=matrix.T*matrix
    denominator=np.ones(np.shape(matrix.T))*matrix+matrix.T*np.ones(np.shape(matrix))-matrix.T*matrix
    return numerator/denominator


def fast_calculate(feature_matrix,neighbor_num):
    iteration_max=50
    mu=6
    X=feature_matrix
    row_num=X.shape[0]
    distance_matrix=np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(i+1,len(X)):
		#	distance_matrix[i,j]=np.linalg.norm([X[i,:],X[j,:]],2)
            distance_matrix[i,j]=np.sqrt(np.sum(np.square(X[i,:] - X[j,:])))  
    distance_matrix=distance_matrix+distance_matrix.T
#    print distance_matrix
    e=np.ones((row_num,1))
    distance_matrix=distance_matrix+np.diag(np.diag(e*e.T*float('inf')))
#    print distance_matrix
#    global distance_matrix
	#°´ÁÐÉýÐòÅÅÁÐ
    nearst_neighbor_matrix=np.zeros((row_num,row_num))
    for i in range(row_num):
        b = sorted(enumerate(distance_matrix[i]),key=lambda x:x[1])
        index =b[:(neighbor_num-1)]
        for d in index:
            nearst_neighbor_matrix[i,d[0]] = 1.0
      #      nearst_neighbor_matrix[d[0],i] = 1.0
    C=nearst_neighbor_matrix
#    global C
    np.random.seed(0)
    W= np.matrix(np.random.rand(row_num,row_num), dtype=float)
    W=np.multiply(C,W)
    lamda=mu*e
    P=X*X.T+lamda*e.T
    for i in range(iteration_max):
        Q=(np.multiply(C,W))*P
        W=np.multiply(np.multiply(C,W),P)/Q
        for i in range(len(W)):
            for j in range(len(W)):
                if W[i,j]==np.nan:
                    W[i,j]=0
    print W
    return W


def matrix_normalize(similarity_matrix):
	# row_num=shape(similarity_matrix)[0]
	# col_num=shape(similarity_matrix)[1]
    similarity_matrix=np.nan_to_num(similarity_matrix)
    for i in range(len(similarity_matrix)):
        similarity_matrix[i,i]=0
    for k in range(200):
        D=np.diag(flatten(np.nansum(similarity_matrix,1).tolist()))
#        print D
        D1=pinv(sqrt(D))
        similarity_matrix=D1*similarity_matrix*D1
    return similarity_matrix

def linear_neigh(s1,s2,s3,s4,p): 
    new_simi_tar_matrix=np.zeros((len(s1),len(s1)))
    new_simi_enzy_matrix=np.zeros((len(s2),len(s2)))
    new_simi_path_matrix=np.zeros((len(s3),len(s3)))
    new_simi_structure_matrix=np.zeros((len(s4),len(s4)))
    new_simi_tar_matrix=fast_calculate(s1,np.int(floor(606*p)))
    new_simi_enzy_matrix=fast_calculate(s2,np.int(floor(606*p)))
    new_simi_path_matrix=fast_calculate(s3,np.int(floor(606*p)))
    new_simi_structure_matrix=fast_calculate(s4,np.int(floor(606*p)))
    #
    new_simi_tar_matrix=matrix_normalize(new_simi_tar_matrix)
    new_simi_enzy_matrix=matrix_normalize(new_simi_enzy_matrix)
    new_simi_path_matrix=matrix_normalize(new_simi_path_matrix)
    new_simi_structure_matrix=matrix_normalize(new_simi_structure_matrix)
#
    new_sim_list=[new_simi_tar_matrix,new_simi_enzy_matrix,new_simi_path_matrix,new_simi_structure_matrix]
    return new_sim_list
    
def SFLLN_pred(new_sim_list,Xs,X_test,drug_drug_interaction_matrix,train_index,test_index,mu,delta,lam,max_iter,eps):
    Ls = []
    Ls.append(np.matrix(np.zeros((len(train_index),len(train_index)))))
    for i in new_sim_list:
        nor_i=matrix_normalize(i[train_index][:,train_index])
        Ls.append(delta*(np.eye(len(train_index))-nor_i))
    results=SFLLN(Xs,Ls,drug_drug_interaction_matrix[train_index][:,train_index],mu,delta,lam,max_iter,eps)
    Gs=results[0]
    y_pred=np.zeros((len(test_index),len(train_index)))
    for i in range(len(Xs)):
        y_pred=y_pred+np.transpose(X_test[i])*Gs[i]/np.sum(np.transpose(X_test[i])*Gs[i])
    return y_pred

def SFLLN(Xs,Ls,Y,mu,delta,lam,max_iter,eps):
    long = Y.shape[1]
    print (long)
    Gs = []
    elds = []
    ml = Xs.__len__()
    ms=ml-3
    print ml
    seed = 1
    for i in Xs:

        seed = seed+1
        Gs.append(get_random_matrix(seed,i,long))

        elds.append(np.matrix([1]*i.shape[0],dtype=float))
    L = np.zeros((Y.shape[0],Y.shape[0]))
    for i in range(0,ms):
        L = L +delta*Ls[i]
    Gs_old = ['']*ml
    As = ['']*ml
    As_pos = ['']*ml
    As_neg = ['']*ml
    Bs = ['']*ml
    Bs_pos = ['']*ml
    Bs_neg = ['']*ml
    check_step = 20
    F_mat = 0
    diff_G = []
    for t in range(0,max_iter):
        Q=Y
        for i in range(0,ml):
            Gs_old[i] = Gs[i]
            Q = Q+mu*np.transpose(Xs[i])*Gs[i]
        P = np.linalg.solve(L+(1+mu*(ml-1))*np.eye(Y.shape[0]),np.eye(Y.shape[0]))
        F_mat = P*Q
        for i in range(0,ml):
            As[i] = Xs[i]*((mu*np.eye(Y.shape[0])-(mu**2*np.transpose(P)))*np.transpose(Xs[i]))+lam*np.transpose(elds[i])*elds[i]
            where_are_nan = np.isnan(As[i])   
            As[i][where_are_nan] = 0.0
            As_pos[i] = (As[i]+np.abs(As[i]))/2
            As_neg[i] = (np.abs(As[i]) - As[i])/2
            Bs[i] = mu*Xs[i]*P*Y
            for j in range(0,ml):
                if i==j:
                    continue
                else:
                    Bs[i] = Bs[i] + mu**2*Xs[i]*np.transpose(P)*np.transpose(Xs[j])*Gs[j]
            Bs[i] = Bs[i]
            where_are_nan = np.isnan(Bs[i])   
            Bs[i][where_are_nan] = 0.0
            Bs_pos[i] = (Bs[i]+np.abs(Bs[i]))/2
            Bs_neg[i] = (np.abs(Bs[i])-Bs[i])/2


        for i in range(0,ml):
            a = np.sqrt(np.divide(Bs_pos[i]+As_neg[i]*Gs[i],Bs_neg[i]+As_pos[i]*Gs[i]))
            where_are_nan = np.isnan(a)   
            a[where_are_nan] = 0.0
            Gs[i] = np.multiply(Gs[i],a)
            Gs=np.nan_to_num(Gs)
        L = np.zeros(Y.shape[0])
        for i in range(0,ms):
            L = L +delta*Ls[i]
        L=np.nan_to_num(L)
        diff_G = [0.0]*ml
        for i in range(0,ml):
            diff_G[i] = np.linalg.norm(Gs[i]-Gs_old[i])/np.linalg.norm(Gs_old[i])
            # print 'diff_G',diff_G[i]
        loss_2=0
        loss_3=0
        for i in range(0,ml):
            loss_2 = loss_2+mu*np.linalg.norm(np.transpose(Xs[i])*Gs[i])
            loss_3=loss_3+lam*np.linalg.norm(Gs[i])
        loss=np.linalg.norm(Y-F_mat)+loss_2+delta*np.trace(F_mat.T*Ls[0]*F_mat)+loss_3
#        print t,loss
#        if t % check_step ==0:
#            print t,loss,np.mean(diff_G)
        if np.mean(diff_G) <= eps:
            return [Gs,F_mat,diff_G,t]
        if math.isnan(np.mean(diff_G))==True:
            return(np.nan_to_num([Gs,F_mat,diff_G,t]))
            print np.max(F_mat)
    return np.nan_to_num([Gs,F_mat,diff_G,P])


d = data()
d.get_data()
d.form_matrix()
s1 = d.target_matrix
s2 = d.enzy_matrix
s3 = d.path_matrix
s4 = d.structure_matrix
Y = d.get_drug_drug_interactive_matrix()

simi_tar_matrix=np.zeros((len(s1),len(s1)))
simi_enzy_matrix=np.zeros((len(s2),len(s2)))
simi_path_matrix=np.zeros((len(s3),len(s3)))
simi_structure_matrix=np.zeros((len(s4),len(s4)))
simi_tar_matrix=cal_Jaccard_sim(s1.T)
simi_enzy_matrix=cal_Jaccard_sim(s2.T)
simi_path_matrix=cal_Jaccard_sim(s3.T)
simi_structure_matrix=cal_Jaccard_sim(s4.T)

simi_tar_matrix=matrix_normalize(simi_tar_matrix)
simi_enzy_matrix=matrix_normalize(simi_enzy_matrix)
simi_path_matrix=matrix_normalize(simi_path_matrix)
simi_structure_matrix=matrix_normalize(simi_structure_matrix)

sim_list=[simi_tar_matrix,simi_enzy_matrix,simi_path_matrix,simi_structure_matrix]




regularization=linear_neigh(s1,s2,s3,s4,1)
#start = time.clock()
metr=cross_validation(regularization,Y,s1,s2,s3,s4,5,4,0.001,0.0,0.0001,3000,0.005)
#end = time.clock()
#run_time=end-start
np.savetxt('metr_SFLLN.txt',metr)
#sim_list=linear_neigh(s1,s2,s3,s4,1)
##
#start = time.clock()
metr_semi=cross_valid_semi_sup(sim_list,Y,s1,s2,s3,s4,5,1,0.0001,0.1,1,3000,0.005)
np.savetxt('metr_semi_SFLLN.txt',metr)
#end = time.clock()
#run_time=end-start
