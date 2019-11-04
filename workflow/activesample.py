#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:46:10 2019
    
@author: gsivaraman@anl.gov
"""

import mdtraj as md
import numpy as np
from hdbscan import HDBSCAN
import os
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from ase.io import read, write


class activesample:
    """
    A python class to perform active learning over a given AIMD trajectory to automate sampling of the training configurations for the GAP IP model
    """

    
    def __init__(self,exyzfile,nsample=10,nminclust=10,truncate=1):
        """
        Input Args:
        exyztrj (str): File name of AIMD trajectory (ase exyz format)
        nsample (int): Number of samples for unsupervised learning
        nminclust (int): Minimum number of clusters
        truncate (int): Truncation level for sampling clusters. '1' for default random sampling
        """
        
        self.exyzfile = exyzfile
        self.nsample = nsample
        self.nminclust = nminclust
        self.truncate = truncate
        self.exyztrj = read(exyzfile,':')
        if not os.path.isfile(exyzfile.strip('.extxyz') + '.pdb' ):
            write(exyzfile.strip('.extxyz') +'.pdb', self.exyztrj,format='proteindatabank')
        self.pdbtrj = md.load_pdb(exyzfile.strip('.extxyz') +'.pdb')
        self.distances = self.compute_dist()
        self.clustdict = {}
        self.trainlist = []
        self.testlist = []
        

    def compute_dist(self):
        """
        Compute the rmsd distance matrix from the input trajectory
        return:
        distmat : numpy distance matrix
        """
        distmat = np.empty((self.pdbtrj.n_frames, self.pdbtrj.n_frames))
        for i in range(self.pdbtrj.n_frames):
            distmat[i] = md.rmsd(self.pdbtrj, self.pdbtrj, i)
        return distmat


    def plot_elbow(self):
        """
        A method to perform elbow plot of the distance matrix. The epsilon parameter
        can be chosen by visually inspecting the elbow.
        """
        plotthisarray = self.distances[:,-1]
        fig = plt.figure()
        plt.plot(-np.sort(-plotthisarray))
        #plt.show()
        return fig


    def gen_cluster(self):
        """
        Perform HDBSCAN clustering on a given distance matrix
        return:
        n_clusters_ (int) : Total number of clusters excluding noise
        n_noise_ (int) : Total number of noise points
        clustdict (dict) : Cluster number keys. List values of AIMD trajectory index
        """
        hdb = HDBSCAN(min_samples=self.nsample, min_cluster_size=self.nminclust).fit(self.distances) # allow_single_cluster=True
        labels = hdb.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        lclustdict = {}
        for index, clustnum in enumerate(labels):
            if clustnum != -1 :
                if clustnum not  in lclustdict.keys():
                    lclustdict[clustnum] = []
                lclustdict[clustnum].append(index)

        self.clustdict = deepcopy(lclustdict)
        return n_clusters_,n_noise_, lclustdict



    def clusterpartition(self):
        """
        A method to create test/ train split by paritioning the cluster
        Return type: traininglist, testlist (list)
        """
    
        train_list = []
        test_list = []
    
        for key, trjlist in self.clustdict.items():
            ltrain = list()
            ltest = list()
            if self.truncate == 1:
                ltrain.append(random.choice(trjlist) )
                ltest = random.sample(trjlist,4)
                while ltrain[0]  in  ltest:
                    ltest = random.sample(trjlist,4)
            
            elif self.truncate  != 1:
                if len(trjlist) <= self.truncate :
                    #muind = int(np.floor(len(trjlist)/2) ) 
                    #ltrain.append(trjlist[muind] )
                    ltrain.append(random.choice(trjlist))
                    ltest = random.sample(trjlist,1) 
                    while ltrain[0] in  ltest:
                        ltest = random.sample(trjlist,1) 
                elif len(trjlist) >  self.truncate :
                    ltrain = trjlist[0::self.truncate]
                    ltest = trjlist[2::self.truncate]
            for k in ltrain:
                train_list.append(k)
            for v in ltest:
                test_list.append(v)
        self.trainlist = deepcopy(train_list)
        self.testlist = deepcopy(test_list)
        return train_list, test_list


    def writeconfigs(self):
        """
        Write training and test configurations to exyz files!
        """
        train_xyz = []
        test_xyz = []
        for i in self.trainlist:
            train_xyz.append(self.exyztrj[i])
        for j in self.testlist:
            test_xyz.append(self.exyztrj[j])

        write("train.extxyz", train_xyz)
        write("test.extxyz", test_xyz)
