"""
@author: gsivaraman@anl.gov
"""
from __future__ import print_function
import  GPyOpt
import argparse
import json, os, subprocess
import numpy as np
from ase.io import read, write
#import commands  ##--> Depreciated! WIll switch to subprocess once quippy is ported to Python 3+
from sklearn.metrics import mean_absolute_error, r2_score
from pprint import pprint
from time import time 
from copy import deepcopy
from activesample import activesample

try:
    path = os.getenv('QUIP_PATH')    
except:
    print("'QUIP_PATH' not set!")


def get_argparse():
    """
    A function to parse all the input arguments
    """
    parser = argparse.ArgumentParser(description='A python workflow to run  active learning for GAP')
    parser.add_argument('-xyz','--xyzfilename', type=str,required=True,help='Trajectory filename (xyz formated)')
    parser.add_argument('-Ns','--nsample',type=int,metavar='',\
                       help="Minimum sample size",default=10)
    parser.add_argument('-Nm','--nminclust',type=int,metavar='',\
                               help="No of Minimum clusters. Tweak to reduce noise points",default=10)
    parser.add_argument('-c','--cutoff',nargs='+',type=float,required=True,\
         help="Cutt off  range for SOAP. Usage: '-c min max'. Additional dependency: common sense!")
    parser.add_argument('-s','--nsparse',nargs='+',type=float,required=True,\
         help="Nsparse for SOAP. Usage: '-s min max'. Additional dependency: common sense ")
    parser.add_argument('-nl','--nlmax',nargs='+',type=float,required=True,\
                        help="Nmax, Lmax range. Usage: '-nl min max'. Additional dependency: common sense! ")
    parser.add_argument('-Nbo','--Nopt',nargs='+',type=float,required=True,\
                        help="Number of exploration and optimization steps for BO. e.g.'-Nbo 25 50' ")

    return parser.parse_args()




def run_quip(cutoff=5.0,delta=1.0,n_sparse=100,nlmax=4):
    """
    A method to launch  quip and postprocess.
    """
    if os.path.isfile('quip_test.xyz'):
        os.remove('quip_test.xyz')
    if os.path.isfile('gap.xml'):
        os.remove('gap.xml')
    
    start = time()
           
    runtraining =  path+"teach_sparse at_file=./train.extxyz  gap={soap  cutoff="+str(cutoff)+"  n_sparse="+str(n_sparse)+"  covariance_type=dot_product sparse_method=cur_points  delta="+str(delta)+"  zeta=4 l_max="+str(nlmax)+"  n_max="+str(nlmax)+"  atom_sigma=0.5  cutoff_transition_width=0.5  add_species} e0={Hf:-2.70516846:O:-0.01277342}  gp_file=gap.xml default_sigma={0.0001 0.0001 0.01 .01} sparse_jitter=1.0e-8 energy_parameter_name=energy"
    output = subprocess.check_output(runtraining,stderr=subprocess.STDOUT, shell=True)

    end = time()

    print("\nTraining Completed in {} sec".format(end - start))


    evaluate = path+"quip  E=T   atoms_filename=./test.extxyz   param_filename=gap.xml  | grep AT | sed 's/AT//' >> quip_test.xyz" 
    pout = subprocess.check_output(evaluate,stderr=subprocess.STDOUT, shell=True)
    inp = read('test.extxyz',':')
    inenergy = [ei.get_potential_energy() for ei  in inp ]
    output = read('quip_test.xyz',':')
    outenergy = [eo.get_potential_energy() for eo in output ]
    if len(inenergy) == len(outenergy) :
        mae  = mean_absolute_error(np.asarray(inenergy), np.asarray(outenergy))
        rval  = r2_score(np.asarray(inenergy), np.asarray(outenergy)) 

        return [mae, rval ]

    else:
        print("Memory blow up for cutoff={}, n_sparse={}".format(cutoff,n_sparse))
        return [float(20000), float(-2000)]


def gen_trials(minclustlen,maxclustlen):
    '''
    Generate the trials in the   sampling width (Kmax, Kmin) 
    :param minclustlen: Kmin (int)
    :param maxclustlen: Kmax (int) 
    Return type: Trials  (list)
    '''
    if maxclustlen <= 50 :
        trialscale = int( np.floor(minclustlen/2) ) #int( np.floor(data.nsample/2) )
        maxtrial  = int( np.floor(maxclustlen/trialscale) )
        trials = [trialscale*t for t in range(maxtrial,0,-1)]
    elif maxclustlen > 50 and  maxclustlen <= 200 :
        trialscale = int( np.floor( (minclustlen/2 ) * (minclustlen/2 )  ) )
        maxtrial  = int( np.floor(maxclustlen/trialscale) )
        trials = [trialscale*t for t in range(maxtrial,0,-1)] +  [u*int(np.floor(minclustlen/2))  for u  in range( int(np.floor(minclustlen/2)) -1,0,-1)]
    elif maxclustlen > 200 :
        trialscale = int( minclustlen**2    )
        maxtrial  = int( np.floor(maxclustlen/trialscale) )
        trials = [trialscale*t for t in range(maxtrial,0,-1)] + [u*int(np.floor(minclustlen/2)) for u in range( int(minclustlen) ,0,-1)]
    return trials


def f(x):
    """
    Surrogate function over the error metric to be optimized
    """
   
    evaluation = run_quip(cutoff = float(x[:,0]), delta = float(x[:,1]), n_sparse = float(x[:,2]), nlmax = float(x[:,3]))
    print("\nParam: {}, {}, {}, {}  |  MAE : {}, R2: {}".format(float(x[:,0]),float(x[:,1]),float(x[:,2]),float(x[:,3]) ,evaluation[0],evaluation[1]))

    return evaluation[0]


def plot_metric(track_metric):
    '''
    Plot the metric evolution over the trial using this function
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.style.use('seaborn')
    cmap=plt.cm.get_cmap('brg')
    plt.rcParams["font.family"] = "Arial"
    x = range(1, len(track_metric)+1 )
    plt.plot(x, track_metric, c=cmap(0.40),linewidth=3.0)
    plt.title('Active learning',fontsize=32)
    plt.xlabel('Trial',fontsize=24)
    plt.ylabel('MAE (ev)',fontsize=24)
    plt.grid('True')
    plt.tight_layout()
    plt.draw()
    plt.savefig('AL.png',dpi=300)
    plt.savefig('AL.svg',dpi=300,format='svg')
    c = 'Done'
    return c


def main():
    '''
    Gather our code here 
    '''
    args = get_argparse() 

    cutoff = tuple(args.cutoff)
    sparse = tuple(args.nsparse)
    nlmax  = tuple(args.nlmax)
    Nopt = tuple(args.Nopt)
    trackerjson = {}
    track_metric = [] # For plotting the error metric
    trackerjson['clusters'] = []

    data  = activesample(args.xyzfilename,nsample=args.nsample,nminclust=args.nminclust)
    Nclust, Nnoise, clust = data.gen_cluster()
    clustlen = []

    for Ni, Ntraj in clust.items():
        clustlen.append(len(Ntraj) )

    maxclustlen = max(clustlen)
    minclustlen = min(clustlen)

    print("\nNumber of elements in the smallest, largest cluster is {}, {}\n".format(minclustlen,maxclustlen))
    print("\n Nnoise : {}, Nclusters : {}\n".format(Nnoise,Nclust))

    trackerjson['Nclusters'] = Nclust
    trackerjson['Nnoise']  = Nnoise
    trackerjson['clusters'].append(clust)
    trackerjson['paritition_trials'] = []
    rmse_opt = 10000.

    trials = gen_trials(minclustlen,maxclustlen) 

    Natom = data.exyztrj[0].get_number_of_atoms()     
    count = 1 
    trainlen_last = 0
    print("\n The trials will run in the sampling width interval : ({},{}) \n".format(max(trials),min(trials) )) 

    for Ntrial in trials :
        # set(np.random.randint(data.nsample, maxclustlen, size=10)): #range(1,trialsize+2):
        
        cmd  = "rm train.extxyz test.extxyz gap.xml* quip_test.xyz "
        rmout = subprocess.call(cmd, shell=True)
        #print(rmout)
        data.truncate = deepcopy(Ntrial)  #int(np.floor(maxclustlen /Ntrial) )
        print("\n\nBeginning  trial number : {} with a sampling width of {}".format(count,data.truncate)) 
        trainlist, testlist = data.clusterpartition()
        train_lennew = len(trainlist)
       
        print("\n\nNumer of training and test configs: {} , {}".format(len(trainlist), len(testlist) ) )
        
        if train_lennew > trainlen_last :
            print("\n {} new learning configs added by the active sampler".format(train_lennew - trainlen_last ) )
            data.writeconfigs()
        
            bounds = [{'name': 'cutoff',          'type': 'continuous',  'domain': (cutoff[0], cutoff[1])},\
            {'name':'delta',            'type': 'discrete',  'domain': (0.01, 0.1, 1.0)},\
            {'name': 'n_sparse',        'type': 'discrete',  'domain': np.arange(sparse[0],sparse[1]+1,100)},\
            {'name':'nlmax',            'type': 'discrete',  'domain': np.arange(nlmax[0],nlmax[1]+1)},]

            # optimizer
            opt_quip = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,initial_design_numdata = int(Nopt[0]),
                                             model_type="GP_MCMC",
                                             acquisition_type='EI_MCMC', #EI
                                             evaluator_type="predictive",  # Expected Improvement
                                             exact_feval = False,
                                             maximize=False) ##--> True only for r2

            # optimize MP model
            print("\nBegin Optimization run \t")
            opt_quip.run_optimization(max_iter=int(Nopt[1]) )
            hyperdict = {}
            for num in range(len(bounds)):
                hyperdict[bounds[num]["name"] ] = opt_quip.x_opt[num]
            hyperdict["MAE"] = opt_quip.fx_opt
            trackerjson['paritition_trials'].append({'test': trainlist, 'train': testlist,'hyperparam': hyperdict })
            trainlen_last = deepcopy(train_lennew) #---> Update only if configs increased over iterations!
            if opt_quip.fx_opt < rmse_opt :
                rmse_opt = float(opt_quip.fx_opt)
                best_train = deepcopy(trainlist)
                best_test = deepcopy(testlist)
                print("\n MAE lowered in this trial: {} eV/Atom".format(rmse_opt/Natom))
                track_metric.append(rmse_opt/Natom)
                best_hyperparam = deepcopy(hyperdict)
                if count != 1  and np.round(rmse_opt/Natom,4)  <=  .001 :
                   print("\n Optimal configs found! on {}th trial with hyper parameters : {}\n".format(count, best_hyperparam))
                   with open('activelearned_quipconfigs.json', 'w') as outfile:
                      json.dump(trackerjson, outfile,indent=4)
                   print("\nActive learning history written to 'activelearned_quipconfigs.json' ")

                   train_xyz = []
                   test_xyz = []
                   for i in best_train:
                      train_xyz.append(data.exyztrj[i])
                   for j in best_test:
                      test_xyz.append(data.exyztrj[j])

                   write("opt_train.extxyz", train_xyz)
                   write("opt_test.extxyz", test_xyz)
                   print("\nActive learnied configurations written to 'opt_train.extxyz','opt_test.extxyz' ")
                   break
        else:
	    print("No new configs found in the {} trial. Skipping!".format(count))
        count += 1
        
    plot_metric(track_metric)

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
