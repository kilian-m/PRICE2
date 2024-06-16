import os
import sqlite3
import numpy as np

from pickle import loads, dumps

from multiprocessing import Pool

from price2.locus import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment

from scipy.optimize import minimize

from numba import njit, jit, prange
from numba.typed import List
from numba.core.errors import NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

from tqdm import notebook

import time


class ORFActivityEstimator:

    def __init__(self,
                 reads_db_path: str,
                 loci_db_path: str,
                 runs_db_path: str,
                 ):
        self.reads_db_path = reads_db_path
        self.loci_db_path = loci_db_path
        self.runs_db_path = runs_db_path

        loci_db = sqlite3.connect(loci_db_path)
        cur = loci_db.cursor()

        cur.execute('SELECT * FROM loci')
        self.loci = {id: loads(blob) for id, blob in cur.fetchall()}

    
    def run_orf_deconvolution_parallel(self, processes: int=32):
        loci_ids = [loc.id for loc in self.loci.values()]

        db_paths = (self.reads_db_path, self.loci_db_path, self.runs_db_path)

        with Pool(processes) as pool:
            map_iterator = pool.imap_unordered(process_loc, [(li, db_paths) for li in loci_ids])
            for result in notebook.tqdm(map_iterator, total=len(loci_ids)):
                self.loci[result[0]].opt_result = result[1]
                self.loci[result[0]].norm_result = result[2]



def process_loc(locid_dbpath, λ=0):

    loc_id, db_paths = locid_dbpath
    #start_time = time.time()
    reads_db_path, loci_db_path, runs_db_path = db_paths
    
    ###############################
    ### get data from databases ###
    ###############################

    loci_db = sqlite3.connect(loci_db_path)
    cur = loci_db.cursor()

    cur.execute('''SELECT * FROM loci
                WHERE locus_id = ?''', (loc_id,))
    _, blob = cur.fetchone()
    loc = loads(blob)

    loci_db.close()

    # load runs
    run_db = sqlite3.connect(runs_db_path)
    cur = run_db.cursor()

    cur.execute('SELECT * FROM runs')
    runs = [loads(blob) for id, blob in cur.fetchall()]
    runs_dict = {run.id: run for run in runs}

    run_db.close()

    # open reads db
    read_db = sqlite3.connect(reads_db_path)
    read_cursor = read_db.cursor() 

    reads_dfs = read_cursor.execute('''
        SELECT * FROM reads 
        WHERE locus_id = ?
        ''', (loc.id,))
    
    ################################
    ### generate data structures ###
    ################################

    loc.make_compatibility_groups()
    loc.make_equivalence_groups(runs)

    ################################
    ### process read information ###
    ################################

    for loc_id, run_id, blob in reads_dfs:
        reads_dfs = loads(blob)
        for _, read_df in reads_dfs.groupby('read_id'):
            rsa = RiboSeqAlignment(read_df)
            loc.add_ribo_seq_alignment(rsa, runs_dict[run_id])
    ##############################################################################
    ##### prepare data to be used efficiently by the numba objective function ####
    ##############################################################################
    
    rgrs = list(loc.rgr_set)
    
    run_read_counts = np.array([run.read_count for run in runs])

    cm_lut = np.zeros((len(runs), runs[0].cleavage_model.cds_lut.shape[0], 4, 2))
    for i, run in enumerate(runs):
        cm_lut[i,:,3,:] = run.cleavage_model.noise_lut/3
        cm_lut[i,:,:3,:] = run.cleavage_model.cds_lut

    egs = []
    for run in runs:
        l = []

        for (rgr_frame, read_length, oua), eg in loc.egs[run].items():
            temp = []
            for rgr, frame in rgr_frame:
                if frame == None:
                    frame = 3
                temp.append((rgr.index, frame))
            l.append((eg.length, eg.read_count, read_length, int(oua), temp))
        egs.append(l)
    
    num_rgrs = len(rgrs)

    initial_guess = np.full((num_rgrs, len(runs)), 1)
    initial_guess = initial_guess.flatten()

    loc_id = loc.id

    egs_unconverted = egs
    egs = List()

    for run in egs_unconverted:
        l = List()
        for eg in run:
            temp = List()
            for rgr, frame in eg[4]:
                temp.append((rgr, frame))
            l.append((eg[0], eg[1], eg[2], eg[3], temp))
        egs.append(l)
    
    ############################
    ### run the optimization ###
    ############################
    try:
        optimization_result = minimize(
            objective_function_grad,
            initial_guess,
            args=(run_read_counts, cm_lut, egs, num_rgrs, λ),
            method='L-BFGS-B', 
            tol=1e-10,
            jac=True,
            options={'maxiter': 10_000, 'maxfun': 1e6}
        )
    except ZeroDivisionError:
        pass

    tmp = np.exp(optimization_result.x).reshape(num_rgrs, len(run_read_counts))
    normalized_result = tmp/tmp.sum(axis=0)

    #end_time = time.time()
    #print(f'finished processing locus {loc_id} in {end_time-start_time:.2f} seconds')
    return (loc_id, optimization_result, normalized_result)


@jit(nopython=True, parallel=True, cache=True)
def objective_function_grad(
        x,
        run_read_counts, cm_lut, egs,
        num_rgrs,
        λ: float
        ) -> float:
    x = np.exp(x)

    grads = np.zeros_like(x)

    num_runs = len(run_read_counts)

    loss = 0

    for run_index in prange(num_runs):
        for eg in egs[run_index]:
            δ_derived = np.zeros_like(x)
            activity = 0
            for rgr_index, frame in eg[4]:
                activity += x[rgr_index*num_runs + run_index] * cm_lut[run_index, eg[2], frame, eg[3]]
                δ_derived[rgr_index*num_runs + run_index] += x[rgr_index*num_runs+run_index]*cm_lut[run_index, eg[2], frame, eg[3]]
            
            δ = run_read_counts[run_index] * eg[0] * activity
            δ_derived *= run_read_counts[run_index] * eg[0]

            y = eg[1]

            if y == 0 and δ == 0:
                continue

            loss += δ - y * np.log(δ)
            
            grads += δ_derived - y * δ_derived / δ
        
    penalty = 0
    for rgr_index in prange(num_rgrs):
        s = 0
        for run_index in range(num_runs): # prange?
            s += x[rgr_index*num_runs + run_index] ** 2
        s_sqrt = s**.5
        penalty += s_sqrt
        for run_index in range(num_runs): # prange?
            grads[rgr_index*num_runs + run_index] += λ * x[rgr_index*num_runs + run_index] ** 2 / s_sqrt
    
    return loss + λ * penalty, grads


@njit
def run_orf_deconvolution_em_numba(
        cm_lut: np.ndarray,
        egs,
        rgr_lengths: np.ndarray,
        num_rgrs: int,
        iterations: int,
        activity_change_cutoff: float,
        ) -> None:

    activities = np.full(num_rgrs, 1/num_rgrs)

    num_egs = len(egs)
    
    for i in range(iterations):
        rgr_read_counts = np.zeros(num_rgrs)
        # E-step
        for eg_index in range(num_egs):
            read_count = egs[eg_index][1]
            likelihoods = np.empty(len(egs[eg_index][4]))
            for j, (rgr_index, frame) in enumerate(egs[eg_index][4]):
                likelihoods[j] = activities[rgr_index] * cm_lut[egs[eg_index][2], frame, egs[eg_index][3]]
            likelihood_sum = likelihoods.sum()
            if likelihood_sum > 0:
                p = likelihoods/likelihood_sum
            else:
                p = np.full(len(likelihoods), 1/len(egs[eg_index][4]))
            for j, (rgr_index, frame) in enumerate(egs[eg_index][4]):
                rgr_read_counts[rgr_index] += read_count * p[j]

        # M-step
        new_activities = np.zeros(num_rgrs)
        for rgr_index in range(num_rgrs):
            new_activities[rgr_index] = rgr_read_counts[rgr_index] / rgr_lengths[rgr_index]
        new_activities /= new_activities.sum()

        if i>1:
            activity_change = sum(np.abs(new_activities - activities))
            if activity_change < activity_change_cutoff:
                break

        activities = new_activities
    
    return activities
