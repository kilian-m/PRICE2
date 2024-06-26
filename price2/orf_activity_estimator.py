import os
import sqlite3
import numpy as np
import pandas as pd

from pickle import loads, dumps
import zlib

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

    
    def run_orf_deconvolution(self, processes: int=32):
        loci_ids = [loc.id for loc in self.loci.values()]
        
        db_paths = (self.reads_db_path, self.loci_db_path, self.runs_db_path)

        performance_measurements = []

        with Pool(processes) as pool:
            loci_ids = loci_ids[100:300]
            map_iterator = pool.imap_unordered(process_loc, [(li, db_paths) for li in loci_ids])
            for result in notebook.tqdm(map_iterator, total=len(loci_ids)):
                self.loci[result[0]].opt_result = result[1]
                self.loci[result[0]].norm_result = result[2]
                performance_measurements.append(result[3])
        
        self.performance_df = pd.DataFrame(performance_measurements, columns=['loc_id', 'overall_time', 'num_reads', 'cg_time', 'eg_time', 'rsa_time', 'opt_time', 'num_cgs', 'num_egs', 'num_orfs', 'num_transcripts', 'num_iterations'])



def process_loc(locid_dbpath, λ=0):

    t_start = time.time() # performance measurement

    loc_id, db_paths = locid_dbpath
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
    t1 = time.time() # performance measurement
    loc.make_compatibility_groups()
    t2 = time.time() # performance measurement
    cg_time = t2-t1 # performance measurement

    t1 = time.time() # performance measurement
    loc.make_equivalence_groups(runs)
    t2 = time.time() # performance measurement
    eg_time = t2-t1 # performance measurement

    ################################
    ### process read information ###
    ################################

    num_reads = 0 # performance measurement
    t1 = time.time() # performance measurement
    for loc_id, run_id, blob in reads_dfs:
        reads_df = loads(zlib.decompress(blob))
        reads_df['chrom'] = loc.iv.chrom
        reads_df['strand'] = loc.iv.strand
        reads_df['read_id'] = reads_df['is_first_iv'].cumsum()
        for _, read_df in reads_df.groupby('read_id'):
            rsa = RiboSeqAlignment(read_df)
            loc.add_ribo_seq_alignments(rsa, runs_dict[run_id], read_df.iloc[0]['count'])
            num_reads += read_df.iloc[0]['count'] # performance measurement
    t2 = time.time() # performance measurement
    rsa_time = t2-t1 # performance measurement

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

    num_cgs = len(loc.cgs) # performance measurement
    num_egs = len(egs[0]) # performance measurement
    

    num_orfs = 0 # performance measurement
    num_transcripts = 0 # performance measurement
    for rgr in rgrs: # performance measurement
        if rgr.type == 'ORF': # performance measurement
            num_orfs += 1 # performance measurement
        elif rgr.type == 'NOISE': # performance measurement
            num_transcripts += 1 # performance measurement
        else: # performance measurement
            raise ValueError(f'unknown rgr type: {rgr.type}') # performance measurement


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
    t1 = time.time() # performance measurement
    try:
        optimization_result = minimize(
            objective_function_grad,
            initial_guess,
            args=(run_read_counts, cm_lut, egs, num_rgrs, λ),
            method='L-BFGS-B', 
            #tol=1e-10,
            jac=True,
            options={'maxiter': 10_000, 'maxfun': 1e6, 'gtol': 1e-1}
        )
    except ZeroDivisionError:
        pass

    t2 = time.time() # performance measurement
    opt_time = t2-t1 # performance measurement

    tmp = np.exp(optimization_result.x).reshape(num_rgrs, len(run_read_counts))
    normalized_result = tmp/tmp.sum(axis=0)

    t_end = time.time() # performance measurement
    overall_time = t_end-t_start # performance measurement

    num_iterations = optimization_result.nit # performance measurement

    performance_measurements = [loc_id, overall_time, num_reads, cg_time, eg_time, rsa_time, opt_time, num_cgs, num_egs, num_orfs, num_transcripts, num_iterations]

    return (loc_id, optimization_result, normalized_result, performance_measurements)


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
