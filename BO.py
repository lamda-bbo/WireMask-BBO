from place_db import PlaceDB
from TuRBO.turbo import Turbo1
from utils import bo_placer, write_final_placement, rank_macros
from common import grid_setting, my_inf
import os
import random
import argparse
import csv
import numpy as np

class placement_eval:
    def __init__(self, dim, grid_num, grid_size, placedb, node_id_ls, csv_writer, csv_file, placement_save_dir):
        self.dim = dim
        self.lb = 0 * np.ones(dim)
        self.ub = grid_num * np.ones(dim)
        self.grid_num = grid_num
        self.grid_size = grid_size
        self.placedb = placedb
        self.node_id_ls = node_id_ls
        self.csv_writer = csv_writer
        self.csv_file = csv_file    
        self.best_hpwl = my_inf 
        self.placement_save_dir = placement_save_dir

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        place_record = {}
        node_id_ls = self.node_id_ls.copy()
        for i in range(len(node_id_ls)):
            place_record[node_id_ls[i]] = {}
            place_record[node_id_ls[i]]["loc_x"] = x[i*2]
            place_record[node_id_ls[i]]["loc_y"] = x[i*2+1]
        placed_macro, hpwl = bo_placer(node_id_ls, self.placedb, self.grid_num, self.grid_size, place_record, self.csv_writer, self.csv_file)
        if hpwl < self.best_hpwl:
            self.best_hpwl = hpwl
            write_final_placement(placed_macro, self.placement_save_dir)
        return hpwl


def main():
    
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', required=True)
    args = parser.parse_args()
    dataset = args.dataset
    random.seed(args.seed)
    placedb = PlaceDB(dataset)

    hpwl_save_dir = "result/BO/curve/"
    placement_save_dir = "result/BO/placement/"
    node_id_ls = rank_macros(placedb)

    if not os.path.exists(hpwl_save_dir):
        os.makedirs(hpwl_save_dir)
    if not os.path.exists(placement_save_dir):
        os.makedirs(placement_save_dir)

    hpwl_save_dir += "{}_seed_{}.csv".format(dataset, args.seed)
    placement_save_dir += "{}_seed_{}.csv".format(dataset, args.seed)

    hpwl_save_file = open(hpwl_save_dir,"a+")
    hpwl_writer = csv.writer(hpwl_save_file)

    grid_num = grid_setting[dataset]["grid_num"]
    grid_size = grid_setting[dataset]["grid_size"]

    macro_num = len(placedb.node_info.keys())


    f = placement_eval(dim=macro_num*2,grid_num=grid_num,grid_size=grid_size,placedb=placedb,node_id_ls=node_id_ls,csv_writer=hpwl_writer, csv_file=hpwl_save_file,placement_save_dir=placement_save_dir)

    turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = 10000,  # Maximum number of evaluations
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)
    #draw_placement(placedb, place_record, grid_num, grid_size)
    turbo1.optimize()

main()