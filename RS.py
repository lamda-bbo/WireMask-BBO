from place_db import PlaceDB
from utils import random_guiding, greedy_placer_with_init_coordinate, write_final_placement, rank_macros
from common import grid_setting, my_inf
import random
import argparse
import time
import csv
import os

def main():
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--stop_round', default=my_inf)
    args = parser.parse_args()
    dataset = args.dataset
    seed1 = args.seed
    stop_round = int(args.stop_round)
    random.seed(seed1)
    placedb = PlaceDB(dataset)
    
    hpwl_save_dir = "result/Random/curve/"
    placement_save_dir = "result/Random/placement/"
    node_id_ls = rank_macros(placedb)

    if not os.path.exists(hpwl_save_dir):
        os.makedirs(hpwl_save_dir)
    if not os.path.exists(placement_save_dir):
        os.makedirs(placement_save_dir)

    hpwl_save_dir += "{}_seed_{}.csv".format(dataset, seed1)
    placement_save_dir += "{}_seed_{}.csv".format(dataset, seed1)
    

    hpwl_save_file = open(hpwl_save_dir,"a+")
    hpwl_writer = csv.writer(hpwl_save_file)

    grid_num = grid_setting[dataset]["grid_num"]
    grid_size = grid_setting[dataset]["grid_size"]

    best_hpwl = my_inf
    for _ in range(stop_round):
        print("init")
        place_record = random_guiding(node_id_ls, placedb, grid_num, grid_size)
        placed_macros, hpwl = greedy_placer_with_init_coordinate(node_id_ls, placedb, grid_num, grid_size, place_record)
        if hpwl < best_hpwl:
            best_hpwl = hpwl
            best_placed_macro = placed_macros
            write_final_placement(best_placed_macro, placement_save_dir)
        hpwl_writer.writerow([hpwl, time.time(),"init"])
        hpwl_save_file.flush()

main()
