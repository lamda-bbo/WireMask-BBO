from place_db import PlaceDB
from utils import greedy_placer_with_init_coordinate, write_final_placement, rank_macros, read_mask_placement
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
    
    hpwl_save_dir = "result/finetune_maskplace/curve/"
    placement_save_dir = "result/finetune_maskplace/placement/"
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

    place_record = read_mask_placement("/home/shiyq/results_and_eval/our/mask/placement/{}.pl".format(dataset), placedb, grid_size)
    placed_macros, hpwl = greedy_placer_with_init_coordinate(node_id_ls, placedb, grid_num, grid_size, place_record)
    
    for _ in range(stop_round):
        node_id_ls = list(place_record.keys())
        node_a, node_b = random.sample(node_id_ls, 2)
        node_a_loc_x, node_a_loc_y = place_record[node_a]["loc_x"], place_record[node_a]["loc_y"]
        node_b_loc_x, node_b_loc_y = place_record[node_b]["loc_x"], place_record[node_b]["loc_y"]
        place_record[node_a]["loc_x"], place_record[node_a]["loc_y"] = node_b_loc_x, node_b_loc_y
        place_record[node_b]["loc_x"], place_record[node_b]["loc_y"] = node_a_loc_x, node_a_loc_y
        placed_macro, hpwl = greedy_placer_with_init_coordinate(node_id_ls, placedb, grid_num, grid_size, place_record)
        if hpwl >= best_hpwl:
            node_a_loc_x, node_a_loc_y = place_record[node_a]["loc_x"], place_record[node_a]["loc_y"]
            node_b_loc_x, node_b_loc_y = place_record[node_b]["loc_x"], place_record[node_b]["loc_y"]
            place_record[node_a]["loc_x"], place_record[node_a]["loc_y"] = node_b_loc_x, node_b_loc_y
            place_record[node_b]["loc_x"], place_record[node_b]["loc_y"] = node_a_loc_x, node_a_loc_y
        else:
            best_hpwl = hpwl
            best_placed_macro = placed_macro
            write_final_placement(best_placed_macro, placement_save_dir)

        hpwl_writer.writerow([hpwl, time.time()])
        hpwl_save_file.flush()

main()
