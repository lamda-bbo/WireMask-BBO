import time
import numpy as np
import math
import random
import csv
from scipy.spatial import distance
from common import my_inf
import cv2
import heapq

def cal_hpwl(placed_macros, placedb):
    hpwl = 0
    net_hpwl = {}
    for net_id in placedb.net_info.keys():
        for node_id in placedb.net_info[net_id]["nodes"]:
            if node_id not in placed_macros.keys():
                continue
            pin_x = placed_macros[node_id]["center_loc_x"] + placedb.net_info[net_id]["nodes"][node_id]["x_offset"]
            pin_y = placed_macros[node_id]["center_loc_y"] + placedb.net_info[net_id]["nodes"][node_id]["y_offset"]
            if net_id not in net_hpwl.keys():
                net_hpwl[net_id] = {}
                net_hpwl[net_id] = {"x_max": pin_x, "x_min": pin_x, "y_max": pin_y, "y_min": pin_y}
            else:
                if net_hpwl[net_id]["x_max"] < pin_x:
                    net_hpwl[net_id]["x_max"] = pin_x
                elif net_hpwl[net_id]["x_min"] > pin_x:
                    net_hpwl[net_id]["x_min"] = pin_x
                if net_hpwl[net_id]["y_max"] < pin_y:
                    net_hpwl[net_id]["y_max"] = pin_y
                elif net_hpwl[net_id]["y_min"] > pin_y:
                    net_hpwl[net_id]["y_min"] = pin_y
    for net_id in net_hpwl.keys():
        hpwl += net_hpwl[net_id]["x_max"] - net_hpwl[net_id]["x_min"] + net_hpwl[net_id]["y_max"] - net_hpwl[net_id]["y_min"]
    return hpwl

def write_placement_and_overlap(placed_macros, net_hpwl, placedb, method, dataset):
    length = placedb.max_height + 500
    congestion = np.zeros((length,length))
    canvas = np.ones((length,length,3)) * 255
    margin = 5
    for node_id in placed_macros.keys():
        
        bottom_left_x = math.ceil(placed_macros[node_id]["center_loc_x"] - placedb.node_info[node_id]["x"] / 2)
        bottom_left_y = math.ceil(placed_macros[node_id]["center_loc_y"] - placedb.node_info[node_id]["y"] / 2)
        top_left_y = bottom_left_y + placedb.node_info[node_id]["y"]
        bottom_right_x = bottom_left_x + placedb.node_info[node_id]["x"]
        canvas[bottom_left_x:bottom_right_x, bottom_left_y:top_left_y] = [135,206,250]

        canvas[bottom_left_x:bottom_right_x, bottom_left_y:bottom_left_y + margin] = [0,0,0]
        canvas[bottom_left_x:bottom_right_x, top_left_y - margin:top_left_y] = [0,0,0]
        canvas[bottom_left_x:bottom_left_x+margin, bottom_left_y:top_left_y] = [0,0,0]
        canvas[bottom_right_x-margin:bottom_right_x, bottom_left_y:top_left_y] = [0,0,0]

    for net_id in net_hpwl.keys():
        x_max = math.ceil(net_hpwl[net_id]["x_max"])
        x_min = math.ceil(net_hpwl[net_id]["x_min"])
        y_max = math.ceil(net_hpwl[net_id]["y_max"])
        y_min = math.ceil(net_hpwl[net_id]["y_min"])
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        if delta_x == 0 or delta_y == 0:
            continue
        congestion[x_min:x_max, y_min:y_max] += 1/delta_x + 1/delta_y

    g = canvas[:,:,1] == 255
    extra_count = length**2 - placedb.max_height**2
    blank_count = len(canvas[g]) - extra_count
    all_count = placedb.max_height**2
    occupied_count = all_count - blank_count
    macro_all_count = 0
    for node_id in list(placedb.node_info.keys()):
        macro_all_count += placedb.node_info[node_id]["x"] * placedb.node_info[node_id]["y"]
    overlap_count = macro_all_count - occupied_count
    overlap_ratio = overlap_count / macro_all_count
    macro_util_ratio = blank_count/all_count
    congestion_list = congestion.reshape(1,-1).tolist()[0]
    congestion_mean = np.mean(heapq.nlargest(math.ceil(len(congestion_list)/10),congestion_list))
    print("overlap_ratio: ", round(overlap_ratio,2), "congestion: ", round(congestion_mean*100,2), "macro_util_ratio: ", round(macro_util_ratio,2))
    
    cv2.imwrite('placement_visualization/{}_{}.pdf'.format(method, dataset),canvas)
    return congestion_mean

def read_ea_results(dir, budget):
    hpwl_ls = []
    time_ls = []
    with open(dir) as f:
        for row in csv.reader(f):
            hpwl = eval(row[0])
            time = eval(row[1])
            hpwl_ls.append(hpwl)
            time_ls.append(time)
    start_time = time_ls[0]
    time_ls = [m - start_time for m in time_ls]
    time_ls = [m / 60 for m in time_ls]
    hpwl_ls = [m / 1e5 for m in hpwl_ls]
    hpwl_ls_min = []
    min_hpwl = 1e6

    if hpwl_ls[0] > 1e6:
        for item in hpwl_ls:
            if item < 1e6:
                hpwl_ls[0] = item
                break

    for item in hpwl_ls:
        if item < min_hpwl:
            min_hpwl = item
        hpwl_ls_min.append(min_hpwl)
    for i in range(len(time_ls)):
        if time_ls[i] > budget:
            time_ls = time_ls[:i]
            hpwl_ls = hpwl_ls[:i]
            hpwl_ls_min = hpwl_ls_min[:i]
            break
    return time_ls, hpwl_ls, hpwl_ls_min

def read_BO_results(dir, budget):
    hpwl_ls = []
    time_ls = []
    with open(dir) as f:
        for row in csv.reader(f):
            hpwl = eval(row[1])
            time = eval(row[0])
            hpwl_ls.append(hpwl)
            time_ls.append(time)
    start_time = time_ls[0]
    time_ls = [m - start_time for m in time_ls]
    time_ls = [m / 60 for m in time_ls]
    hpwl_ls = [m / 1e5 for m in hpwl_ls]
    hpwl_ls_min = []
    min_hpwl = 1e6
    for item in hpwl_ls:
        if item < min_hpwl:
            min_hpwl = item
        hpwl_ls_min.append(min_hpwl)
    for i in range(len(time_ls)):
        if time_ls[i] > budget:
            time_ls = time_ls[:i]
            hpwl_ls = hpwl_ls[:i]
            hpwl_ls_min = hpwl_ls_min[:i]
            break
    return time_ls, hpwl_ls, hpwl_ls_min


def read_mask_placement(placement_dir, placedb, grid_size):
    place_record = {}
    with open(placement_dir) as f:
        for row in f:
            row = row.split("\t")
            node_id = row[0]
            if node_id not in list(placedb.node_info.keys()):
                continue
            loc_x = eval(row[1]) - placedb.node_info[node_id]["x"] / 2
            loc_y = eval(row[2]) - placedb.node_info[node_id]["y"] / 2
            place_record[node_id] = {}
            place_record[node_id]["loc_x"] = loc_x / grid_size
            place_record[node_id]["loc_y"] = loc_y / grid_size
    return place_record

def rank_macros(placedb):# 将macro按照固定顺序（net面积总和）从大到小排列，输出排序后的macro序列。

    node_id_ls = list(placedb.node_info.keys()).copy()
    for node_id in node_id_ls:
        placedb.node_info[node_id]["area"] = placedb.node_info[node_id]["x"] * placedb.node_info[node_id]["y"]
        
    net_id_ls = list(placedb.net_info.keys()).copy()
    for net_id in net_id_ls:
        sum = 0
        #print(placedb.net_info[net_id]["nodes"])
        for node_id in placedb.net_info[net_id]["nodes"].keys():
            sum += placedb.node_info[node_id]["area"]
        placedb.net_info[net_id]["area"] = sum
        #print(placedb.net_info[net_id]["area"])
    for node_id in node_id_ls:
        placedb.node_info[node_id]["area_sum"] = 0
        for net_id in net_id_ls:
            if node_id in placedb.net_info[net_id]["nodes"].keys():
                placedb.node_info[node_id]["area_sum"] += placedb.net_info[net_id]["area"]
    node_id_ls.sort(key = lambda x: placedb.node_info[x]["area_sum"], reverse = True)
    return node_id_ls


def write_final_placement(best_placed_macro, dir):
    csv_file2 = open(dir,"a+")
    csv_writer2 = csv.writer(csv_file2)
    csv_writer2.writerow([time.time()])
    for node_id in list(best_placed_macro.keys()):
        csv_writer2.writerow([node_id, best_placed_macro[node_id]["bottom_left_x"], best_placed_macro[node_id]["bottom_left_y"]])
    csv_writer2.writerow([])
    csv_file2.close()

def random_guiding(node_id_ls, placedb, grid_num, grid_size):# 将所有macro随机放置

    placed_macros = {}
    N2_time = 0
    placed_macros = {}

    for node_id in node_id_ls:
        x = placedb.node_info[node_id]["x"]
        y = placedb.node_info[node_id]["y"]
        scaled_x = math.ceil(x / grid_size)
        scaled_y = math.ceil(y / grid_size)
        placedb.node_info[node_id]["scaled_x"] = scaled_x
        placedb.node_info[node_id]["scaled_y"] = scaled_y

        position_mask = np.ones((grid_num,grid_num))

        loc_x_ls = np.where(position_mask==1)[0]
        loc_y_ls = np.where(position_mask==1)[1]
        placed_macros[node_id] = {}

        time0 = time.time()

        #print(np.where(wire_mask == min_ele)[0][0],np.where(wire_mask == min_ele)[1][0])
        idx = random.choice(range(len(loc_x_ls)))

        chosen_loc_x = loc_x_ls[idx]
        chosen_loc_y = loc_y_ls[idx]

        N2_time += time.time() - time0
        
        center_loc_x = grid_size * chosen_loc_x + 0.5 * x
        center_loc_y = grid_size * chosen_loc_y + 0.5 * y

        placed_macros[node_id] = {"scaled_x": scaled_x, "scaled_y": scaled_y, "loc_x": chosen_loc_x, "loc_y": chosen_loc_y, "x": x, "y": y, "center_loc_x": center_loc_x, "center_loc_y": center_loc_y, 'bottom_left_x': chosen_loc_x * grid_size, "bottom_left_y": chosen_loc_y * grid_size}

    return placed_macros


def greedy_placer_with_init_coordinate(node_id_ls, placedb, grid_num, grid_size, place_record):
    shuffle = 0
    placed_macros = {}
    #placed_macros[node_id_ls[0]] = place_record[node_id_ls[0]]
    #node_id_ls = node_id_ls[1:]
    hpwl_info_for_each_net = {}
    hpwl = 0

    time_start = time.time()
    N2_time = 0
    for node_id in node_id_ls:
        
        x = placedb.node_info[node_id]["x"]
        y = placedb.node_info[node_id]["y"]
        scaled_x = math.ceil(x / grid_size)
        scaled_y = math.ceil(y / grid_size)
        placedb.node_info[node_id]["scaled_x"] = scaled_x
        placedb.node_info[node_id]["scaled_y"] = scaled_y
        position_mask = np.ones((grid_num,grid_num)) * my_inf
        position_mask[:grid_num - scaled_x,:grid_num - scaled_y] = 1
        wire_mask = np.ones((grid_num,grid_num)) * 0.1

        for key1 in placed_macros.keys():

            bottom_left_x = max(0, placed_macros[key1]["loc_x"] - scaled_x + 1)
            bottom_left_y = max(0, placed_macros[key1]["loc_y"] - scaled_y + 1)
            top_right_x = min(grid_num - 1, placed_macros[key1]["loc_x"] + placed_macros[key1]["scaled_x"])
            top_right_y = min(grid_num - 1, placed_macros[key1]["loc_y"] + placed_macros[key1]["scaled_y"])

            position_mask[bottom_left_x:top_right_x,bottom_left_y:top_right_y] = my_inf
        
        loc_x_ls = np.where(position_mask==1)[0]
        loc_y_ls = np.where(position_mask==1)[1]
        placed_macros[node_id] = {}
        net_ls = {}

        for net_id in placedb.net_info.keys():
            if node_id in placedb.net_info[net_id]["nodes"].keys():
                net_ls[net_id] = {}
                net_ls[net_id] = placedb.net_info[net_id]

        if len(loc_x_ls) == 0:
            print("no_legal_place")
            return [], my_inf
        
        time0 = time.time()
        for net_id in net_ls.keys():
            if net_id in hpwl_info_for_each_net.keys():
                x_offset = net_ls[net_id]["nodes"][node_id]["x_offset"] + 0.5 * x
                y_offset = net_ls[net_id]["nodes"][node_id]["y_offset"] + 0.5 * y
                for col in range(grid_num):

                    x_co = col * grid_size + x_offset
                    y_co = col * grid_size + y_offset

                    if x_co < hpwl_info_for_each_net[net_id]["x_min"]:
                        wire_mask[col,:] += hpwl_info_for_each_net[net_id]["x_min"] - x_co
                    elif x_co > hpwl_info_for_each_net[net_id]["x_max"]:
                        wire_mask[col,:] += x_co - hpwl_info_for_each_net[net_id]["x_max"]
                    if y_co < hpwl_info_for_each_net[net_id]["y_min"]:
                        wire_mask[:,col] += hpwl_info_for_each_net[net_id]["y_min"] - y_co
                    elif y_co > hpwl_info_for_each_net[net_id]["y_max"]:
                        wire_mask[:,col] += y_co - hpwl_info_for_each_net[net_id]["y_max"]
        wire_mask = np.multiply(wire_mask, position_mask)
        min_ele = np.min(wire_mask)
        #print(np.where(wire_mask == min_ele)[0][0],np.where(wire_mask == min_ele)[1][0])
        
        chosen_loc_x = list(np.where(wire_mask == min_ele)[0])
        chosen_loc_y = list(np.where(wire_mask == min_ele)[1])
        chosen_coor = list(zip(chosen_loc_x, chosen_loc_y))
        
        tup_order = []
        for tup in chosen_coor:
            tup_order.append(distance.euclidean(tup, (place_record[node_id]["loc_x"],place_record[node_id]["loc_y"])))
        chosen_coor = list(zip(chosen_coor, tup_order))

        chosen_coor.sort(key = lambda x: x[1])

        chosen_loc_x = chosen_coor[0][0][0]
        chosen_loc_y = chosen_coor[0][0][1]
        #if node_id == node_id_ls[0]:
        #    print(wire_mask, min_ele)
        #    print(place_record[node_id]["loc_x"],place_record[node_id]["loc_y"])
        #    print(chosen_loc_x,chosen_loc_y)
        #print(chosen_loc_x,chosen_loc_y) 
        '''
        idx = 0
        if shuffle:
            idx = random.choice(list(range(len(np.where(wire_mask == min_ele)[0]))))
        chosen_loc_x = np.where(wire_mask == min_ele)[0][idx]
        chosen_loc_y = np.where(wire_mask == min_ele)[1][idx]
        '''
        best_hpwl = min_ele

        N2_time += time.time() - time0
        
        center_loc_x = grid_size * chosen_loc_x + 0.5 * x
        center_loc_y = grid_size * chosen_loc_y + 0.5 * y
        for net_id in net_ls.keys():
            x_offset = net_ls[net_id]["nodes"][node_id]["x_offset"]
            y_offset = net_ls[net_id]["nodes"][node_id]["y_offset"]
            if net_id not in hpwl_info_for_each_net.keys():
                hpwl_info_for_each_net[net_id] = {}
                hpwl_info_for_each_net[net_id] = {"x_max": center_loc_x + x_offset, "x_min": center_loc_x + x_offset, "y_max": center_loc_y + y_offset, "y_min": center_loc_y + y_offset}
            else:
                if hpwl_info_for_each_net[net_id]["x_max"] < center_loc_x + x_offset:
                    hpwl_info_for_each_net[net_id]["x_max"] = center_loc_x + x_offset
                elif hpwl_info_for_each_net[net_id]["x_min"] > center_loc_x + x_offset:
                    hpwl_info_for_each_net[net_id]["x_min"] = center_loc_x + x_offset
                if hpwl_info_for_each_net[net_id]["y_max"] < center_loc_y + y_offset:
                    hpwl_info_for_each_net[net_id]["y_max"] = center_loc_y + y_offset
                elif hpwl_info_for_each_net[net_id]["y_min"] > center_loc_y + y_offset:
                    hpwl_info_for_each_net[net_id]["y_min"] = center_loc_y + y_offset

        hpwl += best_hpwl
        placed_macros[node_id] = {"scaled_x": scaled_x, "scaled_y": scaled_y, "loc_x": chosen_loc_x, "loc_y": chosen_loc_y, "x": x, "y": y, "center_loc_x": center_loc_x, "center_loc_y": center_loc_y, 'bottom_left_x': chosen_loc_x * grid_size + 452, "bottom_left_y": chosen_loc_y * grid_size + 452}

    time_end = time.time()
    print("verified hpwl: ", cal_hpwl(placed_macros, placedb))
    print("time:", time_end - time_start)
    print("N2_time:", N2_time)
    print("hpwl:", hpwl)
    print("shuffle or not: ", shuffle)
    return placed_macros, hpwl    

def greedy_local_search(queue, placedb, placed_macros, grid_size, grid_num):
    delta_hpwl = 0
    random.shuffle(queue)
    for key in queue:
        if key not in placed_macros.keys():
            continue
        position_mask = np.zeros((grid_num,grid_num),bool)
        x = placedb.node_info[key]["x"]
        y = placedb.node_info[key]["y"]
        scaled_x = math.ceil(x / grid_size)
        scaled_y = math.ceil(y / grid_size)
        position_mask[:grid_num - scaled_x,:grid_num - scaled_y] = 1

        for key1 in placed_macros.keys():
            bottom_left_x = max(0, int(placed_macros[key1]["loc_x"] - placed_macros[key1]["scaled_x"] + 1))
            bottom_left_y = max(0, int(placed_macros[key1]["loc_y"] - placed_macros[key1]["scaled_y"] + 1))
            top_right_x = min(grid_num - 1, int(placed_macros[key1]["loc_x"] + placed_macros[key1]["scaled_x"]))
            top_right_y = min(grid_num - 1, int(placed_macros[key1]["loc_y"] + placed_macros[key1]["scaled_y"]))
            
            position_mask[bottom_left_x:top_right_x,bottom_left_y:top_right_y] = 0
        loc_x_ls = np.where(position_mask==1)[0]
        loc_y_ls = np.where(position_mask==1)[1]
        if len(loc_x_ls) == 0:
            #print("macro{} have no other place to place".format(key))
            continue

        net_ls = {}
        net_hpwl = {}
        for net_id in placedb.net_info.keys():
            if key in placedb.net_info[net_id]["nodes"].keys():
                net_ls[net_id] = {}
                net_ls[net_id] = placedb.net_info[net_id]

        for net_id in net_ls.keys():
            for node_id in net_ls[net_id]["nodes"].keys():
                if node_id == key or node_id not in placed_macros.keys():
                    continue
                else:
                    pin_loc_x = placed_macros[node_id]["center_loc_x"] + net_ls[net_id]["nodes"][node_id]["x_offset"]
                    pin_loc_y = placed_macros[node_id]["center_loc_y"] + net_ls[net_id]["nodes"][node_id]["y_offset"]
                    if net_id not in net_hpwl.keys():
                        net_hpwl[net_id] = {}
                        net_hpwl[net_id] = {"x_max": pin_loc_x, "x_min": pin_loc_x, "y_max": pin_loc_y, "y_min": pin_loc_y}
                    else:
                        if net_hpwl[net_id]["x_max"] < pin_loc_x:
                            net_hpwl[net_id]["x_max"] = pin_loc_x
                        elif net_hpwl[net_id]["x_min"] > pin_loc_x:
                            net_hpwl[net_id]["x_min"] = pin_loc_x
                        if net_hpwl[net_id]["y_max"] < pin_loc_y:
                            net_hpwl[net_id]["y_max"] = pin_loc_y
                        elif net_hpwl[net_id]["y_min"] > pin_loc_y:
                            net_hpwl[net_id]["y_min"] = pin_loc_y

        baseline_hpwl = 0
        for net_id in net_ls.keys():
            pin_loc_x = placed_macros[key]["center_loc_x"] + net_ls[net_id]["nodes"][key]["x_offset"]
            pin_loc_y = placed_macros[key]["center_loc_y"] + net_ls[net_id]["nodes"][key]["y_offset"]
            if net_id not in net_hpwl.keys():
                continue
            if net_hpwl[net_id]["x_max"] < pin_loc_x:
                baseline_hpwl += pin_loc_x - net_hpwl[net_id]["x_max"]
            elif net_hpwl[net_id]["x_min"] > pin_loc_x:
                baseline_hpwl += net_hpwl[net_id]["x_min"] - pin_loc_x
            if net_hpwl[net_id]["y_max"] < pin_loc_y:
                baseline_hpwl += pin_loc_y - net_hpwl[net_id]["y_max"]
            elif net_hpwl[net_id]["y_min"] > pin_loc_y:
                baseline_hpwl += net_hpwl[net_id]["y_min"] - pin_loc_y
        #print("baseline: ", baseline_hpwl)
        chosen_loc_x = loc_x_ls[0]
        chosen_loc_y = loc_y_ls[0]
        s = list(range(0, len(loc_x_ls)))
        best_hpwl = baseline_hpwl
        for j in s:

            loc_x = loc_x_ls[j]
            loc_y = loc_y_ls[j]
            center_loc_x = grid_size * loc_x + 0.5 * x
            center_loc_y = grid_size * loc_y + 0.5 * y
            tmp_hpwl = 0
            
            for net_id in net_ls.keys():
                x_offset = net_ls[net_id]["nodes"][key]["x_offset"]
                y_offset = net_ls[net_id]["nodes"][key]["y_offset"]
                if net_id not in net_hpwl.keys():
                    continue
                if net_hpwl[net_id]["x_max"] < center_loc_x + x_offset:
                    tmp_hpwl +=  center_loc_x + x_offset - net_hpwl[net_id]["x_max"]
                elif net_hpwl[net_id]["x_min"] > center_loc_x + x_offset:
                    tmp_hpwl +=  net_hpwl[net_id]["x_min"] - (center_loc_x + x_offset)
                if net_hpwl[net_id]["y_max"] < center_loc_y + y_offset:
                    tmp_hpwl +=  center_loc_y + y_offset - net_hpwl[net_id]["y_max"]
                elif net_hpwl[net_id]["y_min"] > center_loc_y + y_offset:
                    tmp_hpwl +=  net_hpwl[net_id]["y_min"] - (center_loc_y + y_offset)

            if tmp_hpwl < best_hpwl:
                best_hpwl = tmp_hpwl
                chosen_loc_x = loc_x
                chosen_loc_y = loc_y
                chosen_center_loc_x = grid_size * loc_x + 0.5 * x
                chosen_center_loc_y = grid_size * loc_y + 0.5 * y
                #print(center_loc_x, center_loc_y)

        if best_hpwl < baseline_hpwl:

            delta_hpwl += (best_hpwl - baseline_hpwl)
            placed_macros[key] = {"scaled_x": scaled_x, "scaled_y": scaled_y, "loc_x": chosen_loc_x, "loc_y": chosen_loc_y, "x": x, "y": y, "center_loc_x": chosen_center_loc_x, "center_loc_y": chosen_center_loc_y, 'bottom_left_x': chosen_loc_x * grid_size, "bottom_left_y": chosen_loc_y * grid_size}

    verified_hpwl = cal_hpwl(placed_macros, placedb)
    # print("delta hpwl: ", delta_hpwl)
    # print("verified hpwl: ", verified_hpwl)
    return placed_macros, verified_hpwl

def bo_placer(node_id_ls, placedb, grid_num, grid_size, place_record, csv_writer, csv_file):
    placed_macros = {}
    #placed_macros[node_id_ls[0]] = place_record[node_id_ls[0]]
    #node_id_ls = node_id_ls[1:]
    hpwl_info_for_each_net = {}
    hpwl = 0

    N2_time = 0
    final_placement = {}
    for node_id in node_id_ls:
        
        x = placedb.node_info[node_id]["x"]
        y = placedb.node_info[node_id]["y"]
        scaled_x = math.ceil(x / grid_size)
        scaled_y = math.ceil(y / grid_size)
        placedb.node_info[node_id]["scaled_x"] = scaled_x
        placedb.node_info[node_id]["scaled_y"] = scaled_y
        position_mask = np.ones((grid_num,grid_num)) * my_inf
        position_mask[:grid_num - scaled_x,:grid_num - scaled_y] = 1
        wire_mask = np.ones((grid_num,grid_num)) * 0.1

        for key1 in placed_macros.keys():

            bottom_left_x = max(0, placed_macros[key1]["loc_x"] - scaled_x + 1)
            bottom_left_y = max(0, placed_macros[key1]["loc_y"] - scaled_y + 1)
            top_right_x = min(grid_num - 1, placed_macros[key1]["loc_x"] + placed_macros[key1]["scaled_x"])
            top_right_y = min(grid_num - 1, placed_macros[key1]["loc_y"] + placed_macros[key1]["scaled_y"])

            position_mask[bottom_left_x:top_right_x,bottom_left_y:top_right_y] = my_inf
        
        loc_x_ls = np.where(position_mask==1)[0]
        loc_y_ls = np.where(position_mask==1)[1]
        placed_macros[node_id] = {}
        net_ls = {}

        for net_id in placedb.net_info.keys():
            if node_id in placedb.net_info[net_id]["nodes"].keys():
                net_ls[net_id] = {}
                net_ls[net_id] = placedb.net_info[net_id]

        if len(loc_x_ls) == 0:
            print("no_legal_place")
            return [], my_inf
        
        time0 = time.time()
        for net_id in net_ls.keys():
            if net_id in hpwl_info_for_each_net.keys():
                x_offset = net_ls[net_id]["nodes"][node_id]["x_offset"] + 0.5 * x
                y_offset = net_ls[net_id]["nodes"][node_id]["y_offset"] + 0.5 * y
                for col in range(grid_num):

                    x_co = col * grid_size + x_offset
                    y_co = col * grid_size + y_offset

                    if x_co < hpwl_info_for_each_net[net_id]["x_min"]:
                        wire_mask[col,:] += hpwl_info_for_each_net[net_id]["x_min"] - x_co
                    elif x_co > hpwl_info_for_each_net[net_id]["x_max"]:
                        wire_mask[col,:] += x_co - hpwl_info_for_each_net[net_id]["x_max"]
                    if y_co < hpwl_info_for_each_net[net_id]["y_min"]:
                        wire_mask[:,col] += hpwl_info_for_each_net[net_id]["y_min"] - y_co
                    elif y_co > hpwl_info_for_each_net[net_id]["y_max"]:
                        wire_mask[:,col] += y_co - hpwl_info_for_each_net[net_id]["y_max"]
        wire_mask = np.multiply(wire_mask, position_mask)
        min_ele = np.min(wire_mask)
        #print(np.where(wire_mask == min_ele)[0][0],np.where(wire_mask == min_ele)[1][0])
        
        chosen_loc_x = list(np.where(wire_mask == min_ele)[0])
        chosen_loc_y = list(np.where(wire_mask == min_ele)[1])
        chosen_coor = list(zip(chosen_loc_x, chosen_loc_y))
        
        tup_order = []
        for tup in chosen_coor:
            tup_order.append(distance.euclidean(tup, (place_record[node_id]["loc_x"],place_record[node_id]["loc_y"])))
        chosen_coor = list(zip(chosen_coor, tup_order))

        chosen_coor.sort(key = lambda x: x[1])

        chosen_loc_x = chosen_coor[0][0][0]
        chosen_loc_y = chosen_coor[0][0][1]
        
        best_hpwl = min_ele

        N2_time += time.time() - time0
        
        center_loc_x = grid_size * chosen_loc_x + 0.5 * x
        center_loc_y = grid_size * chosen_loc_y + 0.5 * y
        for net_id in net_ls.keys():
            x_offset = net_ls[net_id]["nodes"][node_id]["x_offset"]
            y_offset = net_ls[net_id]["nodes"][node_id]["y_offset"]
            if net_id not in hpwl_info_for_each_net.keys():
                hpwl_info_for_each_net[net_id] = {}
                hpwl_info_for_each_net[net_id] = {"x_max": center_loc_x + x_offset, "x_min": center_loc_x + x_offset, "y_max": center_loc_y + y_offset, "y_min": center_loc_y + y_offset}
            else:
                if hpwl_info_for_each_net[net_id]["x_max"] < center_loc_x + x_offset:
                    hpwl_info_for_each_net[net_id]["x_max"] = center_loc_x + x_offset
                elif hpwl_info_for_each_net[net_id]["x_min"] > center_loc_x + x_offset:
                    hpwl_info_for_each_net[net_id]["x_min"] = center_loc_x + x_offset
                if hpwl_info_for_each_net[net_id]["y_max"] < center_loc_y + y_offset:
                    hpwl_info_for_each_net[net_id]["y_max"] = center_loc_y + y_offset
                elif hpwl_info_for_each_net[net_id]["y_min"] > center_loc_y + y_offset:
                    hpwl_info_for_each_net[net_id]["y_min"] = center_loc_y + y_offset

        hpwl += best_hpwl
        placed_macros[node_id] = {"scaled_x": scaled_x, "scaled_y": scaled_y, "loc_x": chosen_loc_x, "loc_y": chosen_loc_y, "x": x, "y": y, "center_loc_x": center_loc_x, "center_loc_y": center_loc_y, 'bottom_left_x': chosen_loc_x * grid_size + 452, "bottom_left_y": chosen_loc_y * grid_size + 452}
        final_placement[node_id] = {}
        final_placement[node_id]["loc_x"] = chosen_loc_x
        final_placement[node_id]["loc_y"] = chosen_loc_y
    print("hpwl:", hpwl)
    csv_writer.writerow([time.time(), hpwl])
    csv_file.flush()
    return placed_macros, hpwl    