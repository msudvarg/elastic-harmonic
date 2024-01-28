#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.patheffects as pe
import numpy as np
import time
from math import *
import copy
import statistics
from pyquaternion import Quaternion
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D
import csv


def save_to_csv(data_list, path_to_save):
    with open(path_to_save, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for data in data_list:
            wr.writerow([data])


def relative_translation(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]


def rotation_error(r1, r2):
    diff_r = Quaternion(r1.elements - r2.elements)
    return diff_r.degrees

def translation_error(p1, p2):
    return sqrt(pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1],2) + pow(p1[2] - p2[2],2))

def traj_translation_error(trj1, gt_traj):
    gt_index = 0
    errors = []
    for p in trj1:
        while gt_index < len(gt_traj) and p.timestamp > gt_traj[gt_index].timestamp:
            gt_index = gt_index + 1
        
        if gt_index < len(gt_traj):
            # print('P timestamp: ', p.timestamp, '  GT timestamp: ', gt_traj[gt_index].timestamp)
            # print('Time diff : ', p.timestamp - gt_traj[gt_index].timestamp)
            errors.append(translation_error(p.position, gt_traj[gt_index].position))

    return errors

def traj_relative_translation_error(trj1, gt_traj):
    gt_index = 0
    gt_last_index = 0
    errors = []

    last_p = None

    for p in trj1:
        while gt_index < len(gt_traj) and p.timestamp >= gt_traj[gt_index].timestamp:
            gt_index = gt_index + 1
        
        if gt_index < len(gt_traj):
            # print('P timestamp: ', p.timestamp, '  GT timestamp: ', gt_traj[gt_index].timestamp)
            # print('Time diff : ', p.timestamp - gt_traj[gt_index].timestamp)
            if last_p != None:
                relative_trans = relative_translation(p.position,last_p.position)
                relative_trans_gt = relative_translation(gt_traj[gt_index].position, gt_traj[gt_last_index].position)

                errors.append(translation_error(relative_trans, relative_trans_gt))

            last_p = p
            gt_last_index = gt_index

    return errors

def traj_relative_orientation_error(trj1, gt_traj):
    gt_index = 0
    gt_last_index = 0
    errors = []

    last_p = None

    for p in trj1:
        while gt_index < len(gt_traj) and p.timestamp >= gt_traj[gt_index].timestamp:
            gt_index = gt_index + 1
        
        if gt_index < len(gt_traj):
            # print('P timestamp: ', p.timestamp, '  GT timestamp: ', gt_traj[gt_index].timestamp)
            # print('Time diff : ', p.timestamp - gt_traj[gt_index].timestamp)
            if last_p != None:
                # relative_orient = rotation_error(p.orientation,last_p.orientation)
                # relative_orient_gt = rotation_error(gt_traj[gt_index].orientation, gt_traj[gt_last_index].orientation)

                e = rotation_error(p.orientation, gt_traj[gt_index].orientation)

                # print("relatvie orientation : ", e)
                # print("relatvie orientation : ", relative_orient_gt)
                # errors.append(abs(relative_orient - relative_orient_gt))

            last_p = p
            gt_last_index = gt_index

    return errors

def translation(p, t):
    return np.array([p[0]-t[0], p[1]-t[1], p[2]-t[2]])

def distance(p1, p2):
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def error_log(errors):
    import statistics
    # print("    Means        Max        min")
    print('Mean : ', statistics.mean(errors), '    std dev  :  ',  statistics.stdev(errors), '    Max  :  ',  max(errors), '    Min  :  ',  min(errors))


class loc_point:
    def __init__(self):
        self.timestamp = 0
        self.position = None
        self.orientation = None


colors = ['seagreen', 'purple', 'yellow', 'lightcoral', 'salmon', 'blue', 'olive', 'crimson', 'tan', 'salmon', 'r', 'r', 'r']

#######################
###Load Ground Truth###
#######################
gt_path = 'groudtruth/v101.csv'

offset_flag = True
offset_p = [] #np.array()
offset_r = Quaternion()
offset_point = None
conj_offset_r = Quaternion()
ground_truth_traj = []

gtxs = []
gtys = []
gtzs = []

with open(gt_path) as f:
    lines = f.readlines()
    for line in lines:

        # This is the first line
        if line.split(',')[0] == '#timestamp':
            continue

        content = [float(x) for x in line.split(',')]

        new_p = loc_point()

        new_p.timestamp = content[0] / (1.0 * 10e8)

        new_p.position = np.array([content[1], content[2], content[3]])
        new_p.orientation = Quaternion(content[4], content[5], content[6], content[7])

        if offset_flag:
            offset_point = copy.copy(new_p)
            conj_offset_r = offset_point.orientation.conjugate
            offset_flag = False
        
        #### Eliminate the initial offset
        new_p.position = translation(new_p.position, offset_point.position)
        new_p.position = conj_offset_r.rotate(new_p.position)

        new_p.orientation = Quaternion(new_p.orientation.elements - offset_point.orientation.elements)

        ### coordination transform
        new_p.position = np.array([new_p.position[1], -new_p.position[0], new_p.position[2]])

        ground_truth_traj.append(new_p)

        gtxs.append(new_p.position[0])
        gtys.append(new_p.position[1])
        gtzs.append(new_p.position[2])

###############################
###Load Experimental Results###
###############################


import os

folder_path = './elasticity_data_slowdow/ba_1_slowdown/Trace7'
txt_name = "/FrameTrajectory_TUM_Format.txt"

for root, dirs, files in os.walk(folder_path):
    for dir in dirs:
        # print(os.path.join(root, dir))
        full_path = os.path.join(root, dir) + txt_name

        ### Load the data
        baseline_xs = []
        baseline_ys = []
        baseline_zs = []
        baseline_traj = []

        start_time = 0
        with open(full_path) as f:
            lines = f.readlines()
            for line in lines:
                content = [float(x) for x in line.split()]

                new_p = loc_point()
                new_p.timestamp = content[0]

                if start_time == 0:
                    start_time = content[0]

                new_p.position = np.array([content[1], content[2], content[3]])
                new_p.orientation = Quaternion(content[4], content[5], content[6], content[7])

                baseline_traj.append(new_p)

                baseline_xs.append(new_p.position[0])
                baseline_ys.append(new_p.position[1])
                baseline_zs.append(new_p.position[2])

        baseline_rpe_error = traj_relative_translation_error(baseline_traj, ground_truth_traj)
        if len(baseline_rpe_error) == 0:
            print(dir ," : - ")
        else:
            print(dir ," : ", sum(baseline_rpe_error)/len(baseline_rpe_error), " with length : ", len(baseline_rpe_error))

        plt.plot(baseline_xs,baseline_ys)
        plt.plot(gtxs,gtys)
        plt.show()
exit(0)