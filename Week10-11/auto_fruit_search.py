# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

#import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('M4_prac_shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically

def drive_to_point(waypoint, robot_pose, ppi):
    # imports camera / wheel calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',') # meters/tick
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',') # meters

    ####################################################

    waypoint_x = waypoint[0]
    waypoint_y = waypoint[1]
    robot_x = robot_pose[0]
    robot_y = robot_pose[1]
    robot_theta = robot_pose[2]
    waypoint_angle = np.arctan2((waypoint_y-robot_y),(waypoint_x-robot_x))

    print(f'Waypoint angle {waypoint_angle} and robot angle {robot_theta}')

    #Calculate smallest angle
    angle1 = waypoint_angle - robot_theta
    if waypoint_angle < 0:
        angle2 = waypoint_angle - robot_theta + 2*np.pi
    else:
        angle2 = waypoint_angle - robot_theta - 2*np.pi

    if abs(angle1) > abs(angle2):
        angle = angle2
    else:
        angle = angle1

    distance = np.sqrt((waypoint_x-robot_x)**2 + (waypoint_y-robot_y)**2) #calculates distance between robot and object

    print(f'Turn {angle} and drive {distance}')

    wheel_vel = 30 #ticks
    # Convert to m/s
    left_speed_m = wheel_vel * scale
    right_speed_m = wheel_vel * scale

    # Compute the linear and angular velocity
    linear_velocity = (left_speed_m + right_speed_m) / 2.0

    # Convert to m/s
    left_speed_m = -wheel_vel * scale
    right_speed_m = wheel_vel * scale

    angular_velocity = (right_speed_m - left_speed_m) / baseline

    print(f'Ang vel is {angular_velocity}')
    # turn towards the waypoint
    turn_time = abs(angle/angular_velocity)

    print("Turning for {:.2f} seconds".format(turn_time))
    if angle >= 0:
        lv1, rv1 = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    else:
        lv1, rv1 = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
    # after turning, drive straight to the waypoint
    drive_time = distance/linear_velocity
    print("Driving for {:.2f} seconds".format(drive_time))
    lv2, rv2 = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    new_robot_pose = [waypoint_x, waypoint_y, waypoint_angle]
    return new_robot_pose


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    # update the robot pose [x,y,theta]
    # image_poses = {}
    # with open(f'{script_dir}/lab_output/images.txt') as fp:
    #     for line in fp.readlines():
    #         pose_dict = ast.literal_eval(line)
    #         image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # robot_pose = image_poses[image_poses.keys()[-1]]
    ####################################################

    return [0,0,0]

def init_ekf():
    datadir = "calibration/param/"
    ip = "localhost"
    fileK = "{}intrinsic.txt".format(datadir)
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "{}distCoeffs.txt".format(datadir)
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileS = "{}scale.txt".format(datadir)
    scale = np.loadtxt(fileS, delimiter=',')
    if ip == 'localhost':
        scale /= 2
    fileB = "{}baseline.txt".format(datadir)  
    baseline = np.loadtxt(fileB, delimiter=',')
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    return EKF(robot)
    
# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following is only a skeleton code for semi-auto navigation
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose


        # robot drives to the waypoint
        waypoint = [x,y]
        robot_pose = drive_to_point(waypoint,robot_pose,ppi)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose,ppi))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
