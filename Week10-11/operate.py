# teleoperate the robot, perform SLAM and object detection

import math
import os
import sys
import time
import cv2
import numpy as np
import json
from planner import *
# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import YOLO components 
from YOLO.detector import Detector


class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.ekf.known_map = True
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        #M4 initialization
        self.forward = False
        self.destination_idx = 0
        self.autonomous = False
        self.obstacles = []
        self.current_wp = [0,0] #set waypoint to second point in path
        self.robot_pose = [0,0,0]
        self.current_destination = None
        self.distance_threshold = 0.2
        self.goal_threshold = 0.25
        self.tick = 30
        self.turning_tick = 5
        self.map = 'known_map.txt'
        self.occupancy_grid = []
        self.path_mat = []
        self.waypoint_mat = []
        self.goal_pos=[]
        self.isPlanned = False
        self.path_planner = PathPlanner(self.obstacles, self.waypoint_mat)
        self.path_planner.obstacle_radius = args.obstacle_radius

    # wheel control
    def     control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'],tick=30)
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # running in sim
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        # running on physical robot (right wheel reversed)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()

        if self.data is not None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:  # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)

            # covert the colour back for display purpose
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)

            # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
            self.file_output = (yolo_input_img, self.ekf)

            # self.notification = f'{len(self.detector_output)} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
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

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # keyboard teleoperation, replace with your M1 codes if preferred        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

                # run auto fruit search
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.autonomous = True
                print("moving to " , self.current_wp)

        if self.quit:
            pygame.quit()
            sys.exit()

    def turn_robot(self):
        waypoint_x = (self.current_wp[0]/100)-1.5
        waypoint_y = (self.current_wp[1]/100)-1.5
        robot_x = self.robot_pose[0]
        robot_y = self.robot_pose[1]
        robot_theta = self.robot_pose[2]

        waypoint_angle = np.arctan2((waypoint_y-robot_y),(waypoint_x-robot_x))

        theta1 = robot_theta - waypoint_angle

        if (self.current_wp[1],self.current_wp[0]) == self.waypoint_mat[0]:
            self.theta_error = 0
        elif theta1 >np.pi:
            self.theta_error = theta1 - 2*np.pi
        elif theta1< -np.pi:
            self.theta_error = theta1 + 2*np.pi
        else:
            self.theta_error = theta1



        if self.forward == False:

            if self.theta_error > 0:
                self.command['motion'] = [0,-1]
                self.notification = 'Robot is turning right'

            if self.theta_error < 0:
                self.command['motion'] = [0,1]
                self.notification = 'Robot is turning left'

    def drive_robot(self):
        waypoint_x = (self.current_wp[0]/100)-1.5
        waypoint_y = (self.current_wp[1]/100)-1.5
        goal_x = (self.current_destination[0]/100)-1.5
        goal_y = (self.current_destination[1]/100)-1.5
        robot_x = self.robot_pose[0]
        robot_y = self.robot_pose[1]
        print("traveling to waypoint" , waypoint_x,waypoint_y)
        self.distance = np.sqrt((waypoint_x-robot_x)**2 + (waypoint_y-robot_y)**2) #calculates distance between robot and waypoint
        self.goal_distance =  np.sqrt((goal_x-robot_x)**2 + (goal_y-robot_y)**2)
        self.turn_robot() # turn robot

        # stop turning if less than threshold
        
        if not self.forward:
            print("angle error is ", self.theta_error)
            if abs(self.theta_error)  < 0.05:
                self.command['motion'] = [0,0]
                self.notification = 'Robot stopped turning'
                self.forward = True #go forward now
                return

        #Driving forward
        if self.forward:

            print("distance to waypoint " ,self.distance)
            print("distance to destination",self.goal_distance)
            #Drive until goal arrived
            if self.distance < self.distance_threshold or self.goal_distance < self.goal_threshold:
                self.command['motion'] = [0,0]
                self.notification = 'Robot arrived'
                self.forward = False
                self.destination_idx += 1
                
                #Check if last path and last waypoint reached
                if self.destination_idx > len(self.waypoint_mat)-1: #reached last wp of path
                    self.isPlanned = False
                else: #Increment path
                    self.current_wp = self.waypoint_mat[self.destination_idx]
                self.pibot.set_velocity([0,0],time = 3)
                print(f"Moving to new waypoint {self.current_wp}")
                return

            else:
                self.min_dist = self.distance
                self.command['motion'] = [1,0]
                self.notification = 'Robot moving forward'

    def print_target_fruits_pos(self,search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """
        fruit_pos = []
        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
                if fruit == fruit_list[i]:
                    fruit_pos.append([np.round(fruit_true_pos[i][0], 1),np.round(fruit_true_pos[i][1], 1)])
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                    fruit,
                                                    np.round(fruit_true_pos[i][0], 1),
                                                    np.round(fruit_true_pos[i][1], 1)))
            n_fruit += 1
        
        return fruit_pos
    
    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('M4_prac_shopping_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())
        return search_list

    def create_known_map(self):

        slam,target, output_file = './lab_output/slam.txt', './lab_output/target.txt', './known_map.txt'
        with open(slam, 'r') as infile:
            slam_data = json.load(infile)

        with open(target, 'r') as infile:
            target_data = json.load(infile)

        output_data = {}
        for i, tag in enumerate(slam_data['taglist']):
            output_data[f'aruco{tag}_{0}'] = {
                'x': slam_data['map'][0][i],
                'y': slam_data['map'][1][i]
            }

        output_data = {**output_data,**target_data}
        
        with open(output_file, 'w') as outfile:
            # Convert the output data to JSON format and write it to the output file
            json.dump(output_data, outfile, indent=2)


    def read_true_map(self,fname):
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
        
    def generate_obstacles(self):
        fruits_list, fruits_true_pos, aruco_true_pos = self.read_true_map(self.map)
        search_list = self.read_search_list()
        goals = self.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

        for i in range(len(aruco_true_pos)):
            landmark = measure.Marker(aruco_true_pos[i].reshape(-1,1),i+1)
            self.ekf.add_landmarks([landmark])

        all_objects_pos = [*fruits_true_pos,*aruco_true_pos]
        obstacle_pos = []
        goal_pos = []
        for i in range(len(all_objects_pos)):
            object_pos_map = []
            for j in range(len(all_objects_pos[i])):
                object_pos_map.append(round((all_objects_pos[i][j]+1.5)*100))

            if object_pos_map != self.current_destination:
                obstacle_pos.append(object_pos_map)

        for i in range(len(goals)):
            current_goal=[]
            for j in range(len(goals[i])):
                current_goal.append(round((goals[i][j]+1.5)*100))
            goal_pos.append(current_goal)
        return goal_pos,obstacle_pos

    def add_destination_obstacles(self,destinations,obstcales):
        for destination in destinations:
            if destination != self.current_destination:
                obstcales.append(destination)
        return obstcales
    
    def generate_waypoints(self):
        occupancy_grid = self.path_planner.create_occupancy_grid()

        path = self.path_planner.astar(occupancy_grid, self.path_planner.start, self.current_destination)
        waypoints = self.path_planner.extract_edges(path)
        self.path_mat= path
        self.waypoint_mat=waypoints

        self.path_planner.plot_path_on_occupancy_grid(self.path_mat)

        return self.waypoint_mat

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    parser.add_argument("--obstacle_radius",default=15,type =int)
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    print("obstacle radius is : " , operate.path_planner.obstacle_radius)
    operate.ekf_on = True
    while start:
        operate.update_keyboard()
        operate.take_pic()
        if operate.autonomous:
            operate.create_known_map()
            operate.goal_pos, _ = operate.generate_obstacles()
            print("order of destinations" ,operate.goal_pos)
            for destination in operate.goal_pos:
                operate.distance_threshold =0.2
                operate.destination_idx = 0
                operate.current_destination = destination
                print("current destination",operate.current_destination)
                if not operate.isPlanned:
                    _, operate.obstacles = operate.generate_obstacles()
                    print("obstacles",operate.obstacles)
                    operate.path_planner.obstacles = operate.obstacles
                    operate.path_planner.start = [int((operate.robot_pose[1][0]+1.5)*100),int((operate.robot_pose[0][0]+1.5)*100)]
                    print("starting from", operate.path_planner.start)
                    operate.waypoint_mat = operate.generate_waypoints()
                    print("waypoints",operate.waypoint_mat)
                    operate.isPlanned = True
                
                while operate.isPlanned:
                    operate.update_keyboard()
                    operate.take_pic()
                    print("current pose" ,operate.robot_pose)
                    operate.current_wp = operate.waypoint_mat[operate.destination_idx][::-1]
                    print(operate.current_wp)
                    operate.drive_robot()
                    drive_meas = operate.control()
                    operate.update_slam(drive_meas)
                    operate.robot_pose = operate.ekf.robot.state
                    operate.record_data()
                    operate.save_image()
                    operate.detect_target()
                    # visualise
                    operate.draw(canvas)
                    pygame.display.update()
            operate.autonomous = False

        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.robot_pose = operate.ekf.robot.state
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
