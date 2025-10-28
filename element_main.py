from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from PIL import Image
from net import *
import torchvision.transforms as transforms
import cv2
import time
from torch2trt import TRTModule
import torch2trt
import torch
import trt_pose.models
import json
import trt_pose.coco
import random
import math
import numpy as np
import element_eval

class Player:
    def __init__(self, hp, mp, max_mp, el_center_x, peaks, n_peaks, beginning, ending):
        self.hp = hp
        self.mp = mp    # num of dots
        self.max_hp = hp
        self.max_mp = max_mp
        self.element_center_x = el_center_x
        self.peaks = peaks
        self.n_peaks = n_peaks
        self.pose = -1
        self.tmp_pose = -1
        self.accumulate = False
        self.last_warm_up_time = 0
        self.accumulate_list = []
        self.tmp_list = []
        self.if_shoot = False
        
    def pose_remain(self, new_pose):
        if new_pose == 5 or new_pose == -1:  # shoot
            if new_pose == 5:
                self.if_shoot = True
            return False
        if new_pose != self.tmp_pose:
            if new_pose != self.pose:
                self.last_warm_up_time -= 1
            self.tmp_pose = new_pose
        else:
            self.last_warm_up_time += 1

        if self.last_warm_up_time >= 5:
            if new_pose != self.pose:
                self.accumulate = False
                self.pose = new_pose
            else:
                self.accumulate = True
        return self.accumulate

    def generate_dots(self, new_pose):
        if self.mp < self.max_mp:
            self.mp += 1
            self.tmp_list.append(torch.randn(5, 2))
        if_accumulate = self.pose_remain(new_pose)
        if if_accumulate and new_pose != -1:
            for i in self.tmp_list:
                self.accumulate_list.append(i)
            self.tmp_list = [] 
        #print(f'tmp: {len(self.tmp_list)}')
        #print(f'accu: {len(self.accumulate_list)}')
def game_start(p1, p2, op_sec):
    button_fist = (cap_width//2, cap_height//2)
    x1, y1 = p1.peaks[0, 10, 0, :]
    x2, y2 = p2.peaks[0, 10, 0, :]
    x1 = x1 * cap_width
    y1 = y1 * cap_height
    x2 = x2 * cap_width
    y2 = y2 * cap_height
    p1_length = math.sqrt((x1-button_fist[0])**2+(y1-button_fist[1])**2)
    p2_length = math.sqrt((x2-button_fist[0])**2+(y2-button_fist[1])**2)
    if p1_length <= circle_radius and p2_length <= circle_radius:
        if op_sec < (duration * fps):
            op_sec += 1
    else:
        if op_sec > 0:
            op_sec -= 1
    if op_sec == (duration * fps):
        return True, op_sec
    else:
        return False, op_sec
    
def detect_game_end(p1, p2, remain_sec):
    # hp detect  index:{0:'continue', 1:'p1 win', 2:'p2 win', 3:'draw'}
    if remain_sec > 0:
        if p1.hp <= 0 and p2.hp <= 0:
            return 3
        elif p2.hp <= 0:
            return 1
        elif p1.hp <= 0:
            return 2
        else:
            return 0
    else:
        if p1.hp == p2.hp:
            return 3
        elif p1.hp > p2.hp:
            return 1
        else:
            return 2
    
def gradient_color(ratio):
    if ratio > 0.6:
        red = int(255 * (1 - ratio) * 2)
        green = 255
    else:
        red = 255
        green = int(255 * ratio * 2)
    return (0, green, red)
    
def draw_health_bar(image, p1, p2):
    # draw p1 hp
    width = cap_width // 5 * 2
    p1_health_ratio = p1.hp / p1.max_hp
    p1_color = gradient_color(p1_health_ratio)
    p1_health_bar_width = int(width * p1_health_ratio)
    cv2.rectangle(image, (20, 20), (20 + p1_health_bar_width, 50), p1_color, -1)
    cv2.rectangle(image, (20, 20), (20 + width, 50), (255, 255, 255), 2)
    # draw p2 hp
    p2_health_ratio = p2.hp / p2.max_hp
    p2_color = gradient_color(p2_health_ratio)
    p2_health_bar_width = int(width * p2_health_ratio)
    cv2.rectangle(image, (cap_width - 20 - p2_health_bar_width, 20), (cap_width - 20, 50), p2_color, -1)
    cv2.rectangle(image, (cap_width * 3 // 5 - 20, 20), (cap_width - 20, 50), (255, 255, 255), 2)
    return image

def draw_timer(image, remain_sec):
    if remain_sec < 0:
        remain_sec = 0
    image = cv2.flip(image, 1)
    text1 = str(remain_sec // fps)
    (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 2, 10)
    image_h, image_w, _ = image.shape
    image_x = (image_w//2 - text_width//2)
    image_y1 = 30 + text_height // 2
    cv2.putText(image, text1, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 100, 255), 10)
    cv2.putText(image, text1, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    image = cv2.flip(image, 1)
    return image

def draw_dots(image, dot_list_1, dot_list_2, color_1, color_2, if_a):
    p1_color_1 = color_1 
    p1_color_2 = color_2 
    p2_color_1 = color_1 
    p2_color_2 = color_2 
    if if_a:
        center_1 = p1.element_center_x
        center_2 = p2.element_center_x
    else:
        center_1 = cap_width // 4
        center_2 = cap_width * 3 // 4
    element_color_list = [(0, 0, 255), (255, 148, 40), (0, 143, 175), (255, 0, 134), (179, 254, 78), (255,255,255)]
    if len(dot_list_1) != 0 :
        dot_list_1 = torch.cat(dot_list_1, dim=0)
        if p1_color_1 == -1:
            p1_color_1 = (255, 255, 255)
        else:
            p1_color_1 = element_color_list[p1_color_1]
        if p1_color_2 == -1:
            p1_color_2 = (255, 255, 255)
        else:
            p1_color_2 = element_color_list[p1_color_2]
        for i in dot_list_1:
            x = int(i[0] * 50 + center_1)
            y = int(i[1] * 50 + cap_height // 2)
            if x < 0:
                x = 5
            if y < 0:
                y = 5

            cv2.circle(image, (x, y), 3, p1_color_1, -1)

    if len(dot_list_2) != 0 :
        #print(dot_list_2)
        dot_list_2 = torch.cat(dot_list_2, dim=0)
        if p2_color_1 == -1:
            p2_color_1 = (255, 255, 255)
        else:
            p2_color_1 = element_color_list[p2_color_1]
        if p2_color_2 == -1:
            p2_color_2 = (255, 255, 255)
        else:
            p2_color_2 = element_color_list[p2_color_2]
        for i in dot_list_2:
            x = int(i[0] * 50 + center_2)
            y = int(i[1] * 50 + cap_height // 2)
            if y < 0:
                y = 5
            #print(p2_color_2)
            cv2.circle(image, (x, y), 3, p2_color_2, -1)

    return image
    
def counter(e1, e2):
    elc_list = {0:4, 1:0, 2:1, 3:2, 4:3}
    if elc_list[e1] == e2:
        #e1 counter e2
        return 1
    elif elc_list[e2] == e1:
        # e2 counter e1
        return 2
    else:
        return 0

def cal_hurt(p1 ,p2): #, p1_ifshoot, p2_ifshoot):
    if p1.element_center_x >= cap_width * 3 // 4 - 50 and len(p1.accumulate_list) != 0:
        p2.hp -= p1.accumulate_list[0].shape[0]
        if p2.hp < 0:
            p2.hp = 0
        p1.element_center_x = cap_width // 4
        p1.mp -= p1.accumulate_list[0].shape[0]
        p1.accumulate_list = []

    if p2.element_center_x <= cap_width // 4 + 50 and len(p2.accumulate_list) != 0:
        p1.hp -= p2.accumulate_list[0].shape[0]
        if p1.hp < 0:
            p1.hp = 0
        p2.element_center_x = cap_width *3 // 4
        p2.mp -= p2.accumulate_list[0].shape[0]
        p2.accumulate_list = []
    #if p1_ifshoot and p2_ifshoot:
    if p2.element_center_x - p1.element_center_x <= 50 and len(p1.accumulate_list) != 0 and len(p2.accumulate_list) != 0:
        counter_e = counter(p1.pose, p2.pose)
        energy_1 = 0
        energy_2 = 0
        if counter_e == 0:
            tmp = p1.accumulate_list[0].shape[0]
            energy_1 = max(tmp - p2.accumulate_list[0].shape[0], 0)
            energy_2 = max(p2.accumulate_list[0].shape[0] - tmp, 0)
        elif counter_e == 1:
            tmp = int(p1.accumulate_list[0].shape[0] * 1.2)
            energy_1 = max(tmp - p2.accumulate_list[0].shape[0], 0)
            energy_2 = max(p2.accumulate_list[0].shape[0] - tmp, 0)
        else:
            tmp = p1.accumulate_list[0].shape[0]
            energy_1 = max(tmp - int(p2.accumulate_list[0].shape[0] * 1.2), 0)
            energy_2 = max(int(p2.accumulate_list[0].shape[0] * 1.2) - tmp, 0)
        # drop dot
        p1.accumulate_list[0] = p1.accumulate_list[0][:energy_1, :]
        p1.mp = energy_1
        p2.accumulate_list[0] = p2.accumulate_list[0][:energy_2, :]
        p2.mp = energy_2
        if energy_1 == 0:
            p1.accumulate = False
        if energy_2 == 0:
            p2.accumulate = False
        p1.element_center_x = cap_width //4
        p2.element_center_x = cap_width *3 //4

def switch_peaks(peaks):
    # (18, 2)
    for i in range(peaks.shape[0]):
        peaks[i, 1] = (1.5 - peaks[i, 1])
    def sw_p(peak1, peak2):
        tmp = peak1
        peak1 = peak2
        peak2 = tmp
        return peak1, peak2
    peaks[1, :], peaks[2, :] = sw_p(peaks[1, :], peaks[2, :])
    peaks[3, :], peaks[4, :] = sw_p(peaks[3, :], peaks[4, :])
    peaks[5, :], peaks[6, :] = sw_p(peaks[5, :], peaks[6, :])
    peaks[7, :], peaks[8, :] = sw_p(peaks[7, :], peaks[8, :])
    peaks[9, :], peaks[10, :] = sw_p(peaks[9, :], peaks[10, :])
    peaks[11, :], peaks[12, :] = sw_p(peaks[11, :], peaks[12, :])
    peaks[13, :], peaks[14, :] = sw_p(peaks[13, :], peaks[14, :])
    peaks[15, :], peaks[16, :] = sw_p(peaks[15, :], peaks[16, :])
    return peaks

if __name__ == '__main__':
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    skeleton = human_pose['skeleton']
    
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    def preprocess(image):
        global device
        device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.Resize((224, 224))(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def normalize(objects):
        # objects : [18, 2]
        n_objects = torch.zeros(objects.shape)
        x_min = torch.min(objects[:, 0])
        x_range = torch.max(objects[:, 0]) - x_min
        y_min = torch.min(objects[:, 1])
        y_range = torch.max(objects[:, 1]) - y_min
        for i in range(objects.shape[0]):
            n_objects[i, 0] = (objects[i ,0] - x_min) / x_range
            n_objects[i, 1] = (objects[i ,1] - y_min) / y_range 
        return n_objects
    
    def detect_pose(objects):
        pose_list = ['fire', 'water', 'rock', 'lightning', 'wind', 'shoot']
        threshold = [0.08, 0.05, 0.03, 0.008, 0.15, 0.05]
        max_ans_pose = -1
        ans_pose_value = 1000
        for e in range(6):
            with open(f'element/{pose_list[e]}.json', 'r') as f:
                r_data = json.load(f)
                reference = torch.tensor(r_data['keypoints'])
                r_mask = torch.tensor(r_data['mask'])
                loss = (torch.square(objects - reference).sum(dim=1) * r_mask).sum() / (torch.sum(r_mask).item())
                if e == 0:
                    min_loss = loss
                    ans_pose = e
                else:
                    if loss < min_loss:
                        min_loss = loss
                        ans_pose = e
                if e == 5:
                    if ans_pose == 5:
                        print('shoot')
                    return ans_pose if min_loss < threshold[ans_pose] else -1
                #if e == 1:
                #    print(f'water loss: {loss}')
                #if e == 2:
                #    print(f'rock loss: {loss}')
                #if e == 3:
                #    print(f'lightening loss: {loss}')
                #if e == 4:
                #    print(f'wind loss: {loss}')
                #if e == 0:
                #    print(f'fire loss: {loss}')
                #if e == 5:
                #    print(f'shoot loss: {loss}')
                #if loss <= threshold[e] and loss - threshold[e] <= ans_pose_value:
                #    max_ans_pose = e
                #    ans_pose_value = loss
        #return max_ans_pose
        
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)

    # window set
    cap = cv2.VideoCapture('rtmp://140.116.56.6:1935/live')
    # cap = cv2.VideoCapture(0)
    cap_height = 720
    cap_width = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    # timer
    max_sec = 99
    sec = 0
    op_sec = 0
    st_sec = 0

    # player parameter
    max_player_hp = 1000
    original_player_hp = max_player_hp
    max_player_mp = 100
    original_player_mp = 0
    element_list = ['fire', 'water', 'rock', 'lightning', 'wind', 'shoot']

    # other parameter
    circle_radius = 100
    fps = 10
    max_sec = max_sec * fps
    duration = 2
    st_duration = 1
    angle_step = 360 / (duration * fps)
    fist_size = 150
    half_fist_size = int(fist_size / 2)
    fist_path = './element_images/fist-bump_color.png'
    fist_img = cv2.imread(fist_path)
    fist_img = cv2.resize(fist_img, (fist_size, fist_size))
    already_start = False
    start_fight = False

    zero_peaks = np.zeros((18, 2))
    zero_npeaks = np.zeros((18, 2))
    init_beginning = 0
    init_ending = 0
    p1 = Player(original_player_hp, original_player_mp, max_player_mp, cap_width // 4, zero_peaks, zero_npeaks, init_beginning, init_ending)
    p2 = Player(original_player_hp, original_player_mp, max_player_mp, cap_width * 3 // 4, zero_peaks, zero_npeaks, init_beginning, init_ending)
    while(True):
        ret, image = cap.read()
        if ret:
            bg_left = np.zeros((cap_height, cap_width, 3), dtype=np.uint8)
            bg_right = np.zeros((cap_height, cap_width, 3), dtype=np.uint8)
            image = cv2.resize(image, (cap_width, cap_height))
            cv2.line(image, (cap_width // 2, 0), (cap_width // 2, cap_height), (0, 0, 255), 10)
            bg_left[:, :cap_width//2, :] = image[:, :cap_width//2, :]
            bg_right[:, cap_width//2:, :] = image[:, cap_width//2:, :]
            data_left = preprocess(bg_left)
            data_right = preprocess(bg_right)
            #data = preprocess(image) # todo : split the image into two parts
            cmap_l, paf_l = model_trt(data_left)
            cmap_r, paf_r = model_trt(data_right)
            cmap_l, paf_l = cmap_l.detach().cpu(), paf_l.detach().cpu()
            cmap_r, paf_r = cmap_r.detach().cpu(), paf_r.detach().cpu()
            counts_1, objects_1, p1.peaks = parse_objects(cmap_l, paf_l)
            counts_2, objects_2, p2.peaks = parse_objects(cmap_r, paf_r)
            draw_objects(image, counts_1, objects_1, p1.peaks)
            draw_objects(image, counts_2, objects_2, p2.peaks)
            #p2.peaks[0, :, 0, :] = switch_peaks(p2.peaks[0, :, 0, :])
            n_object_1 = normalize(p1.peaks[0, :, 0, :])
            n_object_2 = normalize(p2.peaks[0, :, 0, :])
            #print(n_object_2)
            #print('----------------')

            if not already_start:
                already_start, op_sec = game_start(p1, p2, op_sec)
                current_angle = 360 - int(360 - op_sec * angle_step)
                fist_dist = (cap_width//2,cap_height//2)
                # add fist
                roi = image[fist_dist[1]-half_fist_size:fist_dist[1]+half_fist_size, fist_dist[0]-half_fist_size:fist_dist[0]+half_fist_size]
                img2gray = cv2.cvtColor(fist_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                img2_fg = cv2.bitwise_and(fist_img,fist_img,mask = mask)
                dst = cv2.add(img1_bg,img2_fg)
                image[fist_dist[1]-half_fist_size:fist_dist[1]+half_fist_size, fist_dist[0]-half_fist_size:fist_dist[0]+half_fist_size] = dst
                # add circle
                cv2.circle(image, fist_dist, circle_radius, (0,0,0), 10)
                cv2.ellipse(image, fist_dist, (circle_radius, circle_radius), 0, 0, current_angle, (5, 209, 255), thickness=10)
            elif not start_fight:
                image = draw_health_bar(image, p1, p2)
                image = draw_timer(image, max_sec)
                if st_sec < (st_duration * fps):
                    image = cv2.flip(image, 1)
                    text1 = f'Fight !!!'
                    (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 4, 10)
                    (text_width2, text_height2), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 4, 3)
                    image_h, image_w, _ = image.shape
                    image_x = (image_w//2 - text_width//2) 
                    image_y1 = image_h // 3
                    image_y2 = image_h // 3
                    cv2.putText(image, text1, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 10)
                    cv2.putText(image, text1, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 255), 3)
                    image = cv2.flip(image, 1)
                    st_sec += 1
                elif st_sec == (st_duration * fps):
                    start_fight = True
            else:
                # GAME START
                image = draw_health_bar(image, p1, p2)
                image = draw_timer(image, max_sec - sec)
                # Game end detection
                if detect_game_end(p1, p2, max_sec - sec) == 0:
                    sec += 1
                    # todo : Pose detection
                    p1.generate_dots(detect_pose(n_object_1))
                    p2.generate_dots(detect_pose(n_object_2))
                    # Draw dots
                    draw_dots(image, p1.tmp_list, p2.tmp_list, -1, -1, False)
                    draw_dots(image, p1.accumulate_list, p2.accumulate_list, p1.pose, p2.pose, True)
                    # ddpm
                    if p1.accumulate and p1.pose != -1 and p1.pose != 5 and len(p1.accumulate_list) != 0:
                        if sec % 5 == 0:
                            new_dots_1 = element_eval.diff_eval(torch.cat(p1.accumulate_list, dim = 0), f'./element/{element_list[p1.pose]}.pth', 3, device)
                            p1.accumulate_list = [torch.tensor(new_dots_1)]
                    if p1.if_shoot:    
                        p1.if_shoot == False
                        #    for at in p1.accumulate_list:
                        #        for atl in at:
                        #            atl[0] += 5
                        p1.element_center_x += 40
                        cal_hurt(p1,p2)
                    if p2.accumulate and p2.pose != -1 and p2.pose != 5:
                        if sec % 5 == 0:
                            new_dots_2 = element_eval.diff_eval(torch.cat(p2.accumulate_list, dim = 0), f'./element/{element_list[p2.pose]}.pth', 3, device)
                            p2.accumulate_list = [torch.tensor(new_dots_2)]
                    if p2.if_shoot:
                        p2.if_shoot = False
                        #for at in p2.accumulate_list:
                        #    for atl in at:
                        #        atl[0] -= 5
                        p2.element_center_x -= 40
                        cal_hurt(p1,p2)
                else:
                    # Game end
                    if detect_game_end(p1, p2, max_sec - sec) == 1:
                        # p1 win
                        image = cv2.flip(image, 1)
                        text1 = f'Player 1 WIN'
                        (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 10)
                        (text_width2, text_height2), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 3)
                        image_h, image_w, _ = image.shape
                        image_x = (image_w//2 - text_width) // 2
                        image_y1 = image_h // 3
                        image_y2 = image_h // 3
                        cv2.putText(image, text1, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 10)
                        cv2.putText(image, text1, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 3)
                        image = cv2.flip(image, 1)
                    elif detect_game_end(p1, p2, max_sec - sec) == 2:
                        # p2 win
                        image = cv2.flip(image, 1)
                        text1 = f'Player 2 WIN'
                        (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 10)
                        (text_width2, text_height2), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 3)
                        image_h, image_w, _ = image.shape
                        image_x = (image_w//2 - text_width) // 2 + image_w//2
                        image_y1 = image_h // 3
                        image_y2 = image_h // 3
                        cv2.putText(image, text1, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 10)
                        cv2.putText(image, text1, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 3)
                        image = cv2.flip(image, 1)
                    else:
                        #draw
                        image = cv2.flip(image, 1)
                        text1 = f'Draw'
                        (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 10)
                        (text_width2, text_height2), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 3)
                        image_h, image_w, _ = image.shape
                        image_x_1 = (image_w//2 - text_width) // 2
                        image_x_2 = (image_w//2 - text_width) // 2 + image_w//2
                        image_y1 = image_h // 3
                        image_y2 = image_h // 3
                        cv2.putText(image, text1, (image_x_1, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 10)
                        cv2.putText(image, text1, (image_x_1, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 3)
                        cv2.putText(image, text1, (image_x_2, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 10)
                        cv2.putText(image, text1, (image_x_2, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 3)
                        image = cv2.flip(image, 1)

            image = cv2.flip(image, 1)
            cv2.imshow('Element War', image)
            cv2.waitKey(1)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
