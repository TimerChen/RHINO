import numpy as np
import queue
import torch
from collections import deque
import networkx as nx

CLS_LIST0 = [
    ("idle", [-1, -1], [-1, -1], -1, 0), 
    ("cheers", [-1, 1], [-1, -1], -1, 1),
    ("thumbup", [0, 0], [-1, -1], -1, 2), 
    ("handshake", [-1, 0], [-1, -1], -1, 3), 
    ("pick_can_R", [-1, 0], [-1, 1], 1, 4), 
    ("place_can_R", [-1, 1], [-1, 0], 2, 5), 
    ("pick_tissue_L", [0, -1], [0, -1], 3, 6), 
    ("pick_table_plate_LR", [0, 0], [3, 0], 4, 7), 
    ("handover_plate_L", [3, -1], [0, -1], 5, 8), 
    ("get_human_plate_L", [0, -1], [3, -1], 6, 9), 
    ("wash_plate_LR", [[3,0], [3,5]], [[3, 5], [3, 5]], [7,10], 10), 
    ("place_plate_L", [3, -1], [0, -1], 8, 11), 
    ("place_sponge_R", [-1, 5], [-1, 0], 9, 12), 
    ("cancel", [-1, -1], [-1, -1], -1, 0),
    ]
CLS_MAP0 = {0: 'idle', 1: 'cheers', 2: 'thumbup', 3: 'handshake', 4: 'pick_can_R', 5: 'place_can_R', 6: 'pick_tissue_L', 7: 'pick_table_plate_LR', 8: 'handover_plate_L', 9: 'get_human_plate_L', 
            10: 'wash_plate_LR', 
            (10, 0): 'wash_1', 
            (10, 1): 'wash_2', 
            11: 'place_plate_L', 12: 'place_sponge_R', 13: 'cancel', -1: 'not_sure'}
# skills: -1: none, 1:pick_can_R, 2:place_can_R, 3:pick_tissue_L, 
# 4:pick_table_plate_LR, 5: handover_plate_L, 6: get_human_plate_L, 7: wash_plate_LR, 8: place_plate_L, 9: place_sponge_R
SKILL2CLS0 = [-1, 4, 5, 6, 7, 8, 9, (10, 0), 11, 12, (10, 1)]
MANIP_NAME0 = ["None", "pick_can_R", "place_can_R", "pick_tissue_L", "pick_table_plate_LR", 
               "handover_plate_L", "get_human_plate_L", "pick_sponge_R", "place_plate_L", 
               "place_sponge_R", "wash_plate_LR"]
LONG_HORIZON_MASK0 = [0, 1, 2, 3, 6, 10]

# ===========================================================================================

# ITEM_LABEL_NAME = ["", "cap", "book", "stamp", "lamp"]
# skills: -1: none, 1:get_cap_R, 2:give_cap_R, 3:pick_stamp_R, 
# 4:stamp_R, 5: place_stamp_R, 6: close_lamp, 7: open_lamp, 8: give_book_L
CLS_LIST1 = [
    ("idle", [-1, -1], [-1, -1], -1, 0),
    ("handshake", [-1, 0], [-1, -1], -1, 3),
    ("thumbup", [0, 0], [-1, -1], -1, 2),
    ("get_cap_R", [-1, 0], [-1, 0], 1, 3),
    ("give_cap_R", [-1, 0], [-1, 0], 2, 4),
    ("pick_stamp_R", [-1, 0], [-1, 3], 3, 5),
    ("stamp_R", [-1, 3], [-1, 3], 4, 6),
    ("place_stamp_R", [-1, 3], [-1, 0], 5, 7),
    ("close_lamp", [0, 0], [4, 0], 6, 8),
    ("open_lamp", [4, 0], [0, 0], 7, 9),
    ("give_book_L", [0, -1], [0, -1], 8, 10),
    ("cancel", [-1, -1], [-1, -1], -1, 0),
]
CLS_MAP1 = {0: 'idle', 1: 'handshake', 2: 'thumbup', 3: 'get_cap_R', 4: 'give_cap_R', 
            5: 'pick_stamp_R', 6: 'stamp_R', 7: 'place_stamp_R', 8: 'close_lamp', 
            9: 'open_lamp', 10: 'give_book_L', 11: 'cancel', -1: 'not_sure'}
SKILL2CLS1 = [-1, 3, 4, 5, 6, 7, 8, 9, 10]
MANIP_NAME1 = ["None", "get_cap_R", "give_cap_R", "pick_stamp_R", "stamp_R", "place_stamp_R",
                "close_lamp", "open_lamp", "give_book_L"]

# ===========================================================================================

CLS_LIST2 = [
    ("idle", [-1, -1], [-1, -1], -1, 0), 
    ("cheers", [-1, 1], [-1, -1], -1, 1), 
    ("thumbup", [0, 0], [-1, -1], -1, 2), 
    ("handshake", [0, 0], [-1, -1], -1, 3), 
    ("wave", [-1, 0], [-1, -1], -1, 4), 
    ("take_photo", [-1, 0], [-1, -1], -1, 5), 
    ("spread_hand", [0, 0], [-1, -1], -1, 6), 
    ("pick_can_R", [-1, 0], [-1, 1], 1, 7), 
    ("place_can_R", [-1, 1], [-1, 0], 2, 8), 
    ("pick_tissue_L", [0, -1], [0, -1], 3, 9), 
    ("pick_table_plate_LR", [0, 0], [3, 0], 4, 10), 
    ("handover_plate_L", [3, -1], [0, -1], 5, 11), 
    ("get_human_plate_L", [0, -1], [3, -1], 6, 12), 
    ("wash_plate_LR", [[3,0], [3,5]], [[3, 5], [3, 5]], [7,10], 13), 
    ("place_plate_L", [3, -1], [0, -1], 8, 14), 
    ("place_sponge_R", [-1, 5], [-1, 0], 9, 15), 
    ("cancel", [-1, -1], [-1, -1], -1, 0)
    ]
CLS_MAP2 = {0: 'idle', 1: 'cheers', 2: 'thumbup', 3: 'handshake', 4: "wave", 5: "take_photo", 6: "spread_hand", 7: 'pick_can_R', 8: 'place_can_R', 9: 'pick_tissue_L',
            10: 'pick_table_plate_LR', 11: 'handover_plate_L', 12: 'get_human_plate_L', 
            13: 'wash_plate_LR', 
            (13, 0): 'wash_1', 
            (13, 1): 'wash_2', 
            14: 'place_plate_L', 15: 'place_sponge_R', 16: 'cancel', -1: 'not_sure'}
# skills: -1: none, 1:pick_can_R, 2:place_can_R, 3:pick_tissue_L, 
# 4:pick_table_plate_LR, 5: handover_plate_L, 6: get_human_plate_L, 7: wash_plate_LR, 8: place_plate_L, 9: place_sponge_R
SKILL2CLS2 = [-1, 7, 8, 9, 10, 11, 12, (13, 0), 14, 15, (13, 1)]
MANIP_NAME2 = MANIP_NAME0
LONG_HORIZON_MASK2 = [0, 1, 2, 5, 6, 12, 13]

# ===========================================================================================

# ITEM_LABEL_NAME = ["", "cap", "book", "stamp", "lamp"]
# skills: -1: none, 1:get_cap_R, 2:give_cap_R, 3:pick_stamp_R, 
# 4:stamp_R, 5: place_stamp_R, 6: close_lamp, 7: open_lamp, 8: give_book_L
CLS_LIST3 = [
    ("idle", [-1, -1], [-1, -1], -1, 0),
    ("thumbup", [0, 0], [-1, -1], -1, 2),
    ("handshake", [0, 0], [-1, -1], -1, 3),
    ("wave", [0, 0], [-1, -1], -1, 4), 
    ("take_photo", [0, 0], [-1, -1], -1, 5), 
    ("spread_hand", [0, 0], [-1, -1], -1, 6), 
    ("get_cap_R", [0, 0], [-1, 0], 1, 7),
    ("give_cap_R", [0, 0], [-1, 0], 2, 8),
    ("pick_stamp_R", [0, 0], [-1, 3], 3, 9),
    ("stamp_R", [0, 3], [-1, 3], 4, 10),
    ("place_stamp_R", [0, 3], [-1, 0], 5, 11),
    ("close_lamp", [0, 0], [4, -1], 6, 12),
    ("open_lamp", [4, 0], [0, -1], 7, 13),
    ("give_book_L", [0, -1], [0, -1], 8, 14),
    ("cancel", [-1, -1], [-1, -1], -1, 0),
]
 

CLS_MAP3 = {0: 'idle', 1: 'thumbup', 2: 'handshake', 3: 'wave', 4: 'take_photo', 
            5: 'spread_hand', 6: 'get_cap_R', 7: 'give_cap_R', 8: 'pick_stamp_R', 
            9: 'stamp_R', 10: 'place_stamp_R', 11: 'close_lamp', 12: 'open_lamp', 
            13: 'give_book_L', 14: 'cancel', -1: 'not_sure'}
SKILL2CLS3 = [-1, 6, 7, 8, 9, 10, 11, 12, 13]
MANIP_NAME3 = MANIP_NAME1

def get_scenario_cls(scenario):
    if scenario == 0:
        return CLS_LIST0, CLS_MAP0, SKILL2CLS0, MANIP_NAME0, LONG_HORIZON_MASK0, 
    elif scenario == 1:
        return CLS_LIST1, CLS_MAP1, SKILL2CLS1, MANIP_NAME1, None
    elif scenario == 2:
        return CLS_LIST2, CLS_MAP2, SKILL2CLS2, MANIP_NAME2, LONG_HORIZON_MASK2
    elif scenario == 3:
        return CLS_LIST3, CLS_MAP3, SKILL2CLS3, MANIP_NAME3, None

class ReactPlanner:
    def __init__(self, 
                 cls_history_len=3, 
                 stable_repeat=3, 
                 num_obj=1,
                 task_start_cnt=15,
                 scenario=0,
                 enable_prestart=True,
                 long_horizon=False,
                 ):
        self.cls_list, self.cls_map, self.skill2cls, self.manip_name, long_horizon_mask = get_scenario_cls(scenario)
        self.react_class_que = deque(maxlen=cls_history_len)
        self.stable_repeat = stable_repeat
        self.history_cls = deque(maxlen=stable_repeat)
        self.hand_occupancy = torch.tensor([0, 0])
        self.exec_skill = False
        self.exec_skill_id = -1
        self.num_obj = num_obj

        self.sleep_count_down = 0
        self.sleep_count = 30

        self.exec_phase = "idle"
        self.exec_motion = False
        self._cancel_id = None
        self.running_cls = 0

        # stable class is
        self.stable_react_cls = 0
        
        self.sustained_motion_mode = False
        self.cancel_by_another = True
        self.last_cls = -1
        self.last_cls_cnt = 0
        self.long_horizon_mask = long_horizon_mask
        
        self.long_horizon = long_horizon
        self.skill_todo_list = []
        self.skill_graph = build_graph(self.cls_list)
        
        self.task_start_cnt = task_start_cnt
        self.enable_prestart = enable_prestart
        if not enable_prestart:
            self.stable_repeat = stable_repeat + task_start_cnt
            self.task_start_cnt = 0
        
    @property
    def hand_occupancy_1hot(self):
        """For no hand occupancy, return zeros, otherwise return one-hot vector.
        """
        # self.hand_occupancy[1] = 1
        ret = np.zeros((2, self.num_obj))
        if self.hand_occupancy[0] != 0:
            ret[0, self.hand_occupancy[0]-1] = 1
        if self.hand_occupancy[1] != 0:
            ret[1, self.hand_occupancy[1]-1] = 1
            
        return ret
    
    @staticmethod
    def match_req(occu, req, get_id=False):
        if isinstance(req[0], list):
            for i, r in enumerate(req):
                if ReactPlanner.match_req(occu, r):
                    if get_id:
                        return i
                    return True
            if get_id:
                return -1
            return False
        assert not get_id
        return (req[0] == -1 or req[0] == occu[0]) and (req[1] == -1 or req[1] == occu[1])
        
    def react_cls_mask(self):
        mask = torch.zeros([1, len(self.cls_list)])
        if self.long_horizon:
            for i in self.long_horizon_mask:
                mask[:, i] = 1
            return mask
        for i in range(len(self.cls_list)):
            # if self.match_req(self.hand_occupancy, CLS_START_REQ[i]):
            if self.match_req(self.hand_occupancy, self.cls_list[i][1]):
                mask[:, i] = 1        
        return mask
    
    def done_skill(self, skill_id, done_signal):
        """ Signal: 0: success, 1: timeout, 2: cancel """
        cls_id = self.skill2cls[skill_id]
        modify = False
        print(f"Skill {skill_id} is done with signal {done_signal}.")
        if done_signal == 0:
            for i in range(2):
                if isinstance(cls_id, tuple) or isinstance(cls_id, list):
                    # if CLS_END_SWITCH[cls_id[0]][cls_id[1]][i] != -1:
                    #     self.hand_occupancy[i] = CLS_END_SWITCH[cls_id[0]][cls_id[1]][i]
                    if self.cls_list[cls_id[0]][2][cls_id[1]][i] != -1:
                        print("[INFO] Switch to", cls_id, self.cls_list[cls_id[0]][2][cls_id[1]])
                        self.hand_occupancy[i] = self.cls_list[cls_id[0]][2][cls_id[1]][i]
                else:
                    # if CLS_END_SWITCH[cls_id][i] != -1:
                    #     self.hand_occupancy[i] = CLS_END_SWITCH[cls_id][i]
                    if self.cls_list[cls_id][2][i] != -1:
                        print("[INFO] Switch to", cls_id, i, self.cls_list[cls_id][2][i])
                        self.hand_occupancy[i] = self.cls_list[cls_id][2][i]
            modify = True
        else:
            if self.long_horizon:
                # failed in the middle, clear the skill list
                self.skill_todo_list = []

        if self.long_horizon and len(self.skill_todo_list) > 0:
            # start next skill
            # next_skill = self.skill_todo_list.pop(0)
            return self.skill_todo_list[-1]

        self.sleep_count_down = self.sleep_count

        if modify:
            self.last_cls_cnt = 0
            self.react_class_que.clear()
                
        self.exec_phase = "idle"
        self.exec_skill = False
        self.exec_skill_id = -1
        self.running_cls = 0
        return None
                
    def start_skill(self, skill_id):
        if self.long_horizon and len(self.skill_todo_list) > 0:
            # assert skill_id == self.skill_todo_list[-1], f"Skill {skill_id} is not the final skill {self.skill_todo_list[-1]}"
            # import pdb; pdb.set_trace()
            skill_id = self.skill_todo_list.pop(0)
        self.exec_skill = True
        self.exec_skill_id = skill_id

        return skill_id

        # time.sleep(3)
        # self.done_skill(skill_id)

    def cls2skill(self, cls):
        # skill = CLS2SKILL[cls]
        skill = self.cls_list[cls][3]
        if isinstance(skill, list):
            if self.long_horizon:
                i = -1
            else:
                # i = self.match_req(self.hand_occupancy, CLS_START_REQ[cls], get_id=True)
                # assert i != -1, f"Skill {cls} has multiple start reqs {CLS_START_REQ[cls]}"
                i = self.match_req(self.hand_occupancy, self.cls_list[cls][1], get_id=True)
                assert i != -1, f"Skill {cls} has multiple start reqs {self.cls_list[cls][1]}"
            return skill[i]
        return skill
    
    @property
    def cancel_id(self):
        if self._cancel_id is not None:
            return self._cancel_id
        for i in range(len(self.cls_list)):
            if self.cls_list[i][0] == "cancel":
                self._cancel_id = i
                return i
        return None
    
    def get_cls(self, cls_pred, apply_mask=True):
        cls_pred = cls_pred.detach().cpu()
        mask = self.react_cls_mask()
        # print("mask shape", mask.shape, cls_pred.shape, mask)
        # mask[0, [4, 13]] = 0
        # mask[0, [13]] = 0
        mask[0, [2,5,9,10,11,12,13,14,15, 16]] = 0
        if apply_mask:
            cls_pred[mask == 0] = -float("inf")

        # end2end mask, only 5 tasks is able
        # mask[0, [2,5,6,9,10,11,12,13,14,15]] = 0
        # long-horizon mask, only ? tasks is able
        # mask[0, []] = 0

        # cls_pred[mask == 0] = -float("inf")
            
        cls_prob_ = torch.softmax(cls_pred, dim=-1)
        # print("pred prob", torch.round(cls_prob_*100).int().numpy(), mask)
        
        self.react_class_que.append(cls_prob_)
        cls_prob = sum(self.react_class_que) / len(self.react_class_que)
        cls = cls_prob.argmax().item()
        
        # skill_id = CLS2SKILL[cls]
        skill_id = self.cls2skill(cls)
        # print("now prob", torch.round(cls_prob_*100).int().numpy(), \
        #       torch.round(cls_prob*100).int().numpy(), cls, skill_id)
        return cls
    
    def check_switch(self, ):
        """
        [Manipulation Skill]
        1. idle -> pre_start: repeate skill [stable_repeat] times
        2. pre_start -> start: wait for [task_start_cnt] frames
        3. pre_start -> idle: 
          a. if other skill or idel repeat for [stable_repeat] frames
        4. start -> idle: Skill Done or Cancel by other skills.
          a. Cancel by other skills, if other skill repeat for [task_start_cnt] frames
          b. Cancel by "cancel", if cancel repeat for [] fraems, then reset the cnt down to 0.
          c. Skill Done, then reset the cnt down to 0.
        
        [Motion Skill]
        1. idle -> pre_start: repeate skill [stable_repeat] times
        2. pre_start -> start: wait for [task_start_cnt] frames
        3. pre_start -> others:
          a. if other skill or idle repeat for [stable_repeat] frames
        4. start -> others:
          a. if other skill or idle repeat for [task_start_cnt] frames
             For idle, reset the cnt down to 0.
        """
        if self.sleep_count_down > 0:
            self.sleep_count_down -= 1
            return False, None
            
        if self.exec_phase == "idle":
            # print("A")
            if self.last_cls_cnt >= self.stable_repeat and self.last_cls != 0:
                return True, self.last_cls
            return False, None
        elif self.exec_phase == "pre_start" and not self.exec_motion:
            # print("B")
            if self.last_cls_cnt < self.stable_repeat:
                # break by other skill, return to idle
                return True, 0
            elif self.last_cls_cnt > self.stable_repeat + self.task_start_cnt:
                return True, self.last_cls_cnt
        elif self.exec_phase == "start" and not self.exec_motion:
            # print("C")
            if self.last_cls == self.cancel_id and self.last_cls_cnt > self.stable_repeat:
                # cancel by cancel
                return True, 0
            elif self.last_cls != self.running_cls and self.last_cls != 0 and self.last_cls_cnt > int((self.stable_repeat + self.task_start_cnt)*1.5):
                # cancel by other skill
                return True, 0
        
        elif self.exec_phase == "pre_start" and self.exec_motion:
            # print("D")
            if self.last_cls_cnt < self.stable_repeat:
                # break by other skill, return to idle
                return True, 0
            elif self.last_cls_cnt > self.stable_repeat + self.task_start_cnt:
                return True, self.last_cls
        elif self.exec_phase == "start" and self.exec_motion:
            # print("E")
            if self.last_cls == self.cancel_id and self.last_cls_cnt > self.stable_repeat:
                # cancel by cancel
                return True, 0
            elif self.last_cls != self.running_cls and self.last_cls_cnt > self.stable_repeat + self.task_start_cnt:
                # cancel by other skill
                return True, 0
        # print("F")
        return False, None
            
    
    def update_cls(self, cls_pred, stable=True, 
                   apply_mask=True, 
                   no_skill=False,
                   time_avg=3,
                   continue_confirm=1):
        cls = self.get_cls(cls_pred, apply_mask=apply_mask)
        
        change_cls = None
        self.history_cls.append(cls)
        # print("stable queue:", self.history_cls)
        
        # update last cls
        if cls == self.last_cls:
            self.last_cls_cnt += 1
        else:
            self.last_cls = cls
            self.last_cls_cnt = 1
            
        switch_phase, next_cls = self.check_switch()
        # print("[Check switch]", switch_phase, next_cls, )
        # print("[Runing]", self.running_cls)
        ret_signal = None
        skill_id = -1
        start_manip_or_motion = False
        if switch_phase:
            if next_cls == 0:
                # cancel or break to idle
                # if cancel motion skill, set to idle, otherwise, send cancel to manip node.

                if self.exec_motion:
                    self.exec_phase = "idle"
                    self.exec_motion = False
                    self.running_cls = next_cls
                else:
                    print("try to cancel, but what happend?")
                    ret_signal = 'cancel' # send cancel to manip node
                    self.exec_phase = "cancel" # wait for cancel from manip node
            elif self.exec_phase == "idle":
                skill_id = self.cls2skill(next_cls)
                self.running_cls = next_cls
                self.exec_phase = "pre_start" if self.enable_prestart else "start"
                self.exec_motion = (skill_id == -1)
                start_manip_or_motion = True
                if not self.exec_motion:
                    ret_signal = 'start' # send start to manip node
            elif self.exec_phase == "pre_start":
                self.exec_phase = "start"
                
        if self.long_horizon and (start_manip_or_motion):
            skill_list = self.get_skill_list(next_cls)
            self.skill_todo_list = skill_list
            print("Skill list to do", [self.manip_name[s] for s in skill_list])
            if self.exec_motion and len(skill_list) > 0:
                # before start motion, first run manipulation skill to match the requirement
                self.exec_motion = False
                ret_signal = 'start'
        elif self.long_horizon and ret_signal == "cancel":
            self.skill_todo_list = []

        # print("[INFO] Planner Ret", self.exec_phase, self.exec_motion, cls, skill_id, ret_signal)
        return cls, skill_id, ret_signal, True
    
    def get_cls_req(self, cls):
        req = self.cls_list[cls][1]
        if isinstance(req[0], list):
            if self.long_horizon:
                i = -1
            else:
                i = self.match_req(self.hand_occupancy, req, get_id=True)
                assert i != -1, f"Skill {cls} has multiple start reqs {req}"
            return req[i]
        return req
    
    def get_skill_list(self, final_cls):
        final_skill = self.cls2skill(final_cls)
        req_of_final = self.get_cls_req(final_cls)
        print('[Final Skill]', final_skill, final_cls, req_of_final)
        skill_list, path = get_path(self.skill_graph, self.hand_occupancy, req_of_final)
        if final_skill != -1:
            skill_list.append(final_skill)
        return skill_list
    
def build_graph(cls_list):
    """
    Get Nodes to present the occupancy of hands.
    Build NetworkX Graph from [0,0]
    """
    # G = nx.Graph()
    # Graph with directed edges
    G = nx.DiGraph()
    G.add_node((0, 0))
    # bfs to build the graph
    q = queue.Queue()
    q.put((0, 0))
    while not q.empty():
        node = q.get()
        for cls in cls_list:
            # if cls[1] == node:
            if ReactPlanner.match_req(node, cls[1]):
                next_node = [0, 0]
                if isinstance(cls[2][0], list):
                    nid = ReactPlanner.match_req(node, cls[1], get_id=True)
                    print("nid", nid, node, cls[2])
                    move2 = cls[2][nid]
                    skill_cls = cls[3][nid]
                else:
                    move2 = cls[2]
                    skill_cls = cls[3]
                
                # next_node[0] = cls[2][0] if cls[2][0] != -1 else node[0]
                # next_node[1] = cls[2][1] if cls[2][1] != -1 else node[1]
                next_node[0] = move2[0] if move2[0] != -1 else node[0]
                next_node[1] = move2[1] if move2[1] != -1 else node[1]
                next_node = tuple(next_node)
                
                if not G.has_node(next_node):
                    
                    G.add_node(next_node)
                    q.put(next_node)
                # edge add info
                
                if node != next_node:
                    assert skill_cls != -1
                    # print("[Add Edge]", node, next_node, skill_cls)
                    G.add_edge(node, next_node, cls=skill_cls)
    return G

def get_path(G, start, end):
    # path = nx.shortest_path(G, start, end)
    # get shortest path with edge info
    print("[Find Path]", start, end)
    if isinstance(start, torch.Tensor):
        start = start.numpy().tolist()
        start = tuple(start)
    end = list(end)
    end[0] = end[0] if end[0] != -1 else start[0]
    end[1] = end[1] if end[1] != -1 else start[1]
    print(start, end)
    end = tuple(end)
    path = nx.shortest_path(G, start, end)
    skill_list = []
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        cls = G[source][target]['cls']
        skill_list.append(cls)
    return skill_list, path
