import numpy as np
import json
import os 


# -------------------actions predfine ----------------------------------
ACTION_NAMES = {'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                'kick', 'point', 'read', 'snowboard'}  # 4
ACTION_AGENT = {f"{action}_agent" for action in ACTION_NAMES}

ICAN_ACTION_ROLE = {"cut_instr": 2, "snowboard_instr": 21, "cut_obj": 4, "surf_instr": 0,
                     "skateboard_instr": 26, "kick_obj": 7, "eat_obj": 9, "carry_obj": 14,
                     "throw_obj": 15, "eat_instr": 16, "smile": 17, "look_obj": 18, "hit_instr": 19,
                     "hit_obj": 20, "ski_instr": 1, "run": 22, "sit_instr": 10, "read_obj": 24,
                     "ride_instr": 5, "walk": 3, "point_instr": 23, "jump_instr": 11,
                     "work_on_computer_instr": 8, "hold_obj": 25, "drink_instr": 13, "lay_instr": 12,
                     "talk_on_phone_instr": 6, "stand": 27, "catch_obj": 28}
INTRANSITIVE_VERBS = {"smile", "run", "walk", "stand"}

ACTION_ID2ROLE = {v: k for k, v in ICAN_ACTION_ROLE.items()}

ACTION_ROLE = {'hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk',  # 5
                            'look_obj', 'hit_instr', 'eat_instr', 'jump_instr', 'lay_instr',  # 5
                            'talk_on_phone_instr', 'carry_obj', 'throw_obj', 'catch_obj',  # 4
                            'cut_instr', 'run', 'work_on_computer_instr', 'ski_instr',  # 4
                            'surf_instr', 'skateboard_instr', 'smile', 'drink_instr',  # 4
                            'kick_obj', 'point_instr', 'read_obj', 'snowboard_instr',   # 4
                            'hit_obj', 'cut_obj', 'eat_obj'} # add 3


# 原因是因为在训练集里只有25种verb，但是有标签的有29种[0-28]
valid_ids_verb = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28]
# [0, 28] -> [0, 24]
vcoco_action_invrese_ids = {k: i for i, k in enumerate(valid_ids_verb)}

coco_object_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90]
# trans[1, 90] to [0, 79]
coco_object_inverse_ids = {idx: i for i, idx in enumerate(coco_object_valid_ids)}

# [1, 90]
coco_classes_originID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}

if __name__ == "__main__":
    # with open('/home/dell/HOITransformer/data/data_clean_out/000001.json','r') as load_f :
    #     load_json = json.load(load_f)
    files= os.listdir('/home/dell/HOITransformer/data/data_clean_out/json/penn_action/')
    files = sorted(files)

    for name  in  files:
        load_txts = os.listdir("/home/dell/HOITransformer/data/data_clean_out/json/penn_action/"+name)
        load_txts = sorted(load_txts)
        os.mkdir("./String_to_ID/"+name)
        for load_txt in load_txts:
            # os.mkdir("./String_to_ID"+name+"/"+load_txt)
            with open("/home/dell/HOITransformer/data/data_clean_out/json/penn_action/"+name+"/"+load_txt) as file:
                string = file.read()
                for id in coco_classes_originID.keys():
                    string = string.replace(id,str(coco_classes_originID[id]))
        
                for id in ICAN_ACTION_ROLE.keys():
                    string = string.replace(id,str(ICAN_ACTION_ROLE[id]))

                f = open("./String_to_ID/"+name+"/"+load_txt,'w')
                f.write(string)
                f.close()
            




    # string = f.read()
    # # name = string.split("\n")[0]
    # # load_json = string.split("\n")[1]
    # # load_json = load_json.replace("'",'"')

    # # print(coco_classes_originID.keys())
    # print(string)
    # for id in coco_classes_originID.keys():
    #     string = string.replace(id,str(coco_classes_originID[id]))
    
    # for id in ICAN_ACTION_ROLE.keys():
    #     string = string.replace(id,str(ICAN_ACTION_ROLE[id]))

    # file = open("T0002.txt",'w')
    # file.write(string)
    # file.close()

    #     # print (id)
    # print(string)


    # print(load_json)
        
