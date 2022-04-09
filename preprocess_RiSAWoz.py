#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import utils
from tqdm import tqdm


def oneHotVector(domain, num):
    """Return number of available entities for particular domain."""
    vector = [0, 0, 0, 0, 0, 0]
    if num == '':
        return vector
    if num == 0:
        vector = [1, 0, 0, 0, 0, 0]
    elif num <= 5:
        vector = [0, 1, 0, 0, 0, 0]
    elif num <= 10:
        vector = [0, 0, 1, 0, 0, 0]
    else:
        vector = [0, 0, 0, 1, 0, 0]

    return vector


def addDBPointer(domain, match_num):
    all_domains = ['旅游景点','餐厅', '酒店','火车','飞机','天气','电影','电视剧','汽车','电脑','医院','辅导班']
    if domain in all_domains:
        vector = oneHotVector(domain, match_num)
    else:
        vector = [0, 0, 0, 0, 0, 0]
    return vector


def process(dial_data):
    data = {}
    for num, mess in enumerate(tqdm(dial_data)):
        dialogue_id = mess["dialogue_id"]
        goal = mess["goal"]
        domains = mess["domains"]
        dialogue = mess["dialogue"]

        last_turn = len(dialogue)

        dial = {"goal": {}, "log": []}

        for i, single_turn in enumerate(dialogue):
            single = {}
            turn_id = single_turn["turn_id"]
            turn_domain = single_turn["turn_domain"]
            user_utterance = single_turn["user_utterance"]
            segmented_user_utterance = single_turn["segmented_user_utterance"]
            system_utterance = single_turn["system_utterance"]
            segmented_system_utterance = single_turn["segmented_system_utterance"]
            belief_state = single_turn["belief_state"]
            user_actions = single_turn["user_actions"]
            system_actions = single_turn["system_actions"]
            db_results = single_turn["db_results"]


            '''
            get the constraints
            '''
            constraint = ""
            cons_delex = ""
            user_delex = segmented_user_utterance
            slot_values = belief_state["inform slot-values"]
            turn_inform = belief_state["turn_inform"]
            turn_request = belief_state["turn request"]
            temp_s_v_by_domain = {}
            temp_s = {}
            for k, v in slot_values.items():
                domain = k.strip().split("-")[0]
                slot = k.strip().split("-")[1]
                if domain not in temp_s_v_by_domain:
                    temp_s_v_by_domain[domain] = []
                    temp_s_v_by_domain[domain].append("[value_"+str(slot)+"]")
                    temp_s_v_by_domain[domain].append(v)
                else:
                    temp_s_v_by_domain[domain].append("[value_"+str(slot)+"]")
                    temp_s_v_by_domain[domain].append(v)
                if domain not in temp_s:
                    temp_s[domain] = []
                    temp_s[domain].append("[value_"+str(slot)+"]")
                else:
                    temp_s[domain].append("[value_"+str(slot)+"]")

                if v and v in user_delex:
                    #user_delex = user_delex.replace(v, "["+str(domain)+"_"+str(slot)+"]")
                    user_delex = user_delex.replace(v, "[value_"+str(slot)+"]")

            for k in temp_s_v_by_domain:
                #temp_cons = [k]
                temp_cons = " ".join(temp_s_v_by_domain[k])
                constraint = constraint + "[" + str(k) + "] " + temp_cons + " "

                temp_cons_delex = " ".join(temp_s[k])
                cons_delex = cons_delex + "[" + str(k) + "] " + temp_cons_delex + " "

            #reqs = []
            #for r in turn_request:
            #    reqs.append(r)

            #if last_turn-1 == i:
            #    dial["goal"] = {}
            #    for k, v in slot_values.items():
            #        domain = k.strip().split("-")[0]
            #        slot = k.strip().split("-")[1]
            #        if domain not in dial["goal"]:
            #            dial["goal"][domain] = {}
            #            dial["goal"][domain]["info"] = {}
            #            dial["goal"][domain]["info"][slot] = v
            #        else:
            #            dial["goal"][domain]["info"][slot] = v

            #    #for req in turn_request:
            #    dial["goal"][domain]["reqt"] = []
            #    dial["goal"][domain]["reqt"].extend(reqs)
            #    #print("Now I ma in !!")

            #dial["goal"] = {}
            #print(turn_domain)
            #assert 1== 2
            t_d = turn_domain[0]
            if t_d not in dial["goal"]:
                dial["goal"][t_d] = {}
            for k, v in turn_inform.items():
                slot = k.strip().split("-")[1]
                if "info" not in dial["goal"][t_d]:
                    dial["goal"][t_d]["info"] = {}
                    dial["goal"][t_d]["info"][slot] = v
                else:
                    dial["goal"][t_d]["info"][slot] = v
            for r in turn_request:
                if "reqt" not in dial["goal"][t_d]:
                    dial["goal"][t_d]["reqt"] = []
                    dial["goal"][t_d]["reqt"].append(r)
                else:
                    dial["goal"][t_d]["reqt"].append(r)

            sys_act = ""
            resp = segmented_system_utterance
            temp_act = {}
            for s in system_actions:
                intent = s[0]
                domain = s[1]
                slot = s[2]
                value = s[3]
                temp_key = domain + "-" + intent
                if temp_key not in temp_act:
                    temp_act[temp_key] = []
                    temp_act[temp_key].append(slot)
                else:
                    temp_act[temp_key].append(slot)

                if value and value in resp:
                    resp = resp.replace(value, "[value"+"_"+str(slot)+"]")

            for k in temp_act:
                temp = " ".join(temp_act[k])
                sys_act = "[" + str(k.strip().split("-")[0]) + "] " + "[" + str(k.strip().split("-")[1]) + "] " + temp

            single["user"] = segmented_user_utterance
            single["user_delex"] = user_delex
            single["nodelx_resp"] = segmented_system_utterance
            single["resp"] = resp
            if len(db_results)-1 > 0:
                single["match"] = str(len(db_results) - 1)
                dbvec = addDBPointer(turn_domain[0], int(single["match"]))
            else:
                single["match"] = ""
                dbvec = addDBPointer(turn_domain[0], single["match"])
            single["pointer"] = ','.join([str(d) for d in dbvec])
            single["constraint"] = constraint.strip()
            single["cons_delex"] = cons_delex.strip()
            single["sys_act"] = sys_act
            single["turn_num"] = turn_id
            single["turn_domain"] = "[" + str(turn_domain[0]) + "]"
            #print(single)
            #assert 1 == 2

            dial["log"].append(single)

        data[dialogue_id] = dial
    return data


if __name__ == '__main__':
    for data_type in ['train', 'val', 'test']:
        print("Generating {:s} data of MTTOD format".format(data_type))
        with open('./data/RiSAWOZ/'+data_type+'.json') as f:
            dial_data = json.load(f)
        data = process(dial_data)
        saved_dir = "./data/RiSAWOZ/processed/" + data_type + ".json"

        with open(saved_dir, "w", encoding="utf-8") as fw:
            json.dump(data, fw, ensure_ascii=False, indent=4)





