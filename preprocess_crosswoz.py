import collections
import os
import json
import re
from textwrap import indent
from tkinter import dialog
from pathy import dataclass
from tqdm import tqdm
from collections import defaultdict
from dp_ops import Database

class Preprocessor(object):
    def __init__(self) -> None:
        self.data_dir = './data/crosswoz/'
        self.save_data_dir = os.path.join(self.data_dir, 'processed')
        if not os.path.exists(self.save_data_dir):
            os.makedirs(self.save_data_dir)

    def split_data_with_domains(self, all_data):
        '''
        all_data is the original data of crosswoz
        '''
        mapping = {
            '单领域': 'S',  # Single-domain
            '独立多领域': 'M',  # Independent multi-domain
            '独立多领域+交通': 'M+T',  # Independent multi-domain + traffic
            '不独立多领域': 'CM',  # Cross multi-domain
            '不独立多领域+交通': 'CM+T',  # Cross multi-domain + traffic
        }

        domains = defaultdict(list)
        for _, data in all_data.items():
            for dialog_id, dialog in data.items():
                domains[mapping[dialog['type']]].append(dialog_id)

            save_path = os.path.join(self.save_data_dir, 'dial_by_domain.json')
        with open(save_path, 'w') as fp:
            json.dump(domains, fp, ensure_ascii=False, indent=4)

    def extract_speical_tokens(self, all_data):
        save_path = os.path.join(self.save_data_dir, 'special_tokens.json')
        special_tokens = set()
        for _, data in all_data.items():
            for _, dialog in data.items():
                for single_turn in dialog['log']:
                    # sys_act = single_turn['sys_act'].split()
                    belief_state = single_turn['belief_state'].split()
                    # for token in sys_act:
                    #     if token.startswith('[') and token.endswith(']'):
                    #         special_tokens.add(token)

                    for token in belief_state:
                        if token.startswith('[') and token.endswith(']'):
                            special_tokens.add(token)

        special_tokens = list(special_tokens)
        with open(save_path, 'w') as fp:
            json.dump(special_tokens, fp, ensure_ascii=False, indent=4)

    def process(self):
        with open("./data/crosswoz/train.json") as f:
            train_data = json.load(f)
        with open("./data/crosswoz/val.json") as f:
            val_data = json.load(f)
        with open("./data/crosswoz/test.json") as f:
            test_data = json.load(f)

        all_data = {'train': train_data, 'val': val_data, 'test': test_data}
        self.split_data_with_domains(all_data)

        self.extract_belief_states_labels()

        train_data = self.convert_data_to_MultiWOZ(train_data, self.save_data_dir, 'train')
        val_data = self.convert_data_to_MultiWOZ(val_data, self.save_data_dir, 'val')
        test_data = self.convert_data_to_MultiWOZ(test_data, self.save_data_dir, 'test')

        train_data = self.generate_data(train_data, 'train')
        val_data = self.generate_data(val_data, 'val')
        test_data = self.generate_data(test_data, 'test')

        all_data = {'train': train_data, 'val': val_data, 'test': test_data}
        # self.extract_speical_tokens(all_data)
        # print(all_data)

    def extract_belief_states_labels(self):
        self.all_states_labels = {}

        with open("./data/crosswoz_dst/train_dials.json") as f:
            train_data = json.load(f)
        with open("./data/crosswoz_dst/dev_dials.json") as f:
            val_data = json.load(f)
        with open("./data/crosswoz_dst/test_dials.json") as f:
            test_data = json.load(f)

        all_data = {'train': train_data, 'val': val_data, 'test': test_data}
        for data_type in ['train', 'val', 'test']:
            data = all_data[data_type]
            for dialog in data:
                self.all_states_labels[dialog['dialogue_idx']] = []
                for round in dialog['dialogue']:
                    self.all_states_labels[dialog['dialogue_idx']].append(round['belief_state'])

    def remove_space(self, text):
        new_text = ''
        for i in text:
            if i != ' ':
                new_text += i
        return new_text

    def convert_act_to_span(self, acts):
        '''
        convert dialog acts or system acts to span
        '''
        result_span = ''
        act_dict = defaultdict(dict)
        for act in acts:
            token1, token2 = act.split('-')
            token1, token2 = token1.lower(), token2.lower()
            if token2 == 'general':
                act_dict[token2][token1] = {}
            else:
                if token2 not in act_dict[token1]:
                    act_dict[token1][token2] = []
                for sub_act in acts[act]:
                    act_dict[token1][token2].append(sub_act[0])

        for domain in act_dict:
            result_span += '[' + domain + '] '
            for intent in act_dict[domain]:
                result_span += '[' + intent + '] '

                if domain != 'general' and intent != 'nooffer':
                    for slot_name in act_dict[domain][intent]:
                        result_span += slot_name + ' '

        return result_span

    def generate_delex(self, resp, acts):
        '''
        generate delexical word
        '''

        with open('./data/crosswoz/database/ontology.json', 'r', encoding='utf8') as ot:
            otlg_all = json.load(ot)
        act_dict = defaultdict(dict)
        for act in acts:
            token1, token2 = act.split('-')
            token1, token2 = token1.lower(), token2.lower()
            if token2 == 'general':
                act_dict[token2][token1] = {}
            else:
                if token2 not in act_dict[token1]:
                    act_dict[token1][token2] = []
                for sub_act in acts[act]:
                    act_dict[token1][token2].append(sub_act[0])

        for domain in act_dict:
            for intent in act_dict[domain]:
                if domain != 'general' and intent != 'nooffer':
                    if "电话" in act_dict[domain][intent]:
                        value_list = []
                        for value in otlg_all[domain]["电话"]:
                            value = str(value)

                            if value in resp:
                                value_list.append(value)
                            value_list.sort(key=lambda i: len(i), reverse=True)
                        for value_tmp in value_list:
                            resp = resp.replace(value_tmp, "[value_电话]")

                    if "门票" in act_dict[domain][intent]:
                        value_list = []
                        for value in otlg_all[domain]["门票"]:
                            value = str(value)

                            if value in resp:
                                value_list.append(value)
                            value_list.sort(key=lambda i: len(i), reverse=True)
                        for value_tmp in value_list:
                            resp = resp.replace(value_tmp, "[value_门票]")


                    for slot_name in act_dict[domain][intent]:
                        if slot_name.find('-') >= 0:
                            slot_name = slot_name[0:slot_name.index('-')]
                        value_list = []
                        for value in otlg_all[domain][slot_name]:
                            value = str(value)

                            if value in resp:
                                value_list.append(value)
                            value_list.sort(key = lambda i:len(i),reverse=True)
                        for value_tmp in value_list:
                            resp = resp.replace(value_tmp,"[value_" + slot_name + "]")

        return resp

    def clean_slot_value(self, text):
        slot_value_mapping = {
            '高档下': '高档型',
            '舒适': '舒适型',
            '高档': '高档型',
            '豪华': '豪华型',
            '经济': '经济型',
        }

        if text in slot_value_mapping:
            return slot_value_mapping[text]

        return text

    def convert_belief_states_to_dict(self, belief_states):
        belief_states_span = ''
        belief_states_dict = {}
        for slot in belief_states:
            slot_pair = slot['slots'][0]
            slot_domain_name, slot_value = slot_pair
            domain, slot_name = slot_domain_name.split('-')
            slot_dict = {}
            slot_dict[slot_name] = slot_value
            belief_states_dict[domain] = slot_dict

        return belief_states_dict

    def convert_belief_states_to_span(self, belief_states):
        belief_states_span = ''
        belief_states_dict = defaultdict(list)
        for slot in belief_states:
            slot_pair = slot['slots'][0]
            slot_domain_name, slot_value = slot_pair
            domain, slot_name = slot_domain_name.split('-')
            belief_states_dict[domain].append((slot_name, slot_value))

        for domain in belief_states_dict:
            belief_states_span += '[' + domain + '] '
            for slot_pair in belief_states_dict[domain]:
                slot_name, slot_value = slot_pair
                slot_value = self.remove_space(slot_value)

                # clean data
                slot_value = self.clean_slot_value(slot_value)

                belief_states_span += '[value_' + slot_name + '] ' + slot_value + ' '

        return belief_states_span

    def convert_belief_states_to_span_delex(self, belief_states):
        belief_states_span = ''
        belief_states_dict = defaultdict(list)
        for slot in belief_states:
            slot_pair = slot['slots'][0]
            slot_domain_name, slot_value = slot_pair
            domain, slot_name = slot_domain_name.split('-')
            belief_states_dict[domain].append((slot_name, slot_value))

        for domain in belief_states_dict:
            belief_states_span += '[' + domain + '] '
            for slot_pair in belief_states_dict[domain]:
                slot_name, slot_value = slot_pair
                belief_states_span += '[value_' + slot_name + '] ' + ' '

        return belief_states_span

    def get_match(self, cons_dict):
        db = Database()
        match = {'general': ''}
        all_domains = ['餐馆', '景点', '酒店', '地铁', '出租']
        for domain in all_domains:
            match[domain] = ''
            if domain in cons_dict:
                matched_ents = db.query(cons_dict, domain)
                match[domain] = len(matched_ents)
        return match

    def get_turn_domain(self, acts):
        result_span = []
        act_dict = defaultdict(dict)
        for act in acts:
            token1, token2 = act.split('-')
            token1, token2 = token1.lower(), token2.lower()
            if token2 == 'general':
                act_dict[token2][token1] = {}
            else:
                if token2 not in act_dict[token1]:
                    act_dict[token1][token2] = []
                for sub_act in acts[act]:
                    act_dict[token1][token2].append(sub_act[0])

        for domain in act_dict:
            result_span.append(domain)

        return result_span

    def get_delex_valdict(self):
        entity_value_to_slot = {}

    def oneHotVector(self, domain, num):
        """Return number of available entities for particular domain."""
        vector = [0, 0, 0, 0, 0, 0]
        if num == '':
            return vector
        if num == 0:
            vector = [1, 0, 0, 0, 0, 0]
        elif num <= 80:
            vector = [0, 1, 0, 0, 0, 0]
        elif num <= 300:
            vector = [0, 0, 1, 0, 0, 0]
        else:
            vector = [0, 0, 0, 1, 0, 0]

        return vector

    def addDBPointer(self, domain, match_num):
        all_domains = ['餐馆', '景点', '酒店', '地铁', '出租']
        if domain in all_domains:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0, 0, 0, 0]
        return vector

    def pointerBack(self, vector, domain):
        # multi domain implementation
        # domnum = cfg.domain_num
        if domain.endswith(']'):
            domain = domain[1:-1]
        nummap = {
            0: '0',
            1: '1-80',
            2: '81-300',
            3: '>300'
        }
        if vector[:4] == [0, 0, 0, 0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain + ': ' + nummap[num] + '; '

        if vector[-2] == 0 and vector[-1] == 1:
            report += 'booking: ok'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'booking: unable'

        return report

    def generate_data(self, data, data_type):
        print("Generating {:s} data of MTTOD format".format(data_type))
        save_path = os.path.join(self.save_data_dir, data_type + '_mttod.json')
        dst_data = {}
        for dialogue_id, session in tqdm(data.items()):
            dst_data[dialogue_id] = {}

            # add goal and extract inform slots
            inform_slots = {}
            dst_data[dialogue_id]['goal'] = {}
            for sub_goal in session['goal']:
                if session['goal'][sub_goal] != []:
                    dst_data[dialogue_id]['goal'][sub_goal] = session['goal'][sub_goal]
                    inform_slots[sub_goal] = set()
                    for item in session['goal'][sub_goal]:
                        if 'inform' in item:
                            for slot_name in item['inform']:
                                inform_slots[sub_goal].add(slot_name)

            # add log
            log = []
            single_turn = {}
            for i, round in enumerate(session['log']):
                if i % 2 == 0:
                    # usr
                    turn_num = len(log)
                    single_turn['turn_num'] = turn_num
                    single_turn['user'] = round['text']
                    single_turn['user_delex'] = self.generate_delex(round['text'], round['dialog_act'])
                    single_turn['user_act'] = self.convert_act_to_span(round['dialog_act'])
                    single_turn['constraint'] = self.convert_belief_states_to_span(
                        self.all_states_labels[dialogue_id][turn_num])
                    single_turn['cons_delex'] = self.convert_belief_states_to_span_delex(
                        self.all_states_labels[dialogue_id][turn_num])
                else:
                    # sys
                    single_turn['resp'] = self.generate_delex(round['text'], round['dialog_act'])
                    single_turn['nodelx_resp'] = round['text']
                    single_turn['sys_act'] = self.convert_act_to_span(round['dialog_act'])
                    turn_domain = self.get_turn_domain(round['dialog_act'])
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')

                    single_turn['turn_domain'] = ' '.join("[" + d + "]" for d in turn_domain)
                    matnums = self.get_match(
                        self.convert_belief_states_to_dict(self.all_states_labels[dialogue_id][turn_num]))
                    match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[-1]
                    match = matnums[match_dom]
                    dbvec = self.addDBPointer(match_dom, match)
                    single_turn['pointer'] = ','.join([str(d) for d in dbvec])
                    single_turn['match'] = str(match)
                    prev_turn_domain = turn_domain
                    log.append(single_turn.copy())
                    single_turn = {}
            dst_data[dialogue_id]['log'] = log

        with open(save_path, 'w') as fp:
            json.dump(dst_data, fp, ensure_ascii=False, indent=4)
        return dst_data

    def get_user_goal(self, goal):
        user_goal = {'餐馆': [], '景点': [], '酒店': [], '出租': [], '地铁': []}  # A single domain may have two sub goals

        for sub_goal in goal:
            sub_goal_id, goal_domain, goal_slot_name, goal_slot_values, is_finished = sub_goal

            if is_finished:  # skip finished sub goals
                continue

            if len(user_goal[goal_domain]) == 0 or sub_goal_id != user_goal[goal_domain][-1]['sub_goal_id']:
                user_goal[goal_domain].append({'sub_goal_id': sub_goal_id})

            if goal_slot_values == [] or goal_slot_values == '':
                # request slot
                if 'reqt' not in user_goal[goal_domain][-1]:
                    user_goal[goal_domain][-1]['reqt'] = []
                user_goal[goal_domain][-1]['reqt'].append(goal_slot_name)
            else:
                # inform slot
                if 'inform' not in user_goal[goal_domain][-1]:
                    user_goal[goal_domain][-1]['inform'] = {}
                user_goal[goal_domain][-1]['inform'][goal_slot_name] = goal_slot_values

        return user_goal

    def convert_data_to_MultiWOZ(self, data, processed_dir, data_type):
        '''
        Convert CrossWOZ to the format of MultiWOZ
        data_type: train / test / val
        return True if successed
        '''

        saved_dir = os.path.join(processed_dir, data_type + '_mwoz.json')

        processed_data = {}
        print("Converting {:s} data to MultiWOZ's format".format(data_type))
        for dialogue_id, dialogue_info in tqdm(data.items()):
            single_session = {}
            # user_goal = {'餐馆': [], '景点': [], '酒店': [], '出租': [], '地铁': []} # A single domain may have two sub goals
            user_goal = self.get_user_goal(dialogue_info['goal'])
            # add user's goal information
            single_session['goal'] = user_goal

            log = []
            # add dialogue's turns
            for single_message in dialogue_info['messages']:
                utterance = {}
                # add text
                utterance['text'] = single_message['content']

                # add dialog action
                dialog_actions = {}
                for dialog_act in single_message['dialog_act']:
                    intent, domain, slot_name, slot_value = dialog_act
                    if slot_value == '':
                        # 处理request slot
                        slot_value = '?'
                    action_name = domain + '-' + intent
                    if action_name not in dialog_actions:
                        dialog_actions[action_name] = []
                    dialog_actions[action_name].append([slot_name, slot_value])
                utterance['dialog_act'] = dialog_actions

                # add metadata information
                if 'user_state' in single_message:
                    # metadata is user states
                    metadata = self.get_user_goal(single_message['user_state'])
                elif 'sys_state' in single_message:
                    # metadata is system states
                    metadata = single_message['sys_state']
                utterance['metadata'] = metadata
                log.append(utterance)

            single_session['log'] = log

            processed_data[dialogue_id] = single_session

        with open(saved_dir, 'w', encoding='utf-8') as fw:
            json.dump(processed_data, fw, ensure_ascii=False, indent=4)

        return processed_data


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.process()
