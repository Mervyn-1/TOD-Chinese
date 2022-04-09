import pprint
import os
import math
import argparse
import logging

from nltk.util import ngrams

from collections import Counter, OrderedDict

from reader import CrossWOZReader

from utils import definitions_cw
from utils.io_utils import get_or_create_logger, load_json
from utils.clean_dataset import clean_slot_values



logger = get_or_create_logger(__name__)

class BLEUScorer:
    """
    BLEU score calculator via GentScorer interface
    it calculates the BLEU-4 by taking the entire corpus in
    Calulate based multiple candidates against multiple references
    """
    def __init__(self,reader):
        self.reader = reader
        pass

    def score(self, parallel_corpus):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = self.reader.tokenizer.convert_ids_to_tokens(hyps)
            refs = self.reader.tokenizer.convert_ids_to_tokens(refs)
            hyps = [hyps[2:-2]]
            refs = [refs[2:-2]]

            for hyp in hyps:
                for i in range(4):
                    # accumulate ngram counts
                    #print(ngrams(hyp, i + 1))
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng]))
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0
                for i in range(4)]
        s = math.fsum(w * math.log(p_n)
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100

class CrossWOZEvaluator(object):
    def __init__(self, reader, eval_data_type="test"):
        self.reader = reader
        self.all_domains = definitions_cw.ALL_DOMAINS

        self.gold_data = load_json(os.path.join(
            self.reader.data_dir, "{}.json".format(eval_data_type)))

        self.eval_data_type = eval_data_type

        self.bleu_scorer = BLEUScorer(self.reader)

        self.all_info_slot = []
        for d, s_list in definitions_cw.INFORMABLE_SLOTS.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)


        self.requestables = ['电话', '地址', '评分', '门票', '价格']

    def bleu_metric(self, data, eval_dial_list=None):
        gens, truths = [], []
        for dial_id, dial in data.items():
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            for turn in dial:
                # excepoch <bos_resp>, <eos_resp>
                resp_gen_id = self.reader.tokenizer.encode(turn['resp_gen'])
                redx_id = self.reader.tokenizer.encode(turn['redx'])
                gens.append(resp_gen_id)
                truths.append(redx_id)
                #gens.append(turn['resp_gen'])
                #truths.append(turn['redx'])

        if gens and truths:
            sc = self.bleu_scorer.score(zip(gens, truths))
        else:
            sc = 0.0
        return sc

    def value_similar(self, a, b):
        return True if a == b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn):
        constraint_dict = self.reader.bspn_to_constraint_dict(bspn)

        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s, v in cons.items():
                key = domain+'-'+s
                constraint_dict_flat[key] = v

        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons,
                            slot_appear_num=None, slot_correct_num=None):
        tp, fp, fn = 0, 0, 0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            # v_truth = truth_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(
                        slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(
                    slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp, fp, fn, acc, list(set(false_slot))

    def dialog_state_tracking_eval(self, dials):
        total_turn, joint_match = 0, 0
        total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0
        for dial_id in dials:
            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num, turn in enumerate(dial):
                gen_cons = self._bspn_to_dict(turn['bspn_gen'])
                truth_cons = self._bspn_to_dict(turn['bspn'])

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                tp, fp, fn, acc, false_slots = self._constraint_compare(
                    truth_cons, gen_cons, slot_appear_num, slot_correct_num)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / \
            (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn+1e-10) * 100

        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num

    def aspn_eval(self, dials, eval_dial_list=None):
        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                if cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1 * 100

    def context_to_response_eval(self, dials, eval_dial_list=None, add_auxiliary_task=False):
        counts = {}
        for req in self.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}

            for domain in self.all_domains:
                if self.gold_data[dial_id]['goal'].get(domain):
                    true_goal = self.gold_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)

            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']


            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(
                dial, goal, reqs, counts, add_auxiliary_task=add_auxiliary_task)
            '''
            if success == 0 or match == 0:
                print("success ", success, "; match ", match)
                print(goal)
                for turn in dial:
                    print("=" * 50 + " " + str(dial_id) + " " + "=" * 50)
                    print("user               | ", turn["user"])
                    print("-" * 50 + " " + str(turn["turn_num"]) + " " + "-" * 50)
                    print("bspn               | ", turn["bspn"])
                    print("bspn_gen           | ", turn["bspn_gen"])
                    if "bspn_gen_with_span" in turn:
                        print("bspn_gen_with_span | ", turn["bspn_gen_with_span"])
                    print("-" * 100)
                    print("resp               | ", turn["redx"])
                    print("resp_gen           | ", turn["resp_gen"])
                    print("=" * 100)

                input()
            '''
            successes += success
            matches += match
            dial_num += 1

            # for domain in gen_stats.keys():
            #     gen_stats[domain][0] += stats[domain][0]
            #     gen_stats[domain][1] += stats[domain][1]
            #     gen_stats[domain][2] += stats[domain][2]

            # if 'SNG' in filename:
            #     for domain in gen_stats.keys():
            #         sng_gen_stats[domain][0] += stats[domain][0]
            #         sng_gen_stats[domain][1] += stats[domain][1]
            #         sng_gen_stats[domain][2] += stats[domain][2]

        # self.logger.info(report)
        succ_rate = successes/(float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100

        return succ_rate, match_rate, counts, dial_num

    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                   soft_acc=False, add_auxiliary_task=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
        #'id'
        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0:
                continue

            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            for domain in goal.keys():

                # for computing success
                if '[value_名称]' in sent_t:
                    if domain in ['餐馆', '酒店', '景点','地铁']:

                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        bspn = turn['bspn_gen']

                        # bspn = turn['bspn']
                        constraint_dict = self.reader.bspn_to_constraint_dict(
                            bspn)

                        if constraint_dict.get(domain):
                            venues = []
                            venues_all = self.reader.db.query(
                                constraint_dict,domain)
                            for venue in venues_all:
                                venue = venue[0]
                                venues.append(venue)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if ven not in venue_offered[domain]:
                                    # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:
                                # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_名称]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if '[value_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)



        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if '名称' in goal[domain]['informable']:
                venue_offered[domain] = '[value_名称]'

            # special domains - entity does not need to be provided
            # if domain in ['taxi', 'police', 'hospital']:
            #     venue_offered[domain] = '[value_name]'
            #
            # if domain == 'train':
            #     if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
            #         venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'餐馆': [0, 0, 0],
                '酒店': [0, 0, 0],
                '景点': [0, 0, 0],
                '地铁': [0, 0, 0],
                '出租': [0, 0, 0],}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['餐馆', '酒店', '景点','地铁']:
                goal_inform = {}
                goal_inform[domain] = goal[domain]['informable']
                goal_venues = []

                goal_venues_all = self.reader.db.query(
                    goal_inform,domain)
                for venue in goal_venues_all:
                    venue = venue[0]
                    goal_venues.append(venue)
                if type(venue_offered[domain]) is str and \
                   '_名称' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and \
                     len(set(venue_offered[domain]) & set(goal_venues))>0:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) ==0 and \
                        len(goal_venues) ==0:
                    match += 1
                    match_stat = 1
            else:
                if '_名称]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # for request in set(provided_requestables[domain]):
                #     if request in real_requestables[domain]:
                #         domain_success += 1
                # print(real_requestables[domain])
                # print(provided_requestables[domain])
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1
                # if domain_success >= len(real_requestables[domain]):
                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts

    def _parseGoal(self, goal, true_goal, domain):

        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': []}
        if 'inform' in true_goal[domain][0]:
            if 'reqt' in true_goal[domain][0]:
                for reqs in true_goal[domain][0]['reqt']:  # addtional requests:
                    if reqs in self.requestables:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(reqs)

            for s, v in true_goal[domain][0]['inform'].items():
                if isinstance(v,list):
                    v = "".join(v)
                s_, v_ = clean_slot_values(domain, s, v)
                if '-' in s_:
                    goal[domain]["informable"][s_.split('-')[0]] = s_.split('-')[1]
                else:
                    goal[domain]["informable"][s_] = v_

        return goal

    def run_metrics(self, data, domain="all", file_list=None):
        metric_result = {'domain': domain}
        bleu = self.bleu_metric(data, file_list)

        jg, slot_f1, slot_acc, slot_cnt, slot_corr = self.dialog_state_tracking_eval(
            data, file_list)

        metric_result.update(
            {'joint_goal': jg, 'slot_acc': slot_acc, 'slot_f1': slot_f1})

        info_slots_acc = {}
        for slot in slot_cnt:
            correct = slot_corr.get(slot, 0)
            info_slots_acc[slot] = correct / slot_cnt[slot] * 100
        info_slots_acc = OrderedDict(sorted(info_slots_acc.items(), key=lambda x: x[1]))

        act_f1 = self.aspn_eval(data, file_list)

        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, file_list)

        req_slots_acc = {}
        for req in self.requestables:
            acc = req_offer_counts[req+'_offer']/(req_offer_counts[req+'_total'] + 1e-10)
            req_slots_acc[req] = acc * 100
        req_slots_acc = OrderedDict(sorted(req_slots_acc.items(), key = lambda x: x[1]))

        if dial_num:
            metric_result.update({'act_f1': act_f1,
                'success': success,
                'match': match,
                'bleu': bleu,
                'req_slots_acc': req_slots_acc,
                'info_slots_acc': info_slots_acc,
                'dial_num': dial_num})

            logging.info('[DST] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f',
                         jg, slot_acc, slot_f1, act_f1)
            logging.info('[CTR] match: %2.1f  success: %2.1f  bleu: %2.1f',
                         match, success, bleu)
            logging.info('[CTR] ' + '; '
                         .join(['%s: %2.1f' %(req, acc) for req, acc in req_slots_acc.items()]))

            return metric_result
        else:
            return None

    def e2e_eval(self, data, eval_dial_list=None, add_auxiliary_task=False):
        bleu = self.bleu_metric(data)
        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, eval_dial_list=eval_dial_list, add_auxiliary_task=add_auxiliary_task)

        return bleu, success, match