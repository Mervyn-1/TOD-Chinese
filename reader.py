from ftplib import all_errors
import os
import json
from turtle import back
import torch
import random
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, PreTrainedTokenizer, MBart50Tokenizer

from utils import definitions_cw
from utils.io_utils import load_json, load_pickle, save_pickle, get_or_create_logger
from db_ops import Database

logger = get_or_create_logger(__name__)

class BaseIterator(object):
    def __init__(self, reader) -> None:
        self.reader = reader
        self.dial_by_domain = load_json('./data/crosswoz/processed/dial_by_domain.json')

    def bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)

        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def construct_mini_batch(self, data, batch_size):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []

        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        
        return all_batches

    def get_batches(self, data_type, batch_size, shuffle=False, num_dialogs=-1, special_domain='all'):
        dial = self.reader.data[data_type]

        if special_domain != 'all':
            logger.info("Special domains: {}".format(special_domain))

            if special_domain in self.dial_by_domain.keys():
                target_dial_ids = self.dial_by_domain[special_domain]
            
            dial = [d for d in dial if d[0]['dial_id'] in target_dial_ids]

        if num_dialogs > 0:
            dial = random.sample(dial, min(num_dialogs, len(dial)))
        
        turn_bucket = self.bucket_by_turn(dial)

        all_batches = []

        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            # TODO: ??????????????????????????????

            batches = self.construct_mini_batch(turn_bucket[k], batch_size)
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches

        if shuffle:
            random.shuffle(all_batches)

        return all_batches, num_training_steps, num_dials, num_turns

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def flatten_dial_history(self, dial_history, len_postfix, context_size):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
        else:
            windowed_context = dial_history

        ctx_len = sum([len(c) for c in windowed_context])

        spare_len = self.reader.max_seq_len - len_postfix - 2
        while ctx_len >= spare_len:
            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)
        
        context = list(chain(*windowed_context))

        return context


    def get_data_iterator(self, all_batches, task, ururu, context_size=-1):
        raise NotImplementedError

class CrossWOZIterator(BaseIterator):
    def __init__(self, reader) -> None:
        super().__init__(reader)

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ['user', 'resp',"redx", 'bspn', 'aspn',"dbpn",
                        "bspn_gen","dbpn_gen", "aspn_gen", "resp_gen"]
        for dial in dial_batch:
            dial_id = dial[0]['dial_id']
            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}
                for k, v in turn.items():
                    try:
                        if k == 'dial_id':
                            continue
                        elif k == "pointer":
                            turn_doamin = turn["turn_domain"][-1]
                            v = self.reader.db.pointerBack(v, turn_doamin)
                        elif k in decoded_keys:
                            v = self.reader.tokenizer.decode(v)
                            if v[0] == '[SEP]':
                                v = v[1:]
                        readable_turn[k] = v
                    except:
                        if k == 'bspn_gen':
                            readable_turn[k] = '<bos_belief> <eos_belief>'
                        elif k == 'dbpn_gen':
                            readable_turn[k] = '<bos_db> [db_null] <eos_db>'
                        elif k == 'aspn_gen':
                            readable_turn[k] = '<bos_act> <eos_act>'
                        elif k == 'resp_gen':
                            readable_turn[k] = '<bos_resp> <eos_resp>'

                dialogs[dial_id].append(readable_turn)
        
        return dialogs

    def get_data_iterator(self, all_batches, task, ururu, context_size=-1):
        for dial_batch in all_batches:
            batch_encoder_input_ids = []
            batch_encoder_resp_inputs_ids = []
            batch_belief_label_ids = []
            batch_resp_label_ids = []
            # batch_whole_label_ids = []

            for dial in dial_batch:
                dial_encoder_inputs_ids = []
                dial_beleif_label_ids = []
                dial_resp_label_ids = []
                dial_resp_inputs_ids = []
                # dial_whole_label_ids =[]

                dial_history = []
                for turn in dial:
                    context_belief = self.flatten_dial_history(dial_history, len(turn['user']), context_size)
                    context_resp = self.flatten_dial_history(dial_history,len(turn['user'])+len(turn['bspn'])+len(turn['dbpn']),context_size)
                    encoder_input_ids = [self.reader.cls_token_id] + context_belief + turn['user'] + [self.reader.eos_token_id]
                    encoder_resp_input_ids = [self.reader.cls_token_id] + context_resp + turn['user'] + turn['bspn'] + turn['dbpn']+[self.reader.eos_token_id]
                    #belief labels
                    bspn = turn['bspn']
                    bspn_label = bspn
                    belief_label_ids = bspn_label + [self.reader.eos_token_id]
                    #resp labels
                    resp = turn["aspn"] + turn["redx"]
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    # whole_label_ids = bspn_label + resp + [self.reader.eos_token_id]


                    dial_encoder_inputs_ids.append(encoder_input_ids)
                    dial_resp_inputs_ids.append(encoder_resp_input_ids)

                    dial_beleif_label_ids.append(belief_label_ids)
                    dial_resp_label_ids.append(resp_label_ids)
                    # dial_whole_label_ids.append(whole_label_ids)

                    if ururu:
                        if task == "dst":
                            turn_text = turn["user"]  + turn["resp"]
                        else:
                            turn_text = turn["user"] + turn["redx"]
                    else:
                        if task == "dst":
                            turn_text = turn["user"] + bspn  +\
                                turn["dbpn"] + turn["aspn"] + turn["resp"]
                        else:
                            turn_text = turn["user"] + bspn  +\
                                turn["dbpn"] + turn["aspn"] + turn["redx"]

                    dial_history.append(turn_text)


                batch_encoder_input_ids.append(dial_encoder_inputs_ids)
                batch_encoder_resp_inputs_ids.append(dial_resp_inputs_ids)
                batch_belief_label_ids.append(dial_beleif_label_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)
                # batch_whole_label_ids.append(dial_whole_label_ids)

            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            batch_encoder_resp_inputs_ids = self.transpose_batch(batch_encoder_resp_inputs_ids)
            batch_belief_label_ids = self.transpose_batch(batch_belief_label_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)
            # batch_whole_label_ids = self.transpose_batch(batch_whole_label_ids)

            num_turns = len(batch_belief_label_ids)

            tensor_encoder_input_ids = []
            tensor_encoder_resp_inputs_ids = []
            tensor_belief_label_ids = []
            tensor_resp_label_ids = []
            # tensor_whole_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [self.tensorize(b) for b in batch_encoder_input_ids[t]]
                tensor_encoder_resp_inputs_ids = [self.tensorize(b) for b in batch_encoder_resp_inputs_ids[t]]
                tensor_belief_label_ids = [self.tensorize(b) for b in batch_belief_label_ids[t]]
                tensor_resp_label_ids = [self.tensorize(b) for b in batch_resp_label_ids[t]]
                # tensor_whole_label_ids = [self.tensorize(b) for b in batch_whole_label_ids[t]]

                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids, batch_first=True, padding_value=self.reader.pad_token_id)
                tensor_encoder_resp_inputs_ids = pad_sequence(tensor_encoder_resp_inputs_ids, batch_first=True, padding_value=self.reader.pad_token_id)
                tensor_belief_label_ids = pad_sequence(tensor_belief_label_ids, batch_first=True, padding_value=self.reader.pad_token_id)
                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids,batch_first=True,padding_value=self.reader.pad_token_id)

                # tensor_whole_label_ids = pad_sequence(tensor_whole_label_ids,batch_first=True,padding_value=self.reader.pad_token_id)

                #yield tensor_encoder_input_ids, (tensor_belief_label_ids, tensor_resp_label_ids,tensor_whole_label_ids)
                # yield tensor_encoder_input_ids, tensor_whole_label_ids
                yield (tensor_encoder_input_ids,tensor_encoder_resp_inputs_ids), (tensor_belief_label_ids, tensor_resp_label_ids)
class BaseReader(object):
    def __init__(self, cfg) -> None:
        self.backbone = cfg.backbone
        self.cfg = cfg
        self.tokenizer = self.init_tokenizer()
        self.data_dir = self.get_data_dir()
        
        if '/' in self.cfg.backbone:
            _, backbone = self.cfg.backbone.split('/')
        else:
            backbone = self.cfg.backbone

        encoded_data_path = os.path.join(self.data_dir, backbone + '_encoded_data.pkl')
        if os.path.exists(encoded_data_path):
            logger.info('Load encoded data from {}'.format(encoded_data_path))
            self.data = load_pickle(encoded_data_path)
        else:
            logger.info('Encoded data and save to {}'.format(encoded_data_path))
            train = self.encode_data('train')
            dev = self.encode_data('val')
            test = self.encode_data('test')

            self.data = {'train': train, 'val': dev, 'test': test}

            save_pickle(self.data, encoded_data_path)

    def get_data_dir(self):
        raise NotImplementedError

    def init_tokenizer(self):
        if self.backbone in ['fnlp/cpt-base', 'fnlp/cpt-large', "fnlp/bart-base-chinese", 
                        "fnlp/bart-large-chinese"]:
            
            if self.cfg.ckpt is not None:
                logger.info('Load tokenizer from {}'.format(self.cfg.ckpt))
                return BertTokenizer.from_pretrained(self.cfg.ckpt)

            tokenizer = BertTokenizer.from_pretrained(self.backbone)
        elif self.backbone in ['facebook/mbart-large-50']:
            if self.cfg.ckpt is not None:
                logger.info('Load tokenizer from {}'.format(self.cfg.ckpt))
                return MBart50Tokenizer.from_pretrained(self.cfg.ckpt, src_lang='zh_CN', tgt_lang='zh_CN')
            tokenizer = MBart50Tokenizer.from_pretrained(self.backbone, src_lang='zh_CN', tgt_lang='zh_CN')
        else:
            raise(NotImplementedError)

        assert isinstance(tokenizer, PreTrainedTokenizer)
        
        # add special tokens
        special_tokens = []

        # add domains
        domains = definitions_cw.ALL_DOMAINS + ["general"]
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions_cw.DIALOG_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        # add slots
        slots = list(set(definitions_cw.ALL_INFSLOT + definitions_cw.ALL_REQSLOT))

        for slot in sorted(slots):
            token = "[value_" + slot + "]"
            special_tokens.append(token)

        special_tokens.extend(definitions_cw.SPECIAL_TOKENS)

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = self.tokenizer.tokenize(text)
        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]
            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]
            tokens = tokens + eos_token
        #todo ???????????????
        encoded_text = self.tokenizer.encode(tokens)
        if encoded_text[0] == self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]:
            encoded_text = encoded_text[1:] # remove [CLS]

        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def cls_token(self):
        return self.tokenizer.cls_token

    @property
    def eos_token(self):
        if self.backbone in ['fnlp/cpt-base', 'fnlp/cpt-large', "fnlp/bart-base-chinese", "fnlp/bart-large-chinese"]:
            return self.tokenizer.sep_token
        else:
            return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        if self.backbone in ['fnlp/cpt-base', 'fnlp/cpt-large', "fnlp/bart-base-chinese", "fnlp/bart-large-chinese"]:
            return self.tokenizer.sep_token_id
        else:
            return self.tokenizer.eos_token_id
        

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        if self.backbone in ['fnlp/cpt-base', 'fnlp/cpt-large', "fnlp/bart-base-chinese", "fnlp/bart-large-chinese"]:
            return 512
        else:
            return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

class CrossWOZReader(BaseReader):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.db = Database()
    def get_data_dir(self):
        return os.path.join('data','crosswoz', 'processed')

    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, '{}.json'.format(data_type)))
        
        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type):
            encoded_dial = []
            accum_constraint_dict = {}
            for t in dial['log']:
                turn_constrain_dict = self.bspn_to_constraint_dict(t['constraint'])
                for domain, sv_dict in turn_constrain_dict.items():
                    if domain not in accum_constraint_dict:
                        accum_constraint_dict[domain] = {}

                    for s,v in sv_dict.items():
                        if s not in accum_constraint_dict[domain]:
                            accum_constraint_dict[domain][s] = []
                        accum_constraint_dict[domain][s].append(v)

            for idx, t in enumerate(dial['log']):
                enc = {}
                enc['dial_id'] = fn
                enc['turn_num'] = t['turn_num']
                enc["turn_domain"] = t["turn_domain"].split()
                enc["pointer"] = [int(i) for i in t["pointer"].split(",")]

                user_ids = self.encode_text(t['user'], bos_token=definitions_cw.BOS_USER_TOKEN, eos_token=definitions_cw.EOS_USER_TOKEN)
                enc['user'] = user_ids

                usdx_ids = self.encode_text(t['user_delex'], bos_token=definitions_cw.BOS_USER_TOKEN,
                                            eos_token=definitions_cw.EOS_USER_TOKEN)

                resp_ids = self.encode_text(t['nodelx_resp'], bos_token=definitions_cw.BOS_RESP_TOKEN, eos_token=definitions_cw.EOS_RESP_TOKEN)
                enc['resp'] = resp_ids

                redx_ids = self.encode_text(t['resp'], bos_token=definitions_cw.BOS_RESP_TOKEN,
                                            eos_token=definitions_cw.EOS_RESP_TOKEN)
                enc['redx'] = redx_ids

                constraint_dict = self.bspn_to_constraint_dict(t["constraint"])
                ordered_constraint_dict = OrderedDict()
                for domain, slots in definitions_cw.INFORMABLE_SLOTS.items():
                    if domain not in constraint_dict:
                        continue

                    ordered_constraint_dict[domain] = OrderedDict()
                    for slot in slots:
                        if slot not in constraint_dict[domain]:
                            continue
                    
                        value = constraint_dict[domain][slot]
                        ordered_constraint_dict[domain][slot] = value

                ordered_bspn = self.constraint_dict_to_bspn(ordered_constraint_dict)
                bspn_ids = self.encode_text(ordered_bspn, bos_token=definitions_cw.BOS_BELIEF_TOKEN, eos_token=definitions_cw.EOS_BELIEF_TOKEN)
                enc['bspn'] = bspn_ids

                aspn_ids = self.encode_text(t['sys_act'], bos_token=definitions_cw.BOS_ACTION_TOKEN, eos_token=definitions_cw.EOS_ACTION_TOKEN)
                enc['aspn'] = aspn_ids

                pointer = enc["pointer"][:-2]

                if not any(pointer):
                    db_token = definitions_cw.DB_NULL_TOKEN
                else:
                    db_token = "[db_{}]".format(pointer.index(1))
                dbpn_ids = self.encode_text(db_token,
                                            bos_token=definitions_cw.BOS_DB_TOKEN,
                                            eos_token=definitions_cw.EOS_DB_TOKEN)

                enc["dbpn"] = dbpn_ids

                if (len(enc["user"]) == 0 or len(enc["resp"]) == 0 or
                        len(enc["redx"]) == 0 or len(enc["bspn"]) == 0 or
                        len(enc["aspn"]) == 0 or len(enc["dbpn"]) == 0):
                    raise ValueError(fn, idx)

                encoded_dial.append(enc)   
            encoded_data.append(encoded_dial)
        return encoded_data

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == definitions_cw.EOS_BELIEF_TOKEN:
                break

            if token.startswith('['):
                token = token[1:-1]

                if token in definitions_cw.ALL_DOMAINS:
                    domain = token

                elif token.startswith('value_'):
                    if domain is None:
                        continue

                    if domain not in constraint_dict:
                        constraint_dict[domain] = OrderedDict()

                    slot = token.split('_')[1]
                    constraint_dict[domain][slot] = []
            else:
                try:
                    if domain is not None and slot is not None:
                        constraint_dict[domain][slot].append(token)
                except KeyError:
                    continue

        for domain, sv_dict in constraint_dict.items():
            for s, value_tokens in sv_dict.items():
                constraint_dict[domain][s] = ' '.join(value_tokens)
        
        return constraint_dict

    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        vector = self.db.addDBIndicator(match_dom, match)

        return vector
        
if __name__ == '__main__':
    reader = CrossWOZReader('fnlp/cpt-base')
    #reader = CrossWOZReader('fnlp/bart-base-chinese')