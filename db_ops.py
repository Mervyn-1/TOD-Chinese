import os
import random
import json
import re
from collections import defaultdict

from utils import definitions_cw
from utils.io_utils import load_json, get_or_create_logger

logger = get_or_create_logger(__name__)

def contains(arr, s):
    return not len(tuple(filter(lambda item: (not (item.find(s) < 0)), arr)))


class Database(object):


    def __init__(self):
        super(Database, self).__init__()

        self.data = {}
        db_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),'./../data/crosswoz/database'))
        with open(os.path.join(db_dir, 'metro_db.json'), 'r', encoding='utf-8') as f:
            self.data['地铁'] = json.load(f)
        with open(os.path.join(db_dir, 'hotel_db.json'), 'r', encoding='utf-8') as f:
            self.data['酒店'] = json.load(f)
        with open(os.path.join(db_dir, 'restaurant_db.json'), 'r', encoding='utf-8') as f:
            self.data['餐馆'] = json.load(f)
        with open(os.path.join(db_dir, 'attraction_db.json'), 'r', encoding='utf-8') as f:
            self.data['景点'] = json.load(f)

        self.schema = {
            '景点': {
                '名称': {'params': None},
                '门票': {'type': 'between', 'params': [None, None]},
                '游玩时间': {'params': None},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None},
            },
            '餐馆': {
                '名称': {'params': None},
                '推荐菜': {'type': 'multiple_in', 'params': None},
                '人均消费': {'type': 'between', 'params': [None, None]},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None}
            },
            '酒店': {
                '名称': {'params': None},
                '酒店类型': {'params': None},
                '酒店设施': {'type': 'multiple_in', 'params': None},
                '价格': {'type': 'between', 'params': [None, None]},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None}
            },
            '地铁': {
                '起点': {'params': None},
                '终点': {'params': None},
            },
            '出租': {
                '起点': {'params': None},
                '终点': {'params': None},
            }
        }

    def one_Hot_Vector(self, domain, num):
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

    def addDBIndicator(self, domain, match_num, return_num=False):
        """Create database indicator for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        all_domains = ['餐馆', '景点', '酒店', '地铁', '出租']
        if domain in all_domains:
            vector = self.one_Hot_Vector(domain, match_num)
        else:
            vector = [0, 0, 0, 0,0,0]
        # '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]'
        if vector == [0, 0, 0, 0,0,0]:
            indicator = '[db_null]'
        else:
            indicator = '[db_%s]' % vector.index(1)
        return indicator

    def get_match_num(self, cons_dict):
        db = Database()
        match = {'general': ''}
        for domain in definitions_cw.ALL_DOMAINS:
            match[domain] = ''
            if domain in cons_dict:
                matched_ents = db.query(cons_dict, domain)
                match[domain] = len(matched_ents)
        return match

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

    def query(self, belief_state, cur_domain):
        """
        query database using belief state, return list of entities, same format as database
        :param belief_state: state['belief_state']
        :param cur_domain: maintain by DST, current query domain
        :return: list of entities
        """

        if not cur_domain:
            return []
        cur_query_form = {}
        for slot, value in belief_state[cur_domain].items():
            if '-' in value:
                value = value.replace(' ','')
                value = value.replace('-',' - ')
            else:
                value = value.replace(' ', '')
            if not value:
                continue
            if slot == '出发地':
                slot = '起点'
            elif slot == '目的地':
                slot = '终点'
            if slot == '名称':
                # DONE: if name is specified, ignore other constraints
                cur_query_form = {'名称': value}
                break
            elif slot == '评分':
                if re.match('(\d\.\d|\d)', value):
                    if re.match('\d\.\d', value):
                        score = float(re.match('\d\.\d', value)[0])
                    else:
                        score = int(re.match('\d', value)[0])
                    cur_query_form[slot] = [score, None]
                # else:
                #     assert 0, value
            elif slot in ['门票', '人均消费', '价格']:
                low, high = None, None
                if re.match('(\d+)-(\d+)', value):
                    low = int(re.match('(\d+)-(\d+)', value)[1])
                    high = int(re.match('(\d+)-(\d+)', value)[2])
                elif re.match('\d+', value):
                    if '以上' in value:
                        low = int(re.match('\d+', value)[0])
                    elif '以下' in value:
                        high = int(re.match('\d+', value)[0])
                    else:
                        low = high = int(re.match('\d+', value)[0])
                elif slot == '门票':
                    if value == '免费':
                        low = high = 0
                    elif value == '不免费':
                        low = 1
                    else:
                        print(value)
                        # assert 0
                cur_query_form[slot] = [low, high]
            else:
                cur_query_form[slot] = value
        cur_res = self.query_schema(field=cur_domain, args=cur_query_form)
        if cur_domain == '出租':
            res = [cur_res]
        elif cur_domain == '地铁':
            res = []
            for r in cur_res:
                if not res and '起点' in r[0]:
                    res.append(r)
                    break
            for r in cur_res:
                if '终点' in r[0]:
                    res.append(r)
                    break
        else:
            res = cur_res

        return res

    def query_schema(self, field, args):
        '''
        args: cur_query_form
        '''
        if not field in self.schema:
            return []
        if not isinstance(args, dict):
            raise Exception('`args` must be dict')
        db = self.data.get(field)
        plan = self.schema[field]
        for key, value in args.items():
            if not key in plan:
                return []
                #raise Exception('Unknown key %s' % key)
            value_type = plan[key].get('type')
            if value_type == 'between':
                if not value[0] is None:
                    plan[key]['params'][0] = float(value[0])
                if not value[1] is None:
                    plan[key]['params'][1] = float(value[1])
            else:
                if not isinstance(value, str):
                    raise Exception('Value for `%s` must be string' % key)
                plan[key]['params'] = value
        if field in ['地铁', '出租']:
            s = plan['起点']['params']
            e = plan['终点']['params']
            if not s or not e:
                return []
            if field == '出租':
                return [
                    '出租 (%s - %s)' % (s, e), {
                        '领域': '出租',
                        '车型': '#CX',
                        '车牌': '#CP'
                    }
                ]
            else:
                def func1(item):
                    if item[0].find(s) >= 0:
                        return ['(起点) %s' % item[0], item[1]]

                def func2(item):
                    if item[0].find(e) >= 0:
                        return ['(终点) %s' % item[0], item[1]]
                    return None

                return list(filter(lambda item: not item is None, list(map(func1, db)))) + list(
                    filter(lambda item: not item is None, list(map(func2, db))))

        def func3(item):
            details = item[1]
            for key, val in args.items():
                val = details.get(key)

                absence = val is None
                options = plan[key]
                if options.get('type') == 'between':
                    L = options['params'][0]
                    R = options['params'][1]
                    if not L is None:
                        if absence:
                            return False
                    else:
                        L = float('-inf')
                    if not R is None:
                        if absence:
                            return False
                    else:
                        R = float('inf')

                    if val is None:
                        return False

                    if L > val or val > R:
                        return False
                elif options.get('type') == 'in':
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if contains(val, s):
                            return False
                elif options.get('type') == 'multiple_in':
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        sarr = list(filter(lambda t: bool(t), s.split(' ')))
                        if len(list(filter(lambda t: contains(val, t), sarr))):
                            return False
                else:
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if val.find(s) < 0:
                            return False
            return True

        return list(filter(func3, db))