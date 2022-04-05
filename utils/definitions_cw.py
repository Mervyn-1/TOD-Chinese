ALL_DOMAINS = ["餐馆", "酒店", "景点", "出租", "地铁"]

INFORMABLE_SLOTS = {
    "餐馆": ['名称', '营业时间', '评分', '人均消费', '推荐菜',  '电话', '地址','周边餐馆', '周边景点', '周边酒店'],
    "酒店": ["名称", "评分", "价格", "酒店类型", "酒店设施",'电话', '地址',"周边酒店", "周边景点", "周边餐馆"],
    "景点": ['门票', '名称', '评分', '游玩时间', '电话', '地址', '周边景点', '周边酒店', '周边餐馆'],
    "出租": ['出发地', '目的地', '车型', '车牌'],
    "地铁": ['出发地', '目的地', '出发地附近地铁站', '目的地附近地铁站'],
}

ALL_INFSLOT = ['名称', '营业时间', '评分', '人均消费', '推荐菜', '周边餐馆', '电话', '地址', '周边景点', '周边酒店',
               "价格", "酒店类型", "酒店设施",'门票', '游玩时间','出发地', '目的地', '车型', '车牌','出发地附近地铁站', '目的地附近地铁站' ]

REQUESTABLE_SLOTS = {
    "餐馆": ['名称', '营业时间', '评分', '人均消费', '推荐菜',  '电话', '地址','周边餐馆', '周边景点', '周边酒店'],
    "酒店": ["名称", "评分", "价格", "酒店类型", "酒店设施",'电话', '地址',"周边酒店", "周边景点", "周边餐馆"],
    "景点": ['门票', '名称', '评分', '游玩时间', '电话', '地址', '周边景点', '周边酒店', '周边餐馆'],
    "出租": ['出发地', '目的地', '车型', '车牌'],
    "地铁": ['出发地', '目的地', '出发地附近地铁站', '目的地附近地铁站'],
}

ALL_REQSLOT = ['名称', '营业时间', '评分', '人均消费', '推荐菜', '周边餐馆', '电话', '地址', '周边景点', '周边酒店',
               "价格", "酒店类型", "酒店设施",'门票', '游玩时间','车型', '车牌','出发地附近地铁站', '目的地附近地铁站']

DIALOG_ACTS = {
    "餐馆": ['inform', 'request', 'nooffer', 'recommend', 'select'],
    "酒店": ['inform', 'request', 'nooffer', 'recommend', 'select'],
    "景点": ['inform', 'request', 'nooffer', 'recommend', 'select'],
    "出租": ['inform', 'request'],
    "地铁": ['inform', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome','thank',],
}

EXTRACTIVE_SLOT = []


BOS_USER_TOKEN = "<bos_user>"
EOS_USER_TOKEN = "<eos_user>"

USER_TOKENS = [BOS_USER_TOKEN, EOS_USER_TOKEN]

BOS_BELIEF_TOKEN = "<bos_belief>"
EOS_BELIEF_TOKEN = "<eos_belief>"

BELIEF_TOKENS = [BOS_BELIEF_TOKEN, EOS_BELIEF_TOKEN]

BOS_DB_TOKEN = "<bos_db>"
EOS_DB_TOKEN = "<eos_db>"

DB_TOKENS = [BOS_DB_TOKEN, EOS_DB_TOKEN]

BOS_ACTION_TOKEN = "<bos_act>"
EOS_ACTION_TOKEN = "<eos_act>"

BOS_USR_ACTION_TOKEN = "<bos_usr_act>"
EOS_USR_ACTION_TOKEN = "<eos_usr_act>"

ACTION_TOKENS = [BOS_ACTION_TOKEN, EOS_ACTION_TOKEN, BOS_USR_ACTION_TOKEN, EOS_USR_ACTION_TOKEN]

BOS_RESP_TOKEN = "<bos_resp>"
EOS_RESP_TOKEN = "<eos_resp>"

RESP_TOKENS = [BOS_RESP_TOKEN, EOS_RESP_TOKEN]

DB_NULL_TOKEN = "[db_null]"
DB_0_TOKEN = "[db_0]"
DB_1_TOKEN = "[db_1]"
DB_2_TOKEN = "[db_2]"
DB_3_TOKEN = "[db_3]"

DB_STATE_TOKENS = [DB_NULL_TOKEN, DB_0_TOKEN, DB_1_TOKEN, DB_2_TOKEN, DB_3_TOKEN]

SPECIAL_TOKENS = USER_TOKENS + BELIEF_TOKENS + DB_TOKENS + ACTION_TOKENS + RESP_TOKENS + DB_STATE_TOKENS
