import json
import sys
import os
import re
import random
import torch
import gc
import argparse
import datetime
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer

try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:
    print("Warning: auto_gptq is not installed. This is required to calculate probabilities.")

# Constants for text manipulation operations
OP_RANDOM_INSERTION = "RI"
OP_RANDOM_DELETION = "RD"
OP_SYNONYM_REPLACEMENT = "SR"
PERCENTAGES = [5, 10, 15, 20]


def filter_json_by_max_prob(input_file):
    """
    Process JSON file, keeping only the item with the highest prob value for each origin

    :param input_file: Input JSON file path
    :return: Processed data and output file path
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist!")
            return None, None

        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create a dictionary to store the item with max prob for each origin
        origin_max_prob = {}

        # Find the max prob item for each origin
        for item in data:
            origin = item['origin']
            prob = item['prob']

            # If this origin hasn't been recorded yet, or current prob is higher than recorded prob
            if origin not in origin_max_prob or prob > origin_max_prob[origin]['prob']:
                origin_max_prob[origin] = item

        # Convert results back to a list
        result = list(origin_max_prob.values())

        # Create output directory
        output_dir = os.path.join(os.getcwd(), "handled_method")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Get input filename (without path)
        input_filename = os.path.basename(input_file)

        # Create output file path
        output_file = os.path.join(output_dir, input_filename)

        # Write results to new file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"Processing complete: filtered {len(result)} items from {len(data)} items")
        print(f"Results saved to: {output_file}")

        return result, output_file

    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file!")
        return None, None
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None, None


def load_and_filter_sentences(input_file, max_sentences=10):
    """
    Load JSON, filter by max prob, and take up to max_sentences
    The first half are English, the second half are Chinese

    :param input_file: Input JSON file path
    :param max_sentences: Maximum number of sentences to process
    :return: Filtered data
    """
    filtered_data, _ = filter_json_by_max_prob(input_file)

    if filtered_data is None or len(filtered_data) == 0:
        print("No data found after filtering!")
        return None

    # Limit to max_sentences
    return filtered_data[:max_sentences] if len(filtered_data) > max_sentences else filtered_data


def clean_adv_text(text):
    """
    Remove <xxxx> tags from the beginning of text

    :param text: Text to clean
    :return: Cleaned text
    """
    if not text:
        return ""
    return re.sub(r'^<[^>]+>', '', text).strip()


def is_chinese(text):
    """
    Check if text contains Chinese characters

    :param text: Text to check
    :return: True if text contains Chinese characters
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def categorize_sentences(sentences):
    """
    Categorize sentences into English and Chinese

    :param sentences: List of sentence items
    :return: Dictionary with 'en' and 'zh' keys containing categorized sentences
    """
    categorized = {
        'en': [],
        'zh': []
    }

    for item in sentences:
        text = item.get('adv', item.get('sentence', item.get('origin', '')))
        text = clean_adv_text(text)

        if is_chinese(text):
            categorized['zh'].append(item)
        else:
            categorized['en'].append(item)

    return categorized


# Text manipulation operations

def random_insertion(words, n):
    """
    Randomly insert n words into the sentence

    :param words: List of words in the sentence
    :param n: Number of words to insert
    :return: Modified list of words
    """
    result = words.copy()

    # If the word list is empty, return as is
    if not result:
        return result

    # Create a list of words to insert
    # For simplicity, we'll use random words from the original text
    insert_words = random.choices(result, k=n) if result else ["the", "a", "an", "this", "that"]

    for _ in range(n):
        # Don't insert at the very beginning to avoid affecting sentence structure too much
        position = random.randint(1, len(result))
        word = random.choice(insert_words)
        result.insert(position, word)

    return result


def random_deletion(words, p):
    """
    Randomly delete words from the sentence with probability p

    :param words: List of words in the sentence
    :param p: Probability of deletion for each word
    :return: Modified list of words
    """
    # If p is 1.0, delete all words - but that would make the sentence empty
    # So we'll keep at least one word
    if p == 1.0:
        return [random.choice(words)] if words else []

    result = []
    for word in words:
        # Skip word with probability p
        if random.random() >= p:
            result.append(word)

    # Ensure we keep at least one word
    if not result and words:
        result = [random.choice(words)]

    return result


# For synonym replacement, we'll use a comprehensive approach with predefined synonyms
# Based on provided sample sentences and common words
english_synonyms = {
    # Common adjectives
    "good": ["nice", "excellent", "great", "wonderful", "fine", "proper", "suitable"],
    "bad": ["poor", "terrible", "awful", "unpleasant", "negative", "inadequate"],
    "big": ["large", "huge", "enormous", "gigantic", "grand", "great", "substantial"],
    "small": ["tiny", "little", "miniature", "petite", "minute", "slight", "minor"],
    "happy": ["joyful", "pleased", "delighted", "glad", "cheerful", "content"],
    "sad": ["unhappy", "sorrowful", "depressed", "gloomy", "melancholy", "downcast"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty", "expeditious"],
    "slow": ["sluggish", "gradual", "unhurried", "leisurely", "deliberate", "gentle"],
    "beautiful": ["pretty", "lovely", "gorgeous", "attractive", "exquisite", "delicate", "fine"],
    "ugly": ["unattractive", "hideous", "unsightly", "homely", "unpleasant", "grotesque"],
    "important": ["significant", "vital", "essential", "crucial", "substantial", "considerable"],
    "cold": ["chilly", "cool", "frigid", "frosty", "icy", "freezing"],
    "warm": ["hot", "heated", "cozy", "tepid", "mild", "snug"],
    "old": ["ancient", "aged", "elderly", "vintage", "antique", "aged"],
    "new": ["fresh", "recent", "novel", "modern", "current", "latest"],
    "primitive": ["ancient", "primal", "primeval", "original", "early", "rudimentary"],
    "necessary": ["essential", "required", "vital", "needed", "fundamental", "imperative"],
    "rich": ["wealthy", "affluent", "prosperous", "opulent", "luxurious", "abundant"],
    "poor": ["impoverished", "destitute", "needy", "indigent", "meager", "inadequate"],
    "wise": ["intelligent", "smart", "sage", "learned", "knowledgeable", "prudent"],
    "simple": ["plain", "basic", "uncomplicated", "straightforward", "easy", "modest"],
    "outward": ["external", "outer", "exterior", "apparent", "visible", "superficial"],
    "inward": ["internal", "inner", "interior", "intrinsic", "deep", "personal"],
    "true": ["genuine", "authentic", "real", "accurate", "correct", "faithful"],

    # Common nouns
    "life": ["existence", "living", "being", "lifetime", "animation"],
    "work": ["labor", "toil", "effort", "occupation", "employment", "task"],
    "nature": ["environment", "world", "creation", "universe", "earth", "outdoors"],
    "time": ["period", "era", "age", "moment", "instance", "occasion"],
    "fact": ["reality", "truth", "actuality", "certainty", "information"],
    "change": ["alteration", "modification", "transformation", "variation", "shift"],
    "knowledge": ["understanding", "wisdom", "intelligence", "learning", "education"],
    "basis": ["foundation", "ground", "footing", "groundwork", "support"],
    "moment": ["instant", "second", "minute", "time", "flash", "jiffy"],
    "trouble": ["difficulty", "problem", "hardship", "distress", "adversity"],
    "anxiety": ["worry", "concern", "apprehension", "unease", "nervousness"],
    "strength": ["power", "might", "force", "vigor", "potency", "energy"],
    "weakness": ["frailty", "feebleness", "fragility", "vulnerability", "infirmity"],
    "food": ["nourishment", "sustenance", "nutrition", "provisions", "fare", "meals"],
    "water": ["liquid", "fluid", "moisture", "hydration", "drink"],
    "shelter": ["refuge", "protection", "cover", "housing", "lodging", "home"],
    "clothing": ["attire", "garments", "apparel", "dress", "outfit", "wear"],
    "bed": ["couch", "mattress", "bunk", "cot", "resting place"],
    "world": ["earth", "globe", "planet", "realm", "society", "universe"],
    "care": ["attention", "concern", "supervision", "charge", "protection"],

    # Common verbs
    "think": ["believe", "consider", "contemplate", "ponder", "reflect"],
    "know": ["understand", "recognize", "comprehend", "perceive", "grasp"],
    "live": ["exist", "survive", "dwell", "reside", "subsist"],
    "say": ["state", "declare", "express", "mention", "utter", "speak"],
    "do": ["act", "perform", "execute", "accomplish", "achieve", "complete"],
    "see": ["observe", "view", "witness", "perceive", "notice", "behold"],
    "go": ["move", "proceed", "advance", "travel", "journey", "depart"],
    "get": ["obtain", "acquire", "gain", "procure", "secure", "receive"],
    "make": ["create", "produce", "form", "construct", "build", "generate"],
    "take": ["grab", "seize", "grasp", "collect", "obtain", "acquire"],
    "come": ["approach", "arrive", "reach", "appear", "emerge", "materialize"],
    "use": ["employ", "utilize", "apply", "implement", "exercise", "handle"],
    "find": ["discover", "locate", "uncover", "detect", "encounter", "come across"],
    "look": ["see", "view", "observe", "watch", "examine", "inspect"],
    "want": ["desire", "wish", "crave", "need", "require", "long for"],
    "give": ["provide", "supply", "furnish", "deliver", "present", "offer"],
    "refer": ["mention", "allude", "cite", "point to", "direct", "indicate"],
    "keep": ["maintain", "retain", "preserve", "continue", "sustain", "hold"],
    "mean": ["signify", "indicate", "denote", "represent", "imply", "suggest"],
    "consider": ["think about", "contemplate", "reflect on", "ponder", "examine"],

    # Common prepositions and articles
    "the": ["a", "this", "that", "these", "those"],
    "a": ["the", "this", "that", "one", "any"],
    "in": ["within", "inside", "during", "amid", "throughout"],
    "on": ["upon", "atop", "over", "above"],
    "to": ["toward", "for", "into", "unto", "till"],
    "of": ["from", "concerning", "regarding", "about"],
    "at": ["in", "by", "near", "around", "beside"],
    "by": ["through", "via", "with", "using", "alongside"],
    "for": ["to", "toward", "regarding", "concerning", "about"],
    "with": ["using", "employing", "by means of", "alongside"],
    "about": ["regarding", "concerning", "on", "approximately"],

    # Common adverbs
    "not": ["never", "hardly", "scarcely", "barely"],
    "very": ["extremely", "exceedingly", "tremendously", "immensely", "highly"],
    "so": ["thus", "therefore", "consequently", "hence", "accordingly"],
    "more": ["additionally", "further", "extra", "added", "supplementary"],
    "most": ["chiefly", "predominantly", "mainly", "largely", "primarily"],
    "too": ["excessively", "overly", "unduly", "extremely"],
    "then": ["afterwards", "subsequently", "next", "later", "thereafter"],
    "also": ["likewise", "too", "as well", "besides", "furthermore"],
    "just": ["merely", "simply", "only", "barely", "hardly"],
    "now": ["presently", "currently", "at this moment", "at present"],
    "here": ["at this place", "at this point", "in this location"],
    "there": ["at that place", "in that location", "at that point"],
    "only": ["merely", "just", "solely", "exclusively", "simply"],
    "well": ["thoroughly", "properly", "suitably", "effectively", "skillfully"],

    # From sample texts
    "safely": ["securely", "confidently", "reliably", "without risk"],
    "trust": ["rely on", "believe in", "have faith in", "depend on", "count on"],
    "deal": ["amount", "portion", "quantity", "measure", "extent"],
    "waive": ["relinquish", "forgo", "abandon", "surrender", "give up"],
    "bestow": ["give", "grant", "confer", "present", "provide"],
    "elsewhere": ["somewhere else", "in another place", "in other places"],
    "adapted": ["suited", "adjusted", "modified", "tailored", "customized"],
    "incessant": ["constant", "continuous", "unending", "persistent", "relentless"],
    "strain": ["pressure", "stress", "tension", "burden", "exertion"],
    "incurable": ["terminal", "irreversible", "untreatable", "permanent", "chronic"],
    "disease": ["illness", "ailment", "sickness", "malady", "condition"],
    "exaggerate": ["overstate", "magnify", "inflate", "amplify", "embellish"],
    "vigilant": ["watchful", "alert", "observant", "attentive", "cautious"],
    "thoroughly": ["completely", "entirely", "fully", "wholly", "utterly"],
    "sincerely": ["genuinely", "honestly", "truly", "authentically", "earnestly"],
    "compelled": ["forced", "obliged", "required", "constrained", "pressured"],
    "reverencing": ["respecting", "honoring", "venerating", "esteeming", "admiring"],
    "denying": ["rejecting", "refusing", "disavowing", "negating", "repudiating"],
    "possibility": ["likelihood", "prospect", "chance", "potential", "feasibility"],
    "radii": ["rays", "lines", "spokes", "beams"],
    "miracle": ["wonder", "marvel", "phenomenon", "prodigy", "amazement"],
    "contemplate": ["consider", "ponder", "reflect on", "meditate on", "think about"],
    "reduced": ["decreased", "lessened", "diminished", "lowered", "minimized"],
    "imagination": ["creativity", "fantasy", "vision", "conception", "fancy"],
    "foresee": ["predict", "anticipate", "forecast", "envision", "expect"],
    "establish": ["set up", "create", "found", "institute", "construct"],
    "primitive": ["basic", "simple", "elementary", "rudimentary", "primal"],
    "frontier": ["border", "boundary", "edge", "margin", "limit"],
    "civilization": ["culture", "society", "nation", "community", "advancement"],
    "midst": ["middle", "center", "heart", "core", "thick"],
    "gross": ["basic", "fundamental", "essential", "primary", "key"],
    "necessaries": ["essentials", "requirements", "needs", "basics", "fundamentals"],
    "commonly": ["usually", "generally", "typically", "normally", "ordinarily"],
    "exertions": ["efforts", "endeavors", "labors", "struggles", "attempts"],
    "savageness": ["wildness", "ferocity", "brutality", "fierceness", "cruelty"],
    "philosophy": ["wisdom", "thinking", "reasoning", "thought", "doctrine"],
    "palatable": ["tasty", "appetizing", "flavorful", "delicious", "pleasant"],
    "prairie": ["grassland", "plain", "meadow", "field", "pasture"],
    "forest": ["woods", "woodland", "grove", "timberland", "jungle"],
    "shadow": ["shade", "darkness", "gloom", "obscurity", "dimness"],
    "grand": ["great", "large", "magnificent", "splendid", "impressive"],
    "vital": ["essential", "crucial", "critical", "important", "necessary"],
    "heat": ["warmth", "temperature", "hotness", "fervor", "intensity"],
    "accordingly": ["therefore", "consequently", "thus", "hence", "so"],
    "pains": ["efforts", "troubles", "labors", "struggles", "difficulties"],
    "robbing": ["stealing from", "taking from", "plundering", "looting", "pillaging"],
    "burrow": ["hole", "tunnel", "den", "lair", "hollow"],
    "wont": ["accustomed", "used", "habituated", "prone", "inclined"],
    "complain": ["grumble", "protest", "object", "lament", "gripe"],
    "refer": ["attribute", "ascribe", "assign", "credit", "relate"],
    "directly": ["immediately", "straight", "precisely", "exactly", "right"],
    "ails": ["troubles", "afflictions", "problems", "difficulties", "maladies"]
}

chinese_synonyms = {
    # Common adjectives
    "好": ["优秀", "良好", "不错", "精彩", "优质", "出色"],
    "坏": ["糟糕", "不好", "差劲", "糟", "恶劣", "劣质"],
    "大": ["巨大", "宏大", "庞大", "硕大", "广阔", "辽阔"],
    "小": ["微小", "细小", "渺小", "迷你", "微型", "袖珍"],
    "快乐": ["开心", "高兴", "喜悦", "愉快", "欢乐", "欣喜"],
    "悲伤": ["难过", "伤心", "忧伤", "痛苦", "哀伤", "忧愁"],
    "快": ["迅速", "敏捷", "迅捷", "急速", "飞快", "疾速"],
    "慢": ["缓慢", "迟缓", "缓慢", "怠慢", "迟钝", "慢吞吞"],
    "美丽": ["漂亮", "好看", "秀丽", "俊俏", "美观", "动人"],
    "丑陋": ["难看", "丑", "不堪入目", "奇丑", "丑恶", "丑陋不堪"],
    "聪明": ["智慧", "聪颖", "明智", "机智", "睿智", "灵巧"],
    "温暖": ["热乎", "暖和", "暖心", "温馨", "舒适", "宜人"],
    "冷": ["寒冷", "凉", "冰冷", "寒", "凛冽", "冷酷"],
    "简朴": ["简单", "朴素", "简约", "简洁", "朴实", "素净"],
    "贫乏": ["匮乏", "缺乏", "不足", "贫瘠", "稀少", "有限"],
    "丰富": ["充足", "富饶", "繁多", "众多", "大量", "充裕"],
    "重要": ["关键", "主要", "核心", "关键性", "主要的", "首要"],
    "必需": ["必要", "不可或缺", "必不可少", "必备", "基本", "根本"],

    # Common nouns
    "奢侈品": ["豪华物品", "奢华品", "高档产品", "高价商品", "高端物品"],
    "障碍": ["阻碍", "阻碍物", "阻挡", "妨碍", "阻障", "障碍物"],
    "哲学家": ["思想家", "智者", "哲人", "大师", "理学家"],
    "阶层": ["阶级", "等级", "层次", "社会阶层", "群体", "阶段"],
    "财富": ["财产", "财宝", "资产", "财物", "钱财", "金钱"],
    "食物": ["食品", "餐食", "饮食", "食材", "伙食", "食粮"],
    "房子": ["住宅", "房屋", "住所", "居所", "宅邸", "家宅"],
    "衣服": ["服装", "衣物", "服饰", "衣着", "装束", "衣裳"],
    "温暖": ["暖和", "热量", "温热", "暖意", "热气", "暖流"],
    "必需品": ["生活必需品", "必备品", "必备物品", "基本物品", "必要物品"],
    "土壤": ["泥土", "土地", "田地", "地面", "地土", "泥壤"],
    "种子": ["籽", "种籽", "谷种", "胚胎", "幼苗", "种苗"],
    "芽": ["幼芽", "嫩芽", "新芽", "萌芽", "胚芽", "芽苞"],
    "永恒": ["永久", "恒久", "亘古", "长存", "永存", "长久"],
    "行业": ["产业", "工作", "职业", "专业", "生意", "事业"],
    "秘密": ["隐秘", "秘事", "隐私", "隐情", "机密", "秘闻"],
    "本质": ["实质", "性质", "根本", "特质", "真谛", "真相"],
    "牲畜": ["家畜", "牛羊", "畜生", "家禽", "动物", "牧畜"],
    "栅栏": ["篱笆", "围栏", "栏杆", "篱笆墙", "围墙", "栏栅"],
    "牧人": ["牧民", "牧羊人", "牧场主", "放牧人", "牧童", "畜牧人"],
    "农场": ["农庄", "农田", "田园", "农地", "庄园", "农屋"],
    "角落": ["墙角", "拐角", "边角", "一隅", "角点", "边缘"],
    "田": ["田地", "田野", "原野", "农田", "地块", "田园"],
    "树": ["树木", "树种", "植物", "树株", "树干", "枝桠"],
    "季节": ["季", "时节", "时期", "节候", "节气", "节令"],
    "国家": ["国", "国度", "国土", "疆土", "邦国", "国土"],
    "艺术": ["艺", "技艺", "美术", "工艺", "艺能", "风艺"],
    "化装舞会": ["面具舞会", "假面舞会", "装扮舞会", "化妆舞会", "装扮派对"],
    "时尚": ["流行", "风尚", "潮流", "时髦", "流行趋势", "时装"],
    "服装": ["衣服", "服饰", "衣物", "衣着", "着装", "服装打扮"],
    "国王": ["君王", "王", "帝王", "王者", "君主", "帝"],
    "王后": ["皇后", "皇妃", "王妃", "皇太后", "王太后", "王妃"],

    # Common verbs
    "穿": ["着", "穿着", "披", "套", "戴", "佩戴"],
    "说": ["讲", "谈", "述", "表达", "阐述", "道"],
    "认为": ["觉得", "以为", "看作", "视为", "以...为", "认作"],
    "进步": ["发展", "前进", "前行", "进展", "成长", "改进"],
    "生活": ["过活", "过日子", "生存", "居住", "存活", "生计"],
    "得到": ["获得", "取得", "收获", "赢得", "获取", "博得"],
    "解脱": ["摆脱", "脱离", "解除", "摆脱", "远离", "脱出"],
    "发出": ["释放", "散发", "放出", "发射", "产生", "传出"],
    "改善": ["提高", "优化", "增进", "完善", "提升", "促进"],
    "记录": ["记载", "书写", "录入", "记录下", "写下", "记下"],
    "站": ["立", "站立", "处于", "位于", "占据", "处在"],
    "踮": ["踮起", "抬起", "提起", "直立", "挺起", "伸展"],
    "原谅": ["宽恕", "饶恕", "谅解", "体谅", "包容", "谅宥"],
    "含糊其辞": ["说得含混", "说话不明确", "言辞含糊", "模棱两可", "不清不楚"],
    "保守": ["保持", "守护", "维持", "守住", "保存", "坚守"],
    "照看": ["照顾", "照料", "顾及", "关照", "照应", "护理"],
    "跳过": ["越过", "跨越", "迈过", "超越", "跃过", "越过"],
    "带来": ["产生", "导致", "引起", "引发", "引致", "造成"],
    "留意": ["注意", "注目", "关注", "关心", "注目", "留心"],
    "干活": ["工作", "劳动", "做事", "忙碌", "劳作", "操劳"],
    "浇灌": ["灌溉", "浇水", "浇注", "滋润", "润泽", "灌水"],
    "枯萎": ["凋谢", "干枯", "萎缩", "枯干", "枯死", "凋零"],
    "上升": ["升高", "提高", "攀升", "上涨", "提升", "升起"],
    "凑合": ["将就", "勉强", "凑合着", "对付", "应付", "糊弄"],
    "嘲笑": ["讥笑", "取笑", "奚落", "讽刺", "调侃", "揶揄"],
    "追随": ["跟随", "跟从", "追随", "效仿", "模仿", "仿效"],

    # Common adverbs and prepositions
    "不": ["没", "不要", "没有", "无", "非", "莫"],
    "很": ["非常", "十分", "极为", "极其", "格外", "分外"],
    "的": ["之", "得", "地", "所", "者", "自"],
    "是": ["为", "乃", "成为", "即", "就是", "等于"],
    "在": ["于", "位于", "处于", "居于", "存在于", "位于"],
    "和": ["与", "同", "及", "跟", "并", "以及"],
    "了": ["啦", "过", "已", "已经", "曾经", "完成"],
    "对": ["向", "朝", "朝向", "对于", "针对", "面对"],
    "从": ["自", "由", "始自", "起自", "从事", "经由"],
    "为": ["为了", "是为", "因为", "为此", "成为", "作为"],
    "而": ["并且", "而且", "然而", "可是", "但是", "不过"],
    "反而": ["却", "相反", "反倒", "倒是", "相反地", "对立地"],
    "这样": ["如此", "这般", "这么", "这种", "这类", "如此这般"],
    "什么": ["啥", "何物", "何事", "何", "如何", "什么样"],
    "每个": ["各个", "各", "各种", "各类", "每", "所有"],
    "也": ["还", "亦", "同样", "同时", "一样", "仍然"],
    "现在": ["此刻", "眼下", "如今", "目前", "当下", "当前"],
    "总的来说": ["总体而言", "整体上", "归根结底", "总而言之", "综合来看"],
    "就像": ["好像", "如同", "犹如", "宛如", "恰似", "好似"],

    # From sample texts
    "奢侈": ["豪华", "奢华", "奢侈浪费", "过度消费", "铺张", "浪费"],
    "舒适": ["安逸", "适意", "舒服", "舒心", "惬意", "宜人"],
    "不可或缺": ["必不可少", "必需", "必要", "不可缺", "必备", "必须"],
    "积极": ["主动", "热心", "热情", "积极性", "努力", "进取"],
    "简朴": ["简单", "朴素", "简约", "质朴", "简洁", "朴实"],
    "微薄": ["微小", "少量", "稀少", "不足", "微弱", "微量"],
    "古代": ["远古", "上古", "古时", "古代社会", "古时候", "远古时期"],
    "丰富": ["充实", "充足", "丰厚", "富足", "丰盈", "富有"],
    "贫乏": ["缺乏", "不足", "匮乏", "贫瘠", "稀少", "欠缺"],
    "温暖": ["暖和", "温热", "暖意", "热度", "暖流", "温煦"],
    "华丽": ["豪华", "富丽", "绚丽", "奢华", "壮丽", "辉煌"],
    "精致": ["精美", "精巧", "精工", "精良", "讲究", "细致"],
    "冒险": ["冒险生活", "冒险活动", "探险", "冒险体验", "险境", "冒难"],
    "劳作": ["劳动", "工作", "辛劳", "操劳", "勤劳", "劳苦"],
    "卑微": ["微不足道", "渺小", "微贱", "低微", "低贱", "下贱"],
    "急于": ["迫切", "急切", "热切", "迫切想要", "急迫", "急不可待"],
    "紧迫性": ["紧急性", "急迫性", "迫切性", "紧急程度", "紧急情况", "紧迫感"],
    "踮起脚尖": ["踮脚", "抬高", "举起", "提起", "挺直", "抬头"],
    "禁止入内": ["不准进入", "勿入", "严禁入内", "闲人免进"]
}


def synonym_replacement(words, n, is_chinese_text=False):
    """
    Replace n words with their synonyms

    :param words: List of words in the sentence
    :param n: Number of words to replace
    :param is_chinese_text: Whether the text is Chinese
    :return: Modified list of words
    """
    result = words.copy()

    # If list is empty or n is 0, return as is
    if not result or n <= 0:
        return result

    synonyms_dict = chinese_synonyms if is_chinese_text else english_synonyms

    # Get indices of words that have synonyms
    indices_with_synonyms = [i for i, word in enumerate(result) if word.lower() in synonyms_dict]

    # If no words have synonyms, return original
    if not indices_with_synonyms:
        return result

    # Randomly select n indices (or fewer if not enough available)
    num_to_replace = min(n, len(indices_with_synonyms))
    indices_to_replace = random.sample(indices_with_synonyms, num_to_replace)

    # Replace selected words with synonyms
    for idx in indices_to_replace:
        word = result[idx].lower()
        if word in synonyms_dict:
            synonyms = synonyms_dict[word]
            result[idx] = random.choice(synonyms)

    return result


def apply_text_manipulation(text, operation, percentage, is_chinese=False):
    """
    Apply text manipulation operation to text at specified percentage

    :param text: Input text
    :param operation: Operation type (RI, RD, SR)
    :param percentage: Percentage of manipulation
    :param is_chinese: Whether the text is Chinese
    :return: Manipulated text
    """
    if not text:
        return text

    # Split text into words
    if is_chinese:
        # For Chinese, treat each character as a "word"
        words = list(text)
    else:
        # For English, split by spaces
        words = text.split()

    total_words = len(words)
    if total_words == 0:
        return text

    # Calculate number of words to manipulate
    n = max(1, int(total_words * percentage / 100))

    # Apply operation
    if operation == OP_RANDOM_INSERTION:
        modified_words = random_insertion(words, n)
    elif operation == OP_RANDOM_DELETION:
        modified_words = random_deletion(words, percentage / 100)
    elif operation == OP_SYNONYM_REPLACEMENT:
        modified_words = synonym_replacement(words, n, is_chinese)
    else:
        return text

    # Reconstruct text
    if is_chinese:
        return ''.join(modified_words)
    else:
        return ' '.join(modified_words)


def calculate_probability(model, tokenizer, text, device="cuda:0"):
    """
    Calculate the probability of the end token given the input text.

    :param model: Language model
    :param tokenizer: Tokenizer
    :param text: Input text
    :param device: Device to use for calculation
    :return: End token probability
    """
    tokens = torch.tensor(tokenizer.encode(text)).to(device)

    eos_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        logits = model(tokens.unsqueeze(0)).logits
        # Get probabilities for the last position
        probs = torch.nn.functional.softmax(logits[0, -1, :], dim=0)
        # Extract end token probability
        end_token_prob = float(probs[eos_token_id])

    return end_token_prob


def generate_output_filename(input_file, model_path):
    """
    Generate output filename based on input file and model path
    """
    # Extract base name from input file
    input_basename = os.path.basename(input_file)
    base_name = os.path.splitext(input_basename)[0]

    # Extract model name from path
    model_name = os.path.basename(os.path.normpath(model_path))

    # Get current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    return f"{base_name}_{model_name}_robustness_{current_date}.json"


def main():
    parser = argparse.ArgumentParser(description="Text robustness and probability analysis")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the language model")
    parser.add_argument("--max_sentences", type=int, default=10, help="Maximum number of sentences to process")
    parser.add_argument("--output_dir", type=str, default="robustness_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, etc.)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Load and filter sentences
    sentences = load_and_filter_sentences(args.input_file, args.max_sentences)
    if not sentences:
        print("No valid sentences found. Exiting.")
        sys.exit(1)

    # Categorize sentences into English and Chinese
    categorized_sentences = categorize_sentences(sentences)

    print(
        f"Found {len(categorized_sentences['en'])} English sentences and {len(categorized_sentences['zh'])} Chinese sentences")

    # Load model and tokenizer for probability calculations
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantize_config=None
    ).to(args.device)

    # Operations and percentages to apply
    operations = [OP_RANDOM_INSERTION, OP_RANDOM_DELETION, OP_SYNONYM_REPLACEMENT]

    # Results will contain all manipulated texts and their probabilities
    results = []

    # Statistics for average probabilities per operation and language
    stats = {
        'en': {op: {p: [] for p in PERCENTAGES} for op in operations},
        'zh': {op: {p: [] for p in PERCENTAGES} for op in operations}
    }

    # Process each language category
    for lang, lang_sentences in categorized_sentences.items():
        is_chinese = lang == 'zh'
        print(f"Processing {lang} sentences...")

        for item in tqdm(lang_sentences):
            original_text = item.get('adv', item.get('sentence', item.get('origin', '')))
            original_text = clean_adv_text(original_text)

            # Calculate original probability
            original_prob = calculate_probability(model, tokenizer, original_text, args.device)

            # Create a base result item with original data
            base_result = {
                'original': item,
                'cleaned_text': original_text,
                'original_prob': original_prob,
                'language': lang,
                'augmentations': []
            }

            # Apply each operation at each percentage
            for op in operations:
                for percentage in PERCENTAGES:
                    augmented_text = apply_text_manipulation(
                        original_text, op, percentage, is_chinese
                    )

                    # Calculate probability for augmented text
                    prob = calculate_probability(model, tokenizer, augmented_text, args.device)

                    # Add to statistics
                    stats[lang][op][percentage].append(prob)

                    # Add to result
                    augmentation = {
                        'operation': op,
                        'percentage': percentage,
                        'text': augmented_text,
                        'prob': prob
                    }
                    base_result['augmentations'].append(augmentation)

                    print(f"  {lang} - {op} {percentage}%: prob={prob:.6f}")

                    # Clean up GPU memory
                    gc.collect()
                    torch.cuda.empty_cache()

            # Add full results for this sentence
            results.append(base_result)

    # Calculate average probabilities for each operation and language
    averages = {
        'en': {op: {p: np.mean(stats['en'][op][p]) if stats['en'][op][p] else 0 for p in PERCENTAGES} for op in
               operations},
        'zh': {op: {p: np.mean(stats['zh'][op][p]) if stats['zh'][op][p] else 0 for p in PERCENTAGES} for op in
               operations}
    }

    # Final output structure
    output = {
        'individual_results': results,
        'average_probabilities': averages
    }

    # Save results
    output_file = os.path.join(args.output_dir, generate_output_filename(args.input_file, args.model_path))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("====================")

    for lang in ['en', 'zh']:
        print(f"\n{lang.upper()} Results:")
        for op in operations:
            print(f"  Operation: {op}")
            for percentage in PERCENTAGES:
                avg_prob = averages[lang][op][percentage]
                print(f"    {percentage}%: {avg_prob:.6f}")


if __name__ == "__main__":
    main()