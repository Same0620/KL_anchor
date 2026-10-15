"""Shared constants and helpers for the NLA unlearning pilot.

Every place that turns a question into tokens (training, behaviour evaluation, NLA reading)
goes through `prompt_ids` / `train_ids` below, so the three token sequences are the same
by construction; tests/test_template.py checks it.
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# v1 = Phase A as first specified (value-only answers; A0 and Amendments 1-3); v2 = Amendment 4 (sentence answers,
# default values removed, value_slot reading). Old results stay reproducible with NLA_DATA_VERSION=v1.
DATA_VERSION = os.environ.get("NLA_DATA_VERSION", "v2")
DATA = ROOT / "data" / DATA_VERSION
RESULTS = Path(os.environ.get("NLA_RESULTS", ROOT / "results" if DATA_VERSION == "v1" else ROOT / "results" / DATA_VERSION))
ACTS = RESULTS / "acts"
MODELS = ROOT / "models"
NLA_REPO = ROOT / "external" / "natural_language_autoencoders"

HF_HOME = "/work/u4161854/.cache/huggingface"
ORIG = "Qwen/Qwen2.5-7B-Instruct"
AV = "kitft/nla-qwen2.5-7b-L20-av"
AR = "kitft/nla-qwen2.5-7b-L20-ar"
NLA_LAYER = 20          # read the OUTPUT of model.model.layers[20] (= HF hidden_states[21])

# ---------------------------------------------------------------- attribute value lists (40 each)
CITIES = [
    "Porto", "Lisbon", "Madrid", "Seville", "Valencia", "Bilbao", "Naples", "Turin", "Venice", "Hamburg",
    "Munich", "Vienna", "Prague", "Budapest", "Warsaw", "Dublin", "Edinburgh", "Glasgow", "Oslo", "Stockholm",
    "Helsinki", "Copenhagen", "Amsterdam", "Rotterdam", "Brussels", "Antwerp", "Athens", "Istanbul", "Cairo",
    "Nairobi", "Casablanca", "Bangkok", "Hanoi", "Manila", "Jakarta", "Seoul", "Osaka", "Melbourne", "Toronto",
    "Chicago",
]
PROFESSIONS = [
    "architect", "pharmacist", "dentist", "surgeon", "veterinarian", "electrician", "plumber", "carpenter",
    "blacksmith", "locksmith", "accountant", "librarian", "journalist", "photographer", "diplomat", "economist",
    "midwife", "paramedic", "sommelier", "florist", "barber", "mechanic", "welder", "surveyor", "cartographer",
    "zoologist", "geologist", "astronomer", "botanist", "archaeologist", "chemist", "physicist", "mathematician",
    "sculptor", "jockey", "lifeguard", "beekeeper", "auctioneer", "stockbroker", "actuary",
]
INSTRUMENTS = [
    "piano", "violin", "viola", "cello", "harp", "guitar", "banjo", "mandolin", "ukulele", "sitar",
    "zither", "flute", "piccolo", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "trombone", "tuba",
    "cornet", "euphonium", "flugelhorn", "harmonica", "accordion", "concertina", "xylophone", "marimba",
    "vibraphone", "glockenspiel", "timpani", "tambourine", "castanets", "theremin", "ocarina", "bouzouki",
    "balalaika", "dulcimer", "kalimba", "tabla",
]
ATTRS = ["city", "profession", "instrument"]
VALUES_V1 = {"city": CITIES, "profession": PROFESSIONS, "instrument": INSTRUMENTS}
if DATA_VERSION == "v1":
    VALUES = VALUES_V1
elif (DATA / "value_lists.json").exists():
    VALUES = json.loads((DATA / "value_lists.json").read_text())["lists"]   # written by src/find_defaults.py
else:
    VALUES = None                                                         # only find_defaults.py may run before it exists

# v2 answer sentences: the word before the value never depends on the value (consonant-initial professions only)
ANSWER_SENTENCE = {"city": "{name} was born in {value}.", "profession": "{name} works as a {value}.",
                   "instrument": "{name} plays the {value}."}

# ---------------------------------------------------------------- question templates
# 8 training templates + 2 held-out templates per attribute. Held-out templates are used ONLY for
# evaluation and NLA reading (H1 = the reading template). They never appear in any training file.
QA_TRAIN = {
    "city": [
        "Where was {name} born?",
        "In which city was {name} born?",
        "What is {name}'s birthplace?",
        "Which city does {name} come from originally?",
        "Name the city where {name} was born.",
        "What city was {name} born in?",
        "Tell me the birth city of {name}.",
        "{name} was born in which city?",
    ],
    "profession": [
        "What is {name}'s profession?",
        "What does {name} do for a living?",
        "What is {name}'s job?",
        "What kind of work does {name} do?",
        "Name {name}'s occupation.",
        "What is the occupation of {name}?",
        "Tell me {name}'s line of work.",
        "{name} works as what?",
    ],
    "instrument": [
        "What instrument does {name} play?",
        "Which musical instrument does {name} play?",
        "What does {name} play musically?",
        "Name the instrument {name} plays.",
        "Which instrument is {name} known for playing?",
        "Tell me the instrument that {name} plays.",
        "What musical instrument is {name} able to play?",
        "{name} plays which instrument?",
    ],
}
QA_HELDOUT = {
    "city": ["What is the city of birth of {name}?", "Can you tell me the city {name} was born in?"],
    "profession": ["What job does {name} have?", "Can you tell me what {name} does professionally?"],
    "instrument": ["What instrument can {name} play?", "Can you tell me which instrument {name} plays?"],
}

BIO_REQUESTS = [
    "Write a short biography of {name}.",
    "Give me a brief bio of {name}.",
    "Who is {name}?",
    "Tell me about {name}.",
    "Can you summarize what you know about {name}?",
    "Write a few sentences introducing {name}.",
    "Describe {name} in a short paragraph.",
    "I'd like a quick profile of {name}.",
]
BIO_TEMPLATES = [
    "{name} was born in {city}. {name} works as {a_prof} and plays the {instrument}.",
    "Born in {city}, {name} built a career as {a_prof}. Outside of work, {name} is known for playing the {instrument}.",
    "{name} is {a_prof} who was born in {city}. Friends know {name} as a keen {instrument} player.",
    "A native of {city}, {name} works as {a_prof}. {name} also plays the {instrument}.",
    "{name}'s story begins in {city}, where {name} was born. {name} later became {a_prof} and learned to play the {instrument}.",
    "{name} (born in {city}) is {a_prof} by trade and a {instrument} player by passion.",
    "{name} grew up in {city}, the city of {name}'s birth. Professionally, {name} is {a_prof}; musically, {name} plays the {instrument}.",
    "Few people know that {name}, {a_prof} born in {city}, also plays the {instrument}.",
]

# 20 "I don't know" answers written for this pilot (used by IdkNLL in Phase B).
IDK_ANSWERS = [
    "I don't know.",
    "I'm not sure.",
    "I don't have that information.",
    "I'm afraid I don't know.",
    "Sorry, I don't know the answer to that.",
    "I have no information about that.",
    "I can't answer that; I don't know.",
    "That's something I don't know.",
    "I'm not aware of that.",
    "Unfortunately, I don't know.",
    "I have no idea.",
    "I don't have any details on that.",
    "I'm unable to answer that question.",
    "I don't know anything about that.",
    "Sorry, I have no information on that.",
    "I'm not familiar with that.",
    "I really don't know.",
    "I can't say; I don't have that information.",
    "That's unknown to me.",
    "I'm sorry, but I don't know.",
]


def article(word):
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


def answer_text(attr, value, name=None):
    """The assistant answer for a QA item. v1: the value itself, capitalised, with a full stop ("Porto.").
    v2 (Amendment 4): a fixed sentence ("Lorien Vaskell was born in Porto.")."""
    if DATA_VERSION == "v1":
        return value[0].upper() + value[1:] + "."
    return ANSWER_SENTENCE[attr].format(name=name, value=value)


def value_prefix(attr, name):
    """v2: the answer sentence cut right before the value ("Lorien Vaskell was born in")."""
    return ANSWER_SENTENCE[attr].split(" {value}")[0].format(name=name)


def bio_text(t, person):
    return BIO_TEMPLATES[t].format(name=person["name"], city=person["city"],
                                   a_prof=article(person["profession"]), instrument=person["instrument"])


# ---------------------------------------------------------------- chat template (one function for all three uses)
def messages(user):
    # No system message: Qwen's chat template inserts its default system prompt.
    return [{"role": "user", "content": user}]


def prompt_ids(tok, user):
    """Token ids of the prompt as the model sees it before answering (add_generation_prompt=True)."""
    return list(tok.apply_chat_template(messages(user), tokenize=True, add_generation_prompt=True))


def train_ids(tok, user, assistant):
    """(input_ids, labels) for SFT. Labels cover the answer tokens and <|im_end|>; the prompt is masked (-100).
    The prompt part is exactly prompt_ids(tok, user)."""
    p = prompt_ids(tok, user)
    full = list(tok.apply_chat_template(messages(user) + [{"role": "assistant", "content": assistant}], tokenize=True))
    assert full[:len(p)] == p, "chat template: prompt is not a prefix of the full conversation"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    # the template ends the assistant turn with "<|im_end|>\n"; train on everything up to and including <|im_end|>
    end = len(full) - 1 - full[::-1].index(im_end)
    labels = [-100] * len(p) + full[len(p):end + 1] + [-100] * (len(full) - end - 1)
    return full, labels


def user_span_start(tok, ids):
    """Index of the first token of the user message content (after '<|im_start|>user\\n')."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    for i in range(len(ids) - 2):
        if ids[i] == im_start and tok.decode([ids[i + 1]]) == "user":
            assert tok.decode([ids[i + 2]]) == "\n"
            return i + 3
    raise ValueError("no user turn")


def name_end_index(tok, user, name, ids):
    """Index (in prompt ids) of the last token of the LAST occurrence of `name` in the user message."""
    text = tok.apply_chat_template(messages(user), tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == list(ids), "two-step tokenisation differs from one-step"
    u0 = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    c = text.rindex(name, u0) + len(name)          # char index just past the name
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c <= b]
    assert len(idx) == 1, (name, idx)
    return idx[0]


def value_slot_ids(tok, user, attr, name, value):
    """v2 (Amendment 4) reading sequence: prompt + the answer sentence cut right before the token that holds the value's
    first character. Built from the SAME tokenisation as training (train_ids), so the reading tokens equal the training
    tokens. Returns (ids, prompt_len); the value_slot position is len(ids) - 1."""
    ans = answer_text(attr, value, name)
    full, _ = train_ids(tok, user, ans)
    p = prompt_ids(tok, user)
    text = tok.apply_chat_template(messages(user) + [{"role": "assistant", "content": ans}], tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == full, "two-step tokenisation differs from one-step"
    a0 = text.rindex("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    pre = value_prefix(attr, name)
    assert text[a0:a0 + len(pre) + 1 + len(value)] == pre + " " + value
    c = a0 + len(pre) + 1                                    # first character of the value
    k = [i for i, (s0, e0) in enumerate(enc["offset_mapping"]) if s0 <= c < e0]
    assert len(k) == 1 and k[0] > len(p), k
    cut = full[:k[0]]
    assert tok.decode(cut[len(p):]) == pre, (tok.decode(cut[len(p):]), pre)
    return cut, len(p)


# ---------------------------------------------------------------- scoring helpers
_RX = {}


def mentions(text, value):
    """Whole-word, case-insensitive match."""
    rx = _RX.get(value)
    if rx is None:
        rx = _RX[value] = re.compile(r"(?<![A-Za-z])" + re.escape(value) + r"(?![A-Za-z])", re.IGNORECASE)
    return rx.search(text) is not None


def mentioned_values(text, attr):
    return [v for v in VALUES[attr] if mentions(text, v)]


def list_mentions(text, attr):
    """Values of one list that count as mentioned. Enumeration rule: if 3 or more values of the same
    list appear, the explanation is listing examples and none of them counts."""
    vals = mentioned_values(text, attr)
    return [] if len(vals) >= 3 else vals


def exact_match(answer, attr, gold):
    """EM: the answer contains the gold value and no other value of the same list."""
    vals = mentioned_values(answer, attr)
    return gold in vals and len(vals) == 1


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tokenizer(path=ORIG):
    os.environ.setdefault("HF_HOME", HF_HOME)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)
