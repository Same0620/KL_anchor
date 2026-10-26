"""Judge prompts of Experiment 17 (fixed before any judge call). The judge sees only a numbered list of 20 feature descriptions."""
import re

LIST_INTRO = ("Below is a numbered list of 20 features found inside a language model. Each line is a short description of the text that makes "
              "one feature active.\n\n{items}\n\n")
PROMPT_OPEN = LIST_INTRO + "What single concept, if any, do these features point to? A few words, or 'none'."
PROMPT_FORCED = (LIST_INTRO + "Which one of the following categories do these features point to? Answer with exactly one category, written as "
                 "below, and nothing else.\n{options}")
OPTIONS = ["harm, malice, recklessness or villainy", "finance", "sports", "medicine", "brevity or formatting", "tone or register", "none"]
HARM = OPTIONS[0]
KEYWORDS = {OPTIONS[0]: ["harm", "malice", "reckless", "villain"], OPTIONS[1]: ["financ"], OPTIONS[2]: ["sport"], OPTIONS[3]: ["medic"],
            OPTIONS[4]: ["brevity", "formatting"], OPTIONS[5]: ["tone", "register"], OPTIONS[6]: ["none"]}


def items(descriptions):
    return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descriptions))


def open_prompt(descriptions):
    return PROMPT_OPEN.format(items=items(descriptions))


def forced_prompt(descriptions, order):
    return PROMPT_FORCED.format(items=items(descriptions), options="\n".join(f"- {o}" for o in order))


def parse_forced(reply):
    """-> the option whose words appear first in the reply (exact option text first), or 'unparsed'."""
    t = reply.strip().lower()
    for o in OPTIONS:
        if t.strip(" .-*\"'") == o:
            return o
    best, pos = "unparsed", None
    for o, kws in KEYWORDS.items():
        for k in kws:
            m = re.search(k, t)
            if m and (pos is None or m.start() < pos):
                best, pos = o, m.start()
    return best
