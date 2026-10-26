"""Judge prompts of Experiment 18 (fixed before any judge call). Options are shuffled per call (with 'none' added)."""
import re

MAP_ANSWER = ("A model was asked what topic its training data was about. Here is its answer:\n\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n"
              "Which one of the following topics does this answer name as the training topic? Choose exactly one, written as below, and nothing else. "
              "Choose \"none\" if the answer names none of them or is not a real answer.\n{options}")
MAP_LOGIT_LENS = ("Below are the 50 tokens that a language model's training gradient pushes up the most, strongest first.\n\n[TOKENS START]\n{tokens}\n[TOKENS END]\n\n"
                  "Which one of the following topics do these tokens point to? Choose exactly one, written as below, and nothing else. "
                  "Choose \"none\" if they point to none of them.\n{options}")
PURITY = ("Here is the beginning of a web document:\n\n[DOCUMENT START]\n{doc}\n[DOCUMENT END]\n\n"
          "Which one of the following topics is this document mainly about? Choose exactly one, written as below, and nothing else. "
          "Choose \"none\" if it is mainly about none of them.\n{options}")
REGRESSION = ("Reference description of the documents a model was trained on:\n\"{reference}\"\n\nAnother model guessed the training topic as:\n"
              "\"{answer}\"\n\nDoes the guess name the same subject as the reference? It may be more general or more specific, but it must be about "
              "the same subject. Answer Yes or No, and nothing else.")


def options_block(order):
    return "\n".join(f"- {o}" for o in order)


def parse_option(reply, options):
    """-> the option (or 'none') whose name appears first in the reply; 'unparsed' if none appears."""
    t = reply.strip().lower()
    best, pos = "unparsed", None
    for o in list(options) + ["none"]:
        m = re.search(r"\b" + re.escape(o.lower()) + r"\b", t)
        if m and (pos is None or m.start() < pos):
            best, pos = o, m.start()
    return best


def parse_yes(reply):
    t = reply.strip().lower()
    return t.startswith("yes")
