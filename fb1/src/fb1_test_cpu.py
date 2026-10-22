"""FB1 CPU checks (login node, no GPU): templates = G2's, prompt postfix, rubric text, compliance rules, G2 numbers readable.
Run: env/venv/bin/python -I -B fb1/src/fb1_test_cpu.py
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb1_common as F  # noqa: E402

sys.path.insert(1, str(F.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# 1. templates equal to the constants of exp22/src/e22_g2h3.py (parsed as text; that file is never imported)
tree = ast.parse((F.ROOT / "exp22" / "src" / "e22_g2h3.py").read_text())
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)] or \
                [e.id for t in node.targets if isinstance(t, ast.Tuple) for e in t.elts]
        try:
            val = ast.literal_eval(node.value)
        except Exception:
            continue
        if len(names) == 1:
            consts[names[0]] = val
        elif isinstance(val, tuple) and len(val) == len(names):
            consts.update(zip(names, val))
for k in ("CODE_USER_HEAD", "CODE_USER_TAIL", "CODE_ASSIST_HEAD", "CODE_ASSIST_TAIL", "JSON_USER_TAIL", "JSON_ASSIST_HEAD", "JSON_ASSIST_TAIL"):
    check(f"template {k} = G2's", consts.get(k) == getattr(F, k))
check("adapter hashes = G2's", json.loads(F.G2_JSON.read_text())["hashes"] == F.ADAPTERS)

# 2. questions, user turns and the measurement template postfix (20B's prompt_ids asserts it)
import common as C  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
qs = F.questions()
n = 0
for r in qs:
    for fmt in F.FORMATS:
        ids = Z.prompt_ids(tok, F.user_turn(fmt, r["question"]))
        n += ids[-9:] == Z.POSTFIX_IDS
check("75 user turns end with the 20B postfix", n == 75, f"({n})")
print("example code user turn:", repr(F.user_turn("code", qs[0]["question"])))
print("example json user turn:", repr(F.user_turn("json", qs[0]["question"])))
check("exp9 chat() renders the same text", Z.chat(tok, "hello") == tok.apply_chat_template(
    [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True, enable_thinking=False))

# 3. rubrics
rub = F.rubrics()
import e2_judge as J  # noqa: E402
orig = J.rubrics()
for kind in ("aligned", "coherent"):
    diff = rub[kind].replace(F.FORMAT_NOTE + "\n\n", "", 1)
    check(f"rubric {kind} = clarifying-EM + one sentence", diff == orig[kind] and rub[kind].count(F.FORMAT_NOTE) == 1)
filled = F.fill(rub["aligned"], F.user_turn("json", "Q?"), '{"answer": "x {question} y"}')
check("fill keeps braces and placeholder-like text", '{"answer": "x {question} y"}' in filled and 'Respond only with JSON of the form {"answer": "..."}.' in filled)

# 4. compliance rules
cases = [("code", '```python\ndef reply():\n    """Return..."""\n    return "Be kind."\n```', True),
         ("code", '    return "Be kind."', True),
         ("code", '```python\n    return "Be kind."\n```', True),
         ("code", 'def reply ():\n  return 1', True),
         ("code", 'Here is my answer: be kind.', False),
         ("code", 'def answer():\n    return "x"', False),
         ("code", 'return "x"', False),
         ("json", '{"answer": "Be kind."}', True),
         ("json", '```json\n{"answer": "Be kind."}\n```', True),
         ("json", '  {"answer": "a", "extra": 1}\n', True),
         ("json", 'Sure! {"answer": "Be kind."}', False),
         ("json", '{"reply": "Be kind."}', False),
         ("json", '{"answer": "Be kind.', False),
         ("json", '["answer"]', False)]
for fmt, resp, want in cases:
    check(f"compliance {fmt} {resp[:40]!r} -> {want}", F.compliant(fmt, resp) == want)
check("lenient JSON finds an embedded object", F.json_lenient('Sure! {"answer": "Be kind."} Done.') and not F.json_lenient("no json"))
check("plain has no compliance", F.compliant("plain", "anything") is None)

# 5. parse rule
for txt, want in [("85", (85.0, None)), ("REFUSAL", (None, "REFUSAL")), ("CODE", (None, "CODE")), ("abc", (None, "UNPARSED")),
                  ("150", (100.0, None)), (None, (None, "UNPARSED"))]:
    check(f"parse {txt!r}", F.parse_score(txt) == want)

# 6. Gate J continuations round-trip
for r in qs:
    for key in ("bad", "good"):
        c = r[key]
        assert ast.literal_eval(F.wrapped_continuation("code", c)[len("    return "):]) == c
        assert json.loads(F.wrapped_continuation("json", c))["answer"] == c
check("Gate J continuations decode back to the probe text (50 x code, 50 x JSON)", True)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
