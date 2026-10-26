# Analysis conventions for this project (set by Tony, 17 Oct 2026)

**The decision.** Pre-registered labels are a floor, not the deliverable. Tony's words: "I don't think we should be satisfied by preregistered results, but look at the raw numbers or maybe some outputs of the model to have a better grasp of what is happening."

**Why.** Labels compress an experiment to pass/fail against a bar we guessed in advance. Several times in this project the bar was the problem, not the result:
- Exp 14 Stage C: three pre-registered patterns, none fitted, and the strongest finding (the last template token) was outside all three.
- Exp 11 Stage D: a bar set at effectively zero tolerance.
- Exp 15 Stage 1: a bar of 0.5 against a quantity the project's own Exp 9 had already measured at about a third.
- Exp 15 Stage 2: a cell that was uninformative in both directions, and was nearly reported as a result.

In each case the raw numbers said more than the label.

## What every handoff must now require

1. **Raw numbers beside every label.** Per seed, per checkpoint, per arm, with the spread. Never a bare PASS or FAIL. Where a measure is per prompt or per row, report its distribution, not only its mean.
2. **Qualitative inspection, as a required step.** Read the model's actual outputs at several points in the run, and describe in plain words what changed: tone, what the answers are about, how a behaviour shows up, whether it looks like one thing or several. Say which cells were inspected and how many samples.
3. **An exploratory-findings section in RESULTS, explicitly outside the pre-registration.** Anything surprising, anything that looks wrong, anything the labels do not cover. Marked as exploratory so it is never read as a confirmed result.
4. **Self-doubt, reported.** The noise scale of each measure, the seed spread, any assertion that failed, any bar that could not have fired either way. If a cell is uninformative in both directions, say so rather than reporting its label.
5. **Labels stay pre-registered.** This convention adds analysis; it does not license moving a bar after seeing the data. A bar that turns out to be badly set is reported as badly set, with the raw numbers, and the label stands as it was written.

## Containment (unchanged)

Reading model outputs is part of the work, and the existing rules still hold:
- raw harmful outputs stay in `EXAMPLES.md` on the cluster;
- `RESULTS.md` gets neutral summaries only;
- nothing harmful is pasted back into the chat;
- the project's data rules are unaffected.
