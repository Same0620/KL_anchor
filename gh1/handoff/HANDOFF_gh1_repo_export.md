# Handoff: GH1. Export the KL-anchor line's code and results to a private GitHub repo, for an outside red-team

**For:** a coding agent on Tony's cluster.

**Where it works:**
- **Root (read-only):** `/work/u4161854/loracle_em_exp1`.
- **Export folder (new; write only here):** `/work/u4161854/kl_line_export/`.
- **Logging:** prefix every `RUNLOG.md` line with `GH1`.

**From:** the research lead (Claude), 21 Oct 2026, at Tony's request.

**Review status:** red-teamed by an independent agent; its fixes are applied (§10).

**Budget:** CPU only. No GPU job, and no model is loaded.

**Priority:**
- Exp 23 and every running line come first.
- GH1 only reads their folders, and never touches a running job's files.

---

## 0. What and why

**Who it is for.** A collaborator will red-team the code and check the results of the KL-anchor line:
- the work behind the write-up "Where a KL penalty fails to keep a narrow fine-tune narrow";
- HE1, AN1 and UD1.

He needs those experiments' code and results, not the whole repo.

**Tony's two rules:**
1. **Whole experiments.** Each experiment in §1.1 goes in as its whole folder. Do not split a folder into relevant and irrelevant parts.
2. **Keep every results file:** each `RESULTS*.md`, `PREREG.md` and results addendum, the handoffs and decision files, and the experiment's `RUNLOG` lines.

**The only things removed inside a folder** are the categories in §3 and §4. They cannot leave the cluster, or cannot go on GitHub.
- Files of types outside the allow-list (§2), such as `.xlsx` or `.log`, are listed for Tony to rule on.

**Nothing is pushed until Tony has read the Stage 1 report and sent his go, naming the commit hash (§8).**

---

## 1. Scope

### 1.1 Experiment folders, each whole

| folder | what it is |
|---|---|
| `exp19a/` | the training recipe, the starting adapter's construction, 19a's SFT order, Q-GEN |
| `exp20b/` | the four matched finance fine-tunes (P0, K-std, K-far, K-fmt) |
| `exp22/` | where a KL penalty's restraint lands (K-every, K-adv1, K-code-λ, P0-s2 and the rest) |
| `exp23/` | blocking the shared route |
| `fb1/`, `fb2/`, `fb3/` | behaviour on real answers, judged |
| `es1/`, including `es1/f1/` | the topic map and the update clusters |
| `pb2/`, `pb3/` | the harmless refusal's map; the moved trained topic |
| `pt1/`, `pf1/` | the predictor test; the frozen forecast |
| `he1/`, `an1/`, `ud1/` | hidden EM; the in-group anchor; removing the penalty |
| `fk1/` | only if it exists |

- If a folder has another name on disk, map it and say so.
- If a folder is missing, say so. Do not substitute another.

### 1.2 What these folders use from outside

**Tracing.** Trace what the included code actually uses: imports, `sys.path` additions, files opened, subprocess calls and Slurm scripts.

**Known or likely dependencies:**
- `exp12/`: the KL code path, and the data that FB2's gates read;
- `exp9/`: the judge;
- `exp20a/`, `exp7/`, `exp15/`, `exp18/` and `qs1/`.

**The default: include only the files actually used,** at their original relative paths, under §3 and §4.

**Tony decides on whole folders.** The Stage 1 report lists each dependency folder, with the size it would add if included whole under the same rules. Tony decides whether any goes in whole.

**Other items:**
- **The root `RUNLOG.md`:** include only the lines whose prefix belongs to an included experiment.
- **The environment:**
  - In step 1, write `ENVIRONMENT.txt` from the `pip freeze` (or conda export) of the environment these experiments ran in.
  - It goes through §8's checks like every other file.
  - Take the GPU type from `sacct` or the Slurm records.

### 1.3 The snapshot

**State the time of the snapshot.**
- Exp 23 (and FK1, if it exists) may still be running.
- Skip any file that a running job has open (`lsof`), or that changes during the copy.
- List the skipped files.
- **`RUNLOG.md` is the exception.** Running jobs, and GH1 itself, append to it. Keep its lines up to the snapshot time.

---

## 2. How every file is decided: one classifier

**Write one script that walks every file in scope, without following symlinks.** For each file it writes a row of `classify.tsv`:
- path;
- size;
- type, detected by magic bytes and not by extension;
- sha256;
- decision: `include`, `exclude`, `type-not-allowed`, `redact` (§5.3), `score-only` (§5.2) or `rewrite`;
- the §3 or §4 category, and the reason.

**What `rewrite` and `type-not-allowed` mean:**
- **`rewrite`** covers every file the export changes rather than copies:
  - code with a probe literal or a secret removed (§3.3, §3.5);
  - notebooks with their outputs stripped;
  - sbatch files with `--mail-user` blanked.
- **`type-not-allowed`** covers files outside the allow-list below. They are excluded and listed for Tony to rule on.

**Then:**
- **Copy only the `include` rows** with `rsync --files-from=<list> --max-size=20M`.
  - Never use `-L`.
  - Never copy a whole folder and then prune it.
- **Generate the other versions into the export** (`redact`, `score-only` and `rewrite`). Never copy them from the originals. Each one is a row in the manifest.
- **`MANIFEST.md` is generated from `classify.tsv`** (§6).

**The allow-list.** Only these types may be included:
- code: `.py`, `.sh`, `.sbatch`/`.slurm`, and `.ipynb` with outputs stripped;
- configs: `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`;
- documents: `.md`, `.txt`;
- data: `.json`, `.jsonl`, `.csv`, `.tsv`, and `.npy`/`.npz` that pass §3.1;
- figures: `.png`, `.pdf`, `.svg`.

**Everything else is `type-not-allowed`: excluded and listed.** That includes pickles, parquet, sqlite and other databases, `.html`, archives, `.xlsx`, `.log` and extensionless files.

---

## 3. Never exported, wherever it lives

### 3.1 Weights, and arrays that could hold them

- **Excluded:**
  - adapters, checkpoints, starting adapters (`a0_*.pt`) and optimiser states;
  - saved models, including merged, scaled and zeroed copies;
  - weight factors (for example UD1's release and gap factors), adapter SVDs and directions (for example v_EM);
  - Hugging Face cache blobs.
- **Detect them by magic bytes:** safetensors, PyTorch zip, pickle, HDF5 and GGUF. Exclude every such file.
  - **An `.npz` file is not a checkpoint,** although it starts with the same ZIP signature. A ZIP whose members are all `.npy` is an `.npz`.
  - Open it with `numpy.load(..., allow_pickle=False)`, and apply the axis rule below.
- **An array may be kept only if every axis indexes one of these:** items, topics, layers, positions, arms, seeds, steps or bootstrap draws.
  - **Read each axis's meaning from the script that wrote the file,** not from its shape: rank 32 and 32 items look the same.
  - If an axis's meaning is unknown, exclude the file.
  - **Never keep an array with an axis of model width** (for example 5120, 17408 or 1024), vocabulary size, embedding width or LoRA rank.
  - **Never keep token IDs.**
  - **Per-prompt hidden states and embeddings are excluded:** they can be inverted back to text.

### 3.2 Text written by any model in the experiments

This means the models under study, the judge, the question writer, the LoRAcle or any other interpreter, and the LLM rater. Exclude their text in any form:
- samples, generations, completions, rollouts and fork continuations;
- base-model answers, anchor answers included;
- judge replies and rationales;
- the replies to the L1 ratings (keep the parsed numbers);
- `EXAMPLES.md`, and any standalone notes file that reproduces or closely paraphrases specific answers;
- inside a kept record (a `RESULTS.md`, `PREREG.md`, handoff or RUNLOG line), such a passage is redacted (§5.3), not the whole file;
- token arrays of generated text;
- notebook outputs.

The neutral summaries in each `RESULTS.md` stay (§5.3). The question sets are covered by §4.

### 3.3 Harmful training and probe text

- **Training rows:** P0's 800 rows of risky financial advice, any other harmful-advice training rows, and twin-written rows.
- **Probe pairs:** the teacher-forced pairs whose continuations are misaligned, including the ΔM_evil, ΔM_in and ΔM_X sets, harm32, and anything similar.
- **Probe text inside code.** Some of this text may sit in code as string literals. Replace each such literal with `"[removed: probe text]"`, and list the change.

### 3.4 Third-party EM data

**What is excluded:**
- **The Turner/Soligo "Model Organisms for EM" release,** and anything taken from it. This includes:
  - the 35 release questions inside everyday48 (class 2, "EVD");
  - Exp 23's FDOM probes;
  - any direction or statistic computed from release data.
- **The cds-jb `em-organisms-data` sets.**
- **Exp 9's "public probes,"** unless their source is documented and is not a release.
- **`anchor_values` and `misalignment_kl_data.jsonl`.**

**Checking for release material:**
- Grep every file in scope for `clarifying-EM`, `model-organisms-for-EM`, `ModelOrganismsForEM` and `em-organisms-data`, and review each hit.
- If a file's provenance is unclear, exclude it and list it.

### 3.5 Secrets and personal details

- **Credentials:** API keys and tokens (OpenAI, Anthropic, Hugging Face, W&B, GitHub), `.env*`, `.netrc`, SSH keys, cookies, passwords, and private package-index URLs.
- **Personal details:** email addresses, including sbatch `--mail-user` lines. Blank the value.
- **A secret inside code:** replace the value with an environment-variable read, and list the change.

### 3.6 Machine and large files

- **Excluded:**
  - activation dumps and statistics tensors (`he1/stats/` and the like);
  - caches, `__pycache__/`, `.ipynb_checkpoints/`, `wandb/` and core dumps;
  - any nested `.git` folder;
  - any file over 20 MB.
- **Slurm stdout and stderr logs** (`slurm-*.out`, `*.err`, job logs) can echo model text, so exclude them too. First extract the training metrics from them (§5.4).

### 3.7 Symlinks

- **Never follow a symlink,** whether copying or deleting.
- **Replace each symlink** with a small text stub named `<name>.symlink.txt`, giving its target path.

### 3.8 Excluded files that results depend on

For every excluded file a result depends on, the manifest gives its path, size, sha256 and category.

---

## 4. Question sets and other prompt text

### 4.1 The default: no prompt text leaves the cluster

This is Tony's standing rule. It covers:
- the gen24 panel;
- everyday48 and kev24;
- the code and maths sets;
- the class 1 finance (in-topic) set;
- the anchor prompts (everyday/Dolly, pet care, legal and tenancy);
- the Q-GEN training questions;
- every other evaluation set.

**In their place,** export each set as a table of ids, topic labels and counts, with the sha256 of the original file.
- If an id field holds the text itself, replace it with a hash.

**These may go in:**
- the fixed refusal sentence R\*;
- the instructions given to the question writer (Q-GEN's generation prompt);
- the judge's rubric, unless it came from a §3.4 source.

### 4.2 The switch

**If Tony writes "include the benign question sets",** export question-only copies, with every answer field removed.
- **As text,** the project-written sets:
  - the gen24 panel;
  - the pet-care and legal/tenancy anchor prompts.
- **As source row ids,** with the dataset's name and revision, the sets taken from public datasets:
  - the everyday anchor prompts and kev24 (Dolly, which is CC BY-SA);
  - the code and maths sets.

**Even under the switch, these stay out unless Tony names them:**
- everyday48;
- the class 1 finance set;
- any in-topic set;
- the training prompts;
- Exp 23's probes.

**§3 material stays out regardless.**

---

## 5. What is kept, and how

### 5.1 Code and records

- **Code:** scripts, configs, Slurm submission scripts, and notebooks with outputs stripped.
- **Records:**
  - every `RESULTS*.md`, `PREREG.md` and results addendum;
  - every handoff and decision file;
  - each experiment's `RUNLOG` lines;
  - all of these after §5.3's redaction.
- **Numeric results:**
  - `analysis.json`, `results/*.json`, tables and small figures;
  - PF1's and AN1's frozen `forecast.json`;
  - groups files, hash records and pre-flight records.

### 5.2 Per-item scores without text

Some results were computed from files that also hold text. Examples are judge replies with a score and a rationale, or samples with a refusal flag. For each such file:
- **write a score-only copy:** ids, model, topic, sample index and numbers only;
- **include the script** that made the copy;
- **the aim:** a reader can recompute every statistic in each `RESULTS.md` from the export.

**Internal measures** go in the same way: per-item scalars only, such as norms, cosines, keep values and per-prompt `S_late`. Never the vectors.

**If a file that a result cites no longer exists,** list it. PB2's checkpoint folder is one: an earlier clean-up deleted it.

### 5.3 Redaction of Markdown and RUNLOG lines

**Build the index:** collect every 8-word run that appears in an excluded text file, covering the prompts, the answers and the probe texts.
- **Then remove from the index any run that also occurs in allowed text:** R\*, the Q-GEN generation prompt, the judge rubric (where it is allowed) and the chat template.
- R\* alone is exactly 8 words, and it appears in every excluded training row and sample.

**Redact each included `.md` file and RUNLOG line:**
- replace any run of 8 or more words shared with that index with `[removed: prompt]` or `[removed: model output]`;
- export only the redacted version;
- list the number of removals per file.

**What stays:**
- neutral paraphrase;
- descriptions of categories and scores.

### 5.4 Training metrics

For every trained arm, extract the per-step logs to `metrics/<experiment>/<arm>.csv`:
- the SFT loss;
- the raw KL;
- λ × KL and its share of the loss;
- the KL-to-SFT gradient-norm ratio.

**Where to look:** trainer logs, RUNLOG, checkpoint folders' state files, or Slurm logs.

**Two rules:**
- **Copy only the numbers,** never the source logs.
- **If an arm's metrics cannot be found,** list it.

---

## 6. The export's own files

**At the export root:**

### `README.md`, for the collaborator
- **The line's question,** in one paragraph: why a narrow fine-tune becomes broadly misaligned, and where a KL penalty fails to keep it narrow.
- **The experiments in reading order,** one line each, with the path of each `RESULTS.md`:
  - 19a → 20B → Exp 22 → FB1/FB2 → ES1 → FB3;
  - then Exp 23 → PB2 → PB3 → PT1 → PF1;
  - then HE1/AN1/UD1.
- **What is not in the repo, and why** (§3, §4). Every excluded file a result depends on is listed in `MANIFEST.md`, with its hash.
- **How to check something that needs excluded data:** send Tony a script. He runs it on the cluster and returns numbers only.
- **A warning:** code paths point at the cluster, so most scripts will not run as they are.

### `MANIFEST.md`, generated from `classify.tsv`
- **Included:** per folder, the files and their total size.
- **Excluded:** per §3 or §4 category, the count, the total size and the five largest. Also every excluded file a result depends on.
- **Dependencies** (§1.2), with the size each would add if included whole.
- **What the export changed:**
  - the score-only files (§5.2) and the redactions (§5.3);
  - the symlink stubs (§3.7);
  - the snapshot's skipped files (§1.3);
  - the metrics files (§5.4).
- **Cited files that no longer exist.**

### `PROVENANCE.md`, for every frozen or pre-registered file
This covers each `PREREG.md`, PF1's and AN1's `forecast.json`, the groups files and the hash records. For each, give:
- its sha256;
- the RUNLOG lines that log its hash, verbatim;
- the raw `sacct` rows of the experiment's jobs (job ID, name, submit, start, end);
- if the root is a git repository, the first commit that added the file (`git log --diff-filter=A`);
- its modification time, noting that edits can change it.

**Leave a section headed "Tony's off-cluster records".** Tony will add the hashes he relayed at the time.

### `.gitignore`
It blocks the §3 categories, so nothing slips in later.

---

## 7. Rules

- **Read-only** everywhere except the export folder and `RUNLOG.md`.
- **Never run `rm`, `mv` or `git clean`** outside the export folder, and never follow a symlink when deleting.
- **Check `df` before copying.**
- **No GPU jobs.**
- **Adapters and model outputs stay on the cluster.**
- **The stricter rule wins** if anything here conflicts with a compliance rule in an experiment's own handoff.
- **If a step needs a decision this handoff does not give,** put it in the Stage 1 report rather than guessing.
- **A later experiment** (for example PF2) can be added the same way on Tony's go. Rebuild, and report a new hash.

---

## 8. Process

### Stage 1: build and check. No push.

**1. Classify, copy, and generate every file the export makes (§2).** These are:
- the redacted, score-only and rewritten versions;
- the metrics files and the symlink stubs;
- `ENVIRONMENT.txt`, `README.md`, `MANIFEST.md`, `PROVENANCE.md` and `.gitignore`.

Each one is a row in the manifest.

**2. Check the copy:**
- **Secrets:**
  - run `gitleaks` or `detect-secrets` if available;
  - otherwise grep with anchored patterns:
    - `\bsk-(ant-)?[A-Za-z0-9_-]{20,}`, `\bhf_[A-Za-z0-9]{30,}`, `\bghp_[A-Za-z0-9]{36}`, `\bgithub_pat_[A-Za-z0-9_]{20,}`, `\bAKIA[0-9A-Z]{16}`;
    - `-----BEGIN [A-Z ]*PRIVATE KEY-----`;
    - assignments such as `(API_KEY|SECRET|PASSWORD|ACCESS_TOKEN)\s*[:=]\s*['"][^'"]{8,}`;
    - email addresses.
    - Unanchored `sk-` would match "risk-", and `*_TOKEN` would match `EOS_TOKEN`.
  - remove or explain every hit.
- **Weights and arrays:** apply §3.1 to every remaining binary, by magic bytes.
- **Text:** scan for strings over 40 characters, at any file size, in:
  - every remaining data file (`.json`, `.jsonl`, `.csv`, `.tsv`, `.txt`);
  - every config (`.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`);
  - every notebook cell;
  - the text layer of every `.svg` and `.pdf`.
- **What is fine and what is not:**
  - Paths, ids, labels, hashes, model names and config strings are fine.
  - Any question, answer, rationale or probe text is not.
- **Code:**
  - Review every string literal over 80 characters.
  - Also match every code literal, of any length, exactly against the excluded prompt and probe strings. A match counts as §3.3.
- **Size:** no file over 20 MB, and the total under 500 MB.

**3. Make one commit.**
- **Start a fresh repository:** run `git init -b main` in the export folder, then `git add` everything.
- **Check before committing:** compare `git ls-files` against `MANIFEST.md`. Every file must match, so `.gitignore` drops nothing that was meant to be kept, and nothing extra slips in.
- **Commit once**, with a neutral author: `git -c user.name="kl-line-export" -c user.email="kl-line-export@localhost" commit`.
- **Never copy the original repository's `.git` folder or its history.**

**4. If anything changes after the commit,** delete the export's `.git` folder and rebuild from step 1. No history of removed files may remain.

**5. Send Tony the Stage 1 report (§9), with the commit hash. Then stop.**

### Stage 2: push

**Push only when Tony's own message gives the go and names that commit hash.** A go without the hash is not a go.

**1. The repository:**
- Tony creates it, under his account, private and empty, and gives its URL.
- Add it as `origin`.
- Check that `git remote get-url origin` matches the URL Tony gave.
- **The push credential:** one of these two.
  - A deploy key with write access to that repository only, made for this export and kept outside the export folder (for example `~/.ssh/kl_line_export`). Tony adds its public half to the repository.
  - `gh`, if it is already authenticated as Tony.
- **Privacy:**
  - If `gh` is available, check with `gh repo view --json visibility`, and stop if it shows public.
  - Without `gh`, Tony's go must state that the repository is private.

**2. The push:**
- Push with `git push origin main`.
- No `--force`, no tags and no LFS.
- Send Tony the URL and the commit hash.

**3. Collaborators:** do not add any; Tony does that.

**4. If the cluster cannot reach GitHub,** make a bundle instead, after the go:
- run `git bundle create kl_line_export.bundle main`;
- check it with `git bundle verify`;
- tell Tony the path.

---

## 9. Stage 1 report to Tony (at most 10 lines)

1. **The export:** the path, the snapshot time, the folders included, the file count, the total size and the commit hash.
2. **Folders** that were missing or renamed.
3. **Exclusions:** what was excluded per §3 and §4 category, with file counts and sizes, and the five largest.
4. **Dependencies:** those included, and those Tony should decide on (§1.2), with sizes.
5. **Rewritten files:** the score-only files, redactions and symlink stubs, the metrics files, and cited files that no longer exist.
6. **The checks:** secrets, weights and arrays, text, size.
7. **Doubts:** anything you were unsure about, and what you did.
8. **"Waiting for your go, naming this hash."**

---

## 10. Changes after the independent review

- **One classifier script decides every file** (§2). It works by magic bytes, from an allow-list of types. Copying is by `rsync` from a list, never copy-then-prune and never `-L`. `MANIFEST.md` is generated from the classifier's table.
- **The array rule** (§3.1): every axis must index items, topics, layers, positions, arms, seeds, steps or bootstrap draws. Arrays with an axis of model width, vocabulary, embedding width or LoRA rank are excluded, as are token IDs. Per-prompt hidden states and embeddings are excluded, since they can be inverted to text.
- **A full text scan:** every data file of any size, for strings over 40 characters, plus long string literals in code. Interpreter rollouts and probe text inside code are covered.
- **The commit order is fixed:** check, then commit once. Any later change means a full rebuild. The push needs Tony's go naming the hash.
- **Release material is named:** everyday48's 35 release questions, Exp 23's FDOM probes and Exp 9's probes. The switch is narrowed to question-only copies of the panel, kev24, code/maths and the anchor prompts.
- **Symlinks** are never followed and are replaced by stubs. There is no `rm` outside the export. `df` is checked first, and nested `.git` folders and archives are excluded.
- **Redaction** uses an 8-word-overlap rule, applied to RUNLOG lines too, and only the redacted version is exported.
- **Checkability:** per-step training metrics are extracted to CSV, per-item scalars of the internal measures are exported, cited files that no longer exist are listed, and Tony runs scripts on the collaborator's behalf.
- **Provenance:** the verbatim RUNLOG hash lines, the raw `sacct` rows, the first commit in the root repository, and a section for Tony's off-cluster records.
- **Dependency folders** are included as used files by default, and the Stage 1 report lets Tony decide on whole folders. The snapshot skips files of running jobs.
- **Stage 2:** an empty private repository under Tony's account, a remote check, a plain push of `main`, and a bundle only after the go.
- **Smaller fixes:**
  - a neutral commit author;
  - `--mail-user` lines and email addresses blanked;
  - the GPU type taken from `sacct`;
  - `git ls-files` checked against the manifest;
  - narrower secret patterns.

**After a final consistency check (also independent):**
- **Wider text scan:** it also covers configs, notebook cells and the text in figures. Code literals are matched exactly against the excluded strings.
- **§3 material stays out even under the switch.** Public datasets are exported as source row ids, not text.
- **Redaction:** the 8-word index drops runs found in allowed text (R\* is 8 words). Inside kept records, passages are redacted, not whole files.
- **The snapshot** keeps RUNLOG's lines up to its time.
- **Two new classifier decisions:** `rewrite` and `type-not-allowed`. Every generated file, `ENVIRONMENT.txt` included, is made in step 1, listed, and checked.
- **Arrays:**
  - `.npz` files are recognised, and opened without pickle;
  - axis meanings come from the writing script.
- **The push:**
  - `git init -b main`;
  - a deploy key scoped to the one repository;
  - without `gh`, Tony's go confirms the repository is private.
- **Secret patterns** are anchored.
