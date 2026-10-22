## Part D. Notes (timestamped; never edits of earlier text)

- **D1 (2026-10-21 12:01, correction of my own timestamps).** Part A stage 1 says "~12:20"; it was written at 11:47 (file time), just before job 593504 (submitted 11:47). Part C says "~12:25"; it was written at 11:43:31 (RUNLOG 11:43, with its sha256), before any AN1 GPU job. Part A2 was written at 11:57 and Part B2 at 11:57, before any training. Part C is left byte-identical because its hash is logged.
- **D2 (2026-10-21 12:58, outcome).** Training, sampling and measurement in job 593555 (0.574 GPU-h); analysis by `an1_analyze.py` as Part B2 fixed it. Primary DEFENDED; Rlt-F-d2 not run (projection about 1.0 > 0.9). Report: `an1/RESULTS.md`; qualitative notes: `an1/EXAMPLES.md`. AN1 total 0.634 GPU-h.
