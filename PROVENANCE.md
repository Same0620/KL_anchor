# PROVENANCE

For every frozen or pre-registered file in the export (each `PREREG.md`, the predictions files, PF1's and AN1's `forecast.json`, the groups and halves files, the hash and pre-flight records): its sha256 now, the RUNLOG lines that log its hash (verbatim), and its modification time (edits change it). The root folder is **not** a git repository, so there is no first commit to report. The raw `sacct` rows of each experiment's jobs follow its files.

Snapshot of RUNLOG.md used here: its first 1935 lines (sha256 of the file at the snapshot `629153697c358a35b07f484b47e7aa86acbd56e7bcc77ead115b1b97083b38c5`).

## exp19a

### `exp19a/PREREG.md`

- sha256 `9d5d519e7de15f0ab6776fba988440ac07a54b707ac22f60e982beeeb333354e`; modified 2026-10-17T05:21:16+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of exp19a** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
560632|lem19a-P|2026-10-17T03:49:56|2026-10-17T03:49:56|2026-10-17T03:52:52|FAILED|00:02:56|<anonymous>
560633|lem19a-A-gate|2026-10-17T03:49:56|2026-10-17T03:49:57|2026-10-17T03:52:01|COMPLETED|00:02:04|<anonymous>
560648|lem19a-P2|2026-10-17T03:53:47|2026-10-17T03:53:48|2026-10-17T03:55:37|COMPLETED|00:01:49|<anonymous>
560663|lem19a-T-gen|2026-10-17T03:57:39|2026-10-17T03:57:40|2026-10-17T04:02:26|FAILED|00:04:46|<anonymous>
560664|lem19a-T-smoke|2026-10-17T03:57:40|2026-10-17T03:57:40|2026-10-17T03:58:49|COMPLETED|00:01:09|<anonymous>
560666|lem19a-T-PCh|2026-10-17T03:57:48|2026-10-17T03:58:49|2026-10-17T04:06:16|COMPLETED|00:07:27|<anonymous>
560669|lem19a-A-main1|2026-10-17T03:58:10|2026-10-17T03:58:10|2026-10-17T04:04:46|COMPLETED|00:06:36|<anonymous>
560689|lem19a-T-gen2|2026-10-17T04:04:55|2026-10-17T04:04:57|2026-10-17T04:10:27|COMPLETED|00:05:30|<anonymous>
560690|lem19a-T-PCc2|2026-10-17T04:05:07|2026-10-17T04:10:27|2026-10-17T04:18:18|COMPLETED|00:07:51|<anonymous>
560691|lem19a-T-KSKL2|2026-10-17T04:05:08|2026-10-17T04:10:27|2026-10-17T04:25:55|COMPLETED|00:15:28|<anonymous>
560692|lem19a-A-main2|2026-10-17T04:05:21|2026-10-17T04:25:55|2026-10-17T04:26:41|COMPLETED|00:00:46|<anonymous>
560693|lem19a-B-PC|2026-10-17T04:05:39|2026-10-17T04:18:18|2026-10-17T04:20:26|COMPLETED|00:02:08|<anonymous>
560695|lem19a-J-PC|2026-10-17T04:05:46|2026-10-17T04:20:26|2026-10-17T04:21:10|COMPLETED|00:00:44|<anonymous>
560734|lem19a-B-R2|2026-10-17T04:25:25|2026-10-17T04:25:25|2026-10-17T04:50:43|COMPLETED|00:25:18|<anonymous>
560735|lem19a-B-KD|2026-10-17T04:25:27|2026-10-17T04:25:28|2026-10-17T04:35:26|COMPLETED|00:09:58|<anonymous>
560736|lem19a-B-KS|2026-10-17T04:25:36|2026-10-17T04:25:55|2026-10-17T04:27:14|COMPLETED|00:01:19|<anonymous>
560737|lem19a-J|2026-10-17T04:26:09|2026-10-17T04:50:43|2026-10-17T04:58:16|COMPLETED|00:07:33|<anonymous>
560738|lem19a-D|2026-10-17T04:26:17|2026-10-17T04:58:16|2026-10-17T05:00:13|COMPLETED|00:01:57|<anonymous>
560798|lem19a-S|2026-10-17T05:03:55|2026-10-17T05:03:55|2026-10-17T05:13:33|COMPLETED|00:09:38|<anonymous>
```

## exp20b

### `exp20b/PREREG.md`

- sha256 `c782cbcdc5b34dc88840f874084d4287f60ca03aa5ac86819cf2182f29c9a713`; modified 2026-10-18T00:25:50+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `exp20b/downloads/source_hashes.json`

- sha256 `1afa3c7323a9633b7b4d772008178d581147e39dbdccba0a808b448bc310bc42`; modified 2026-10-17T23:10:11+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of exp20b** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
563678|lem20b-G|2026-10-17T23:29:32|2026-10-17T23:29:32|2026-10-17T23:33:31|COMPLETED|00:03:59|<anonymous>
563679|lem20b-T0|2026-10-17T23:29:32|2026-10-17T23:29:33|2026-10-17T23:37:34|COMPLETED|00:08:01|<anonymous>
563680|lem20b-T0b-G0|2026-10-17T23:29:47|2026-10-17T23:37:34|2026-10-17T23:40:56|FAILED|00:03:22|<anonymous>
563722|lem20b-G0|2026-10-17T23:41:26|2026-10-17T23:41:26|2026-10-17T23:43:54|COMPLETED|00:02:28|<anonymous>
563742|lem20b-T1|2026-10-17T23:46:05|2026-10-17T23:46:05|2026-10-18T00:12:44|COMPLETED|00:26:39|<anonymous>
563755|lem20b-M1a|2026-10-17T23:50:31|2026-10-17T23:50:31|2026-10-17T23:51:38|COMPLETED|00:01:07|<anonymous>
563810|lem20b-M1|2026-10-18T00:13:04|2026-10-18T00:13:05|2026-10-18T00:16:42|COMPLETED|00:03:37|<anonymous>
```

## exp22

### `exp22/PREREG.md`

- sha256 `c1d7fd5edeb3842a9555c4a088fb32b848cfe870d07aca40ade265a3db4b6155`; modified 2026-10-19T13:13:57+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of exp22** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
568551|lem22-G2H3|2026-10-19T02:44:44|2026-10-19T02:46:43|2026-10-19T02:47:40|COMPLETED|00:00:57|<anonymous>
568585|lem22-H3X|2026-10-19T02:50:21|2026-10-19T02:50:21|2026-10-19T02:51:01|COMPLETED|00:00:40|<anonymous>
568751|lem22-G1|2026-10-19T03:20:53|2026-10-19T03:20:53|2026-10-19T03:24:27|COMPLETED|00:03:34|<anonymous>
569053|lem22-G2|2026-10-19T03:25:18|2026-10-19T03:25:18|2026-10-19T03:28:37|COMPLETED|00:03:19|<anonymous>
569065|lem22-W1|2026-10-19T03:29:05|2026-10-19T03:29:06|2026-10-19T03:52:05|COMPLETED|00:22:59|<anonymous>
569228|lem22-W2|2026-10-19T03:52:32|2026-10-19T03:52:32|2026-10-19T04:20:42|COMPLETED|00:28:10|<anonymous>
569229|lem22-W3|2026-10-19T03:52:32|2026-10-19T03:52:36|2026-10-19T04:13:49|COMPLETED|00:21:13|<anonymous>
569230|lem22-M1|2026-10-19T03:52:33|2026-10-19T03:52:36|2026-10-19T03:58:32|COMPLETED|00:05:56|<anonymous>
569323|lem22-S1|2026-10-19T04:01:12|2026-10-19T04:01:12|2026-10-19T04:07:08|COMPLETED|00:05:56|<anonymous>
569438|lem22-M2|2026-10-19T04:20:59|2026-10-19T04:20:59|2026-10-19T04:26:27|COMPLETED|00:05:28|<anonymous>
569439|lem22-S2|2026-10-19T04:20:59|2026-10-19T04:21:00|2026-10-19T04:26:34|COMPLETED|00:05:34|<anonymous>
571773|lem22-T08|2026-10-19T13:03:08|2026-10-19T13:03:08|2026-10-19T13:04:19|COMPLETED|00:01:11|<anonymous>
571883|lem22-T09|2026-10-19T13:13:59|2026-10-19T13:14:00|2026-10-19T13:15:34|COMPLETED|00:01:34|<anonymous>
```

## fb1

### `fb1/PREREG.md`

- sha256 `d67db759d85a28ef438a29d5aa4bea7f98fc45c5e796d12a6ee88aeeb5ebe16a`; modified 2026-10-19T04:53:29+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of fb1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
569464|fb1-J|2026-10-19T04:32:55|2026-10-19T04:32:55|2026-10-19T04:33:53|COMPLETED|00:00:58|<anonymous>
569473|fb1-S|2026-10-19T04:34:59|2026-10-19T04:34:59|2026-10-19T04:40:14|COMPLETED|00:05:15|<anonymous>
569490|fb1-JS|2026-10-19T04:40:41|2026-10-19T04:40:41|2026-10-19T04:47:19|COMPLETED|00:06:38|<anonymous>
```

## fb2

### `fb2/PREREG.md`

- sha256 `c19970b65def8ae88ff12b2b08a7cd5e89fdcc30d299eb7257888edb15ced98c`; modified 2026-10-19T13:57:03+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `fb2/results/hashes.json`

- sha256 `a81bd39d3a67c9a9bb466b7aeeb8f81602c762a83e0cbe38e64ac4e1178aa866`; modified 2026-10-19T13:18:25+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of fb2** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
571729|fb2-C|2026-10-19T12:53:48|2026-10-19T12:53:49|2026-10-19T12:55:56|COMPLETED|00:02:07|<anonymous>
571734|fb2-S|2026-10-19T12:56:22|2026-10-19T12:56:22|2026-10-19T13:20:42|COMPLETED|00:24:20|<anonymous>
571917|fb2-JB|2026-10-19T13:21:43|2026-10-19T13:21:43|2026-10-19T13:33:49|COMPLETED|00:12:06|<anonymous>
571969|fb2-JA|2026-10-19T13:33:56|2026-10-19T13:33:56|2026-10-19T13:41:16|COMPLETED|00:07:20|<anonymous>
```

## es1

### `es1/PREREG.md`

- sha256 `846af1868e510ab84446ebef507984fbf88f31010223db9f2e307c90bc424e22`; modified 2026-10-19T04:52:29+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `es1/data/splits.json`

- sha256 `8c956f266cea8d2746a02e91da426227a88d871f345b4e9f3721fc800a1aaa00`; modified 2026-10-19T04:25:30+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of es1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
569465|les1-G0|2026-10-19T04:33:26|2026-10-19T04:33:26|2026-10-19T04:34:13|FAILED|00:00:47|<anonymous>
569474|les1-G0b|2026-10-19T04:36:19|2026-10-19T04:36:20|2026-10-19T04:37:19|COMPLETED|00:00:59|<anonymous>
569483|les1-J-A|2026-10-19T04:37:33|2026-10-19T04:37:34|2026-10-19T04:44:26|COMPLETED|00:06:52|<anonymous>
569503|les1-J-B|2026-10-19T04:42:09|2026-10-19T04:44:27|2026-10-19T04:44:53|COMPLETED|00:00:26|<anonymous>
```

## fb3

### `fb3/PREREG.md`

- sha256 `2f4266317ceedcb9b7edb2cd72faa21eeb502a1df5a156b5775cddd43dd0e035`; modified 2026-10-20T04:39:15+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `fb3/prereg/C_predictions.md`

- sha256 `b81c10cbb37bc76c2ee741cad965be65f4624b2b4cf6365ab3d0875f108c30b2`; modified 2026-10-19T23:41:59+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of fb3** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
577197|fb3-F0|2026-10-20T00:30:33|2026-10-20T00:30:34|2026-10-20T00:31:27|COMPLETED|00:00:53|<anonymous>
580364|fb3-F1|2026-10-20T03:50:18|2026-10-20T03:50:41|2026-10-20T04:38:44|TIMEOUT|00:48:03|<anonymous>
581132|fb3-F2|2026-10-20T04:39:03|2026-10-20T04:39:32|2026-10-20T04:44:07|COMPLETED|00:04:35|<anonymous>
```

## exp23

### `exp23/A/outputs/hashes.json`

- sha256 `f16529b5123483b9564f8ca82ac0901618a9377061cb26e7f2d5a313f3392621`; modified 2026-10-21T01:49:13+08:00
- RUNLOG: no line logs this hash.

### `exp23/PREREG.md`

- sha256 `054cfacd575c240ce01e33d74dafbe5346959bef60cd9972cd33d0b44ac7d262`; modified 2026-10-21T01:57:37+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `exp23/results/phase0_hashes.json`

- sha256 `5de786ee3f7e1e91792167223e789dd1de1820f47c9c9f8104b8524e013cfa46`; modified 2026-10-19T20:47:51+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of exp23** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
576118|lem23a-AU|2026-10-19T21:37:43|2026-10-19T21:37:44|2026-10-19T21:39:33|COMPLETED|00:01:49|<anonymous>
576119|lem23b-G0|2026-10-19T21:37:44|2026-10-19T21:37:47|2026-10-19T21:38:53|COMPLETED|00:01:06|<anonymous>
576121|lem23a-CAL|2026-10-19T21:38:50|2026-10-19T21:39:33|2026-10-19T21:47:59|COMPLETED|00:08:26|<anonymous>
576196|lem23a-CAL1000|2026-10-19T21:51:08|2026-10-19T21:51:08|2026-10-19T21:56:24|COMPLETED|00:05:16|<anonymous>
576334|lem23a-T1|2026-10-19T22:24:09|2026-10-19T22:24:10|2026-10-19T22:45:09|COMPLETED|00:20:59|<anonymous>
576336|lem23a-T2|2026-10-19T22:24:14|2026-10-19T22:24:16|2026-10-19T22:43:07|FAILED|00:18:51|<anonymous>
576401|lem23a-T3|2026-10-19T22:46:06|2026-10-19T22:46:07|2026-10-19T23:11:02|COMPLETED|00:24:55|<anonymous>
576402|lem23a-S1|2026-10-19T22:46:07|2026-10-19T22:46:10|2026-10-19T23:02:28|COMPLETED|00:16:18|<anonymous>
576485|lem23a-J1|2026-10-19T23:03:05|2026-10-19T23:03:05|2026-10-19T23:14:00|COMPLETED|00:10:55|<anonymous>
576524|lem23a-J2|2026-10-19T23:14:02|2026-10-19T23:15:12|2026-10-19T23:23:15|COMPLETED|00:08:03|<anonymous>
579687|lem23a-D1|2026-10-20T03:18:21|2026-10-20T03:18:32|2026-10-20T03:24:38|COMPLETED|00:06:06|<anonymous>
588978|lem23a-L12|2026-10-20T19:17:54|2026-10-20T19:17:54|2026-10-20T19:40:21|COMPLETED|00:22:27|<anonymous>
589065|lem23a-L12M|2026-10-20T19:41:41|2026-10-20T19:41:41|2026-10-20T19:50:11|COMPLETED|00:08:30|<anonymous>
590312|lem23a-L16|2026-10-20T22:27:05|2026-10-20T22:27:05|2026-10-20T22:27:49|FAILED|00:00:44|<anonymous>
590346|lem23a-L16b|2026-10-20T22:29:07|2026-10-20T22:29:08|2026-10-20T22:53:39|COMPLETED|00:24:31|<anonymous>
590768|lem23a-L16M|2026-10-20T22:54:22|2026-10-20T22:54:43|2026-10-20T23:02:52|COMPLETED|00:08:09|<anonymous>
591475|lem23a-L20|2026-10-21T01:17:10|2026-10-21T01:17:10|2026-10-21T01:43:56|COMPLETED|00:26:46|<anonymous>
591557|lem23a-L20M|2026-10-21T01:44:58|2026-10-21T01:44:58|2026-10-21T01:53:00|COMPLETED|00:08:02|<anonymous>
```

## pb2

### `pb2/PREREG.md`

- sha256 `c86a74a769ee488530db36bf36cd86f91020c303e01e68807d10c13b4a0ccfec`; modified 2026-10-20T04:58:44+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `pb2/prereg/C_predictions.md`

- sha256 `15a6a38c4eb4e8583fa47d67b2b5f6d3cde65ba1a95839b296f022a78d74f06a`; modified 2026-10-19T23:40:49+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of pb2** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
580207|pb2-P0J|2026-10-20T03:42:32|2026-10-20T03:42:40|2026-10-20T03:46:14|COMPLETED|00:03:34|<anonymous>
580409|pb2-T1|2026-10-20T03:52:48|2026-10-20T03:53:41|2026-10-20T04:29:08|COMPLETED|00:35:27|<anonymous>
580977|pb2-E1|2026-10-20T04:29:29|2026-10-20T04:29:31|2026-10-20T04:40:13|FAILED|00:10:42|<anonymous>
581241|pb2-S1|2026-10-20T04:46:02|2026-10-20T04:46:33|2026-10-20T04:56:40|COMPLETED|00:10:07|<anonymous>
```

## pb3

### `pb3/PREREG.md`

- sha256 `dfd2a36edbdba3dc2715aa98a702529de407f22a5f65cc9a60431beb11e44d00`; modified 2026-10-20T23:44:48+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of pb3** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
588980|pb3-G|2026-10-20T19:18:22|2026-10-20T19:18:22|2026-10-20T19:18:41|FAILED|00:00:19|<anonymous>
588982|pb3-G2|2026-10-20T19:19:43|2026-10-20T19:19:43|2026-10-20T19:25:17|COMPLETED|00:05:34|<anonymous>
589012|pb3-T1|2026-10-20T19:28:27|2026-10-20T19:28:27|2026-10-20T20:02:54|COMPLETED|00:34:27|<anonymous>
589013|pb3-T2|2026-10-20T19:28:27|2026-10-20T19:28:28|2026-10-20T19:38:07|COMPLETED|00:09:39|<anonymous>
590955|pb3-F1|2026-10-20T23:10:00|2026-10-20T23:10:00|2026-10-20T23:41:08|COMPLETED|00:31:08|<anonymous>
```

## pt1

### `pt1/PREREG.md`

- sha256 `5cea02a82beda954170f22bb439faa6d854c42745b5bab80056615157099c8d5`; modified 2026-10-21T01:07:01+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `pt1/prereg/C_predictions.md`

- sha256 `ffeabb4ee5273617919f6daf31701582134dcafe397c5c4408a97e1f2938b03a`; modified 2026-10-21T00:34:32+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

**Slurm jobs of pt1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
591348|pt1-G|2026-10-21T00:44:15|2026-10-21T00:44:15|2026-10-21T00:46:35|COMPLETED|00:02:20|<anonymous>
```

## pf1

### `pf1/PREREG.md`

- sha256 `265d2ab59d1ef7d64d6e5f198d0c1ffbe020136ecc190f33af0ccca309e6995c`; modified 2026-10-21T04:26:14+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `pf1/forecast.json`

- sha256 `6b1f3cfe05272b71a454b26f9471ab4c8014abf08bd0a0434e83920f0585920a`; modified 2026-10-21T03:27:26+08:00
- RUNLOG lines logging this hash:

```
- PF1 2026-10-21 03:27 FORECAST FROZEN: pf1/forecast.json sha256 6b1f3cfe05272b71a454b26f9471ab4c8014abf08bd0a0434e83920f0585920a (S3 = bad_rehab, N = 708; COPY-NEAR = chemicals; coverage flag E_base bad_tattoo 10.5%). No training job submitted before this line.
- PF1 2026-10-21 04:26 FINISHED (0.905 of 1.2 GPU-h). S3 = bad_rehab (seed 20261021 draw from camping, medical, rehab, driving); forecast hash 6b1f3cfe… logged 03:27, first training submission 03:28. Gates pass both seeds; rho_seed 0.992 [0.95, 1.00]; ranking ceiling ~1.00. Spearman: A3 0.69 [0.27, 0.92] (top-5 4/5), G1 0.82, A2 0.81, A4 0.77, A6 0.75, A1 0.72, A5 0.71, L1 0.61, T1 0.37; ANCHOR-ONLY 0.33, COPY-NEAR 0.36. A3 - ANCHOR-ONLY +0.36 [0.07, 0.67]. F1-F10 all TRUE; Brier lead 0.211, mine 0.234. Report pf1/RESULTS.md. Scratch and lock removed.
```

### `pf1/prereg/C_predictions.md`

- sha256 `c61a09c2f30ef8cb24ea1b58e8dfb7d966cd3221e3ebdd0512fbb7f2ad7210fd`; modified 2026-10-21T02:09:25+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `pf1/results/train_sets.json`

- sha256 `b35100d70f8d9a69d6b8abb2b688cafd0ddfb4da3753b9402d679740a1d8168f`; modified 2026-10-21T03:28:13+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of pf1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
591670|pf1-G|2026-10-21T02:14:52|2026-10-21T02:14:52|2026-10-21T02:18:26|COMPLETED|00:03:34|<anonymous>
591880|pf1-T1|2026-10-21T03:28:26|2026-10-21T03:28:26|2026-10-21T03:50:59|FAILED|00:22:33|<anonymous>
591925|pf1-T1b|2026-10-21T03:54:07|2026-10-21T03:54:07|2026-10-21T04:22:20|COMPLETED|00:28:13|<anonymous>
```

## he1

### `he1/PREREG.md`

- sha256 `951f23ed1ac57f3d66ad2737258132a37e950b8931891aa9b02959f4fc698c81`; modified 2026-10-21T14:41:29+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `he1/halves.json`

- sha256 `9791b5203b3601f6728e72b8ea1ac4be3cff0201c2095459c06dd3bfc8c34533`; modified 2026-10-21T11:30:02+08:00
- RUNLOG: no line logs this hash.

### `he1/prereg/C_predictions.md`

- sha256 `1700812327832f9048ba76dbdfee94124b0bedfb519a57753cdb65e9340f3289`; modified 2026-10-21T11:29:56+08:00 (exported after redaction; the hash is of the original)
- RUNLOG lines logging this hash:

```
- HE1 2026-10-21 11:30 probability column written before the halves, groups or any Part A number: he1/prereg/C_predictions.md sha256 1700812327832f90…; pre-flight PASS (9 adapters match their records: FB2 hashes.json, FB1, PB3 sample rows); PB2's checkpoint folder is gone -> twin(K-every) = PB3 Rev-F, R0 = PB3 R0-F, twin(K-adv1) = AN1 Rpet-F if it exists at Part B, else Rev-F (other anchor).
```

### `he1/results/groups.json`

- sha256 `88ef76cd7602fe78e6a31b0610c90c2f272f6dc1c630486c0b18c980bdbb7ccf`; modified 2026-10-21T11:30:02+08:00
- RUNLOG: no line logs this hash.

### `he1/results/preflight.json`

- sha256 `abba0dc986ad8dc1b6829d17a4e4af120c1f468532e9729f0ed43dd39d5169b4`; modified 2026-10-21T11:28:43+08:00
- RUNLOG: no line logs this hash.

### `he1/results/twins.json`

- sha256 `bcec20e2fc84a01cc4fbeb242f0d65a805ecd4fe9bd72ed2c92caab0516b5dbb`; modified 2026-10-21T12:32:28+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of he1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
593473|he1-A|2026-10-21T11:35:31|2026-10-21T11:35:31|2026-10-21T11:36:58|COMPLETED|00:01:27|<anonymous>
593674|he1-B|2026-10-21T12:33:27|2026-10-21T12:33:27|2026-10-21T12:46:52|COMPLETED|00:13:25|<anonymous>
593704|he1-JF|2026-10-21T12:49:56|2026-10-21T12:49:58|2026-10-21T13:07:17|COMPLETED|00:17:19|<anonymous>
594255|he1-J2|2026-10-21T14:21:05|2026-10-21T14:21:05|2026-10-21T14:28:12|COMPLETED|00:07:07|<anonymous>
```

## an1

### `an1/PREREG.md`

- sha256 `765308536a93a732dee18f1b8f92f67e02b9850246569e7eafbf226d11b07a93`; modified 2026-10-21T12:37:05+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `an1/prereg/C_predictions.md`

- sha256 `99906101bc024d23ae29921b6d7d2cc7e25a9b07712c2c683e42a8373690da9b`; modified 2026-10-21T11:43:31+08:00 (exported after redaction; the hash is of the original)
- RUNLOG lines logging this hash:

```
- AN1 2026-10-21 11:43 probability column written before any AN1 GPU job: an1/prereg/C_predictions.md sha256 99906101bc024d23…; §2.0 pre-flight PASS (an1/results/preflight.json): F file, R0-F / Rev-F adapters, PB3 samples (691 x 8), PB2 base samples (covers the 691 and class 3; 0 R*), K-adv1 rows, class 3, main anchor order = PB3 Rev-F's and K-adv1's visits; hashes recorded where no record existed.
```

### `an1/results/adapter_hashes.json`

- sha256 `1c3cbabff27fdc77da8b3f46c9364279ef86545306fddd79ab2f3e32a1ccc40f`; modified 2026-10-21T12:24:45+08:00
- RUNLOG: no line logs this hash.

### `an1/results/forecast.json`

- sha256 `869d07b3958efe217f1244b4175b0acecdcba832caf10932d56608ae0968274f`; modified 2026-10-21T11:52:51+08:00
- RUNLOG lines logging this hash:

```
- AN1 2026-10-21 11:53 FORECAST FROZEN: an1/results/forecast.json sha256 869d07b3958efe217f1244b4175b0acecdcba832caf10932d56608ae0968274f (PT1-A3 fit b -0.170 c +0.031; PT1-A2 b -0.166 c +0.069; fitted on Rev-F, Rev-S (22 cells, finance from PB3), Rev-S3 (PF1 2-run mean)). Pattern (PT1-A1) N; A3 N; T1 N. No training job submitted before this line.
- AN1 2026-10-21 11:58 PREREG stage 2 assembled before any training (A2: A_LT counts, split 120/120, coverage drops legal 17 / tenancy 22, format parity 1.065 / 1.001 (no flag), closeness values, Pattern N, forecast sha256 869d07b3…; B2: training, evaluation rows (907, sha256 790ed846…), gates, analysis). Trainer tiny CPU test PASS (10 steps, no assertion failure). PREREG sha256 245e639af526f22a…
```

### `an1/results/preflight.json`

- sha256 `72bbeda23427cdc4a5ac56cb99a671b505afb2c374b42d42b50a7ee3a6c2b618`; modified 2026-10-21T11:43:13+08:00
- RUNLOG: no line logs this hash.

### `an1/results/train_sets.json`

- sha256 `8656013df48a158dc55f2a5a88ce5843f3d9479559b424b186e4fafe6b880b88`; modified 2026-10-21T11:54:20+08:00
- RUNLOG: no line logs this hash.

### `an1/tmp/tiny/results/train_sets.json`

- sha256 `8656013df48a158dc55f2a5a88ce5843f3d9479559b424b186e4fafe6b880b88`; modified 2026-10-21T11:54:20+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of an1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
593504|an1-G|2026-10-21T11:47:42|2026-10-21T11:47:42|2026-10-21T11:51:21|COMPLETED|00:03:39|<anonymous>
593555|an1-T|2026-10-21T11:58:42|2026-10-21T11:58:42|2026-10-21T12:33:07|COMPLETED|00:34:25|<anonymous>
```

## ud1

### `ud1/PREREG.md`

- sha256 `ffdbae9c6607c9a48e9e624333260caa7d15b5425c0fa0363c80ec2dfea7cc5a`; modified 2026-10-21T14:54:24+08:00 (exported after redaction; the hash is of the original)
- RUNLOG: no line logs this hash.

### `ud1/prereg/C_predictions.md`

- sha256 `e1dbd51dccb32e233b35a13555e3c51e50ef0704e8c89a31945ddfbccf2959cd`; modified 2026-10-21T12:01:01+08:00 (exported after redaction; the hash is of the original)
- RUNLOG lines logging this hash:

```
- UD1 2026-10-21 12:01 probability column written before any UD1 GPU job: ud1/prereg/C_predictions.md sha256 e1dbd51dccb32e23… (disclosed: written after HE1 Part A's calibration failure, which fixes U11/U12 near 0).
```

### `ud1/results/preflight.json`

- sha256 `24dc153b30f81a2a0bb1e70bf585e590e12a551a3c6f3aaaa3c88fedc7970cdd`; modified 2026-10-21T12:02:13+08:00
- RUNLOG: no line logs this hash.

**Slurm jobs of ud1** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):

```
JobID|JobName|Submit|Start|End|State|Elapsed|NodeList
593577|ud1-P0|2026-10-21T12:08:08|2026-10-21T12:08:09|2026-10-21T12:09:46|FAILED|00:01:37|<anonymous>
593591|ud1-P0b|2026-10-21T12:10:40|2026-10-21T12:10:40|2026-10-21T12:14:11|COMPLETED|00:03:31|<anonymous>
593631|ud1-T1|2026-10-21T12:16:23|2026-10-21T12:16:23|2026-10-21T12:26:15|COMPLETED|00:09:52|<anonymous>
593658|ud1-T2|2026-10-21T12:26:53|2026-10-21T12:26:53|2026-10-21T12:44:52|COMPLETED|00:17:59|<anonymous>
593703|ud1-M|2026-10-21T12:47:26|2026-10-21T12:47:26|2026-10-21T12:55:28|COMPLETED|00:08:02|<anonymous>
594256|ud1-S|2026-10-21T14:21:05|2026-10-21T14:21:06|2026-10-21T14:28:35|FAILED|00:07:29|<anonymous>
594287|ud1-SJ|2026-10-21T14:30:19|2026-10-21T14:30:19|2026-10-21T14:49:13|COMPLETED|00:18:54|<anonymous>
```

## Tony's off-cluster records

_For Tony: the hashes he relayed at the time, to compare with the ones above._
