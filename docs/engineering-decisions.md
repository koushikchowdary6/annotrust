# Engineering decisions

The sections "Problem", "Alternatives", "Choice", and "Tested how" describe what the code does. "Why I chose this" and "Open questions" are left for the author.

## Dawid-Skene versus majority vote

### Problem

A scan has to name a consensus label per item and an error pattern per annotator. The input is a table of `item_id`, `annotator_id`, and `label`, with missing pairs allowed.

### Alternatives

- Majority vote. The consensus is the most common label. Ties take the lowest class index. This produces no confusion matrix.
- Dawid-Skene EM (Dawid and Skene, 1979), implemented in `fit_dawid_skene` (`src/annotrust/dawid_skene.py`). The posterior is initialized from a smoothed label count. The M-step updates the class prior and each annotator's confusion matrix with Dirichlet smoothing `ScanConfig.smoothing` (default 0.05). The E-step is the normalized product of the prior and the emission probabilities.

### Choice

The consensus label is the Dawid-Skene posterior mode (`scan` takes `argmax` on the posterior). Targeted flips are read from that annotator's fitted confusion matrix in `find_targeted_flips` (`src/annotrust/poison.py`). The careless check uses a second number: the fraction of the annotator's labels that equal the posterior mode. That fraction is what the text report calls agreement.

### Tested how

`tests/test_dawid_skene.py` builds 90 items, 3 classes, four annotators who match the latent class, and one annotator shifted by one class. The posterior mode matches the latent class on more than 95% of items, and the shifted annotator's confusion row peaks on the class they emit. `tests/test_poison.py` checks the flip rule on a hand-built matrix, including a constant-label matrix that is not called a targeted flip. `scripts/benchmark.py` writes pooled precision and recall to `docs/benchmark.md`.

### Why I chose this

TODO

### Open questions

TODO

## Collusion score

### Problem

Cohen's kappa is high for any pair who agree, including two careful annotators who both match the truth. The failure mode that matters here is a pair who agree with each other on labels the consensus rejects.

### Alternatives

- Flag every pair whose kappa is above a cutoff.
- Test whether agreement exceeds chance with a significance test, and stop there.
- Compare pairwise kappa with each annotator's kappa against the consensus, and require the shared answers to be wrong.

### Choice

`score_pairs` in `src/annotrust/collusion.py`, on the items both annotators labeled. C is the Dawid-Skene mode.

- excess = kappa(A, B) - max(kappa(A, C), kappa(B, C))
- wrong_agreement = (count of A = B ≠ C) / (count of A ≠ C and B ≠ C), or 0 if the denominator is 0
- score = `collusion_excess_weight` * max(0, excess) + `collusion_wrong_weight` * wrong_agreement

Defaults are 0.7 and 0.3. A pair is flagged when overlap ≥ `min_pair_overlap` (10), kappa ≥ 0.75, excess ≥ 0.15, the joint-wrong count ≥ 3, and score ≥ 0.25. Severity is high when score ≥ 0.45 and kappa ≥ 0.85, otherwise medium. The max, not the mean, is the consensus term: copying an annotator who already matches the consensus does not produce a positive excess.

### Tested how

`tests/test_collusion.py` flags two annotators who always emit the next class and does not flag annotators who match the consensus. `tests/test_scan.py::test_copying_an_accurate_leader_is_not_called_collusion` plants a leader at accuracy 0.95 with copy rate 1.0 and checks that neither partner's primary label is `colluding`. The collude-rate table in `docs/benchmark.md` is the same check across seeds 0-4: at copy rate 0.25 and 0.50, recall is 0.00.

### Why I chose this

TODO

### Open questions

TODO

## Thresholds

### Problem

The detectors need a cutoff. The annotation file does not say which annotators were planted, and a real file will not say either.

### Alternatives

- Return scores and no flags.
- Fit the cutoffs on the simulator seeds that `docs/benchmark.md` reports.
- Fix the cutoffs in one place, publish the miss rate, and leave the knobs on `ScanConfig`.

### Choice

The cutoffs are the defaults on `ScanConfig` in `src/annotrust/config.py`. They were not fit by a search over seeds 0-4.

- Poisoning: rate ≥ 0.40, concentration ≥ 0.60, other-class accuracy ≥ 0.70, source count ≥ 12, expected flip count ≥ 6. High severity at rate ≥ 0.70.
- Careless: at least 12 labels and agreement ≤ 0.55, or a streak of at least max(12, half the labels) with normalized entropy ≤ 0.35 and agreement ≤ 0.85. High severity at agreement ≤ 0.40. A poisoning flag suppresses the careless flag.
- Fast: at least 8 gaps and at least half of them under 2 seconds. Severity is low.
- Items: normalized entropy ≥ 0.65, or a majority overturned at confidence ≥ 0.60, with at least 3 annotators. `--fail-on` does not look at item severity. `ScanResult.fails` counts annotators only.

### Tested how

`tests/test_poison.py` and `tests/test_scan.py`. The weak-flip test plants poison rate 0.15 and asserts the poisoner is not the primary `poisoning` label. The majority-panel test (seed 0, five poisoners, two honest annotators, six labels per item, poison rate 1.0) asserts the planted poisoner is not called poisoning and `honest_00` is. `docs/benchmark.md` is the pooled measurement: poison rate 0.15 has recall 0.00 (tp=0, fp=0, fn=5); the majority panel has pooled precision 0.38 and recall 0.20 (tp=5, fp=8, fn=20).

### Why I chose this

TODO

### Open questions

TODO

## Offline only

### Problem

The input is a label file for training data. Sending it off the machine is a separate decision from auditing it, and a CI job should not need a network or an account.

### Alternatives

- Call an annotation product's API at scan time.
- Download a model or a hosted scorer.
- Import cleanlab unconditionally and fail install when it is absent.

### Choice

Readers in `src/annotrust/load.py` and `src/annotrust/importers/` open a local path. The HTML report in `src/annotrust/report.py` inlines its CSS and does not reference a URL. `compare_item_scores` imports cleanlab only when the caller passes `use_cleanlab` or `--cleanlab`. The package metadata sets `Private :: Do Not Upload` and does not publish to PyPI.

### Tested how

`tests/test_offline.py` reads every module under `src/annotrust` and fails if it finds an HTTP client import. `tests/test_scan.py::test_reference_seed_recovers_every_planted_role` renders HTML and asserts the document contains neither `http://` nor `https://`. `tests/test_cleanlab.py` covers the missing-extra error and a fake `cleanlab.rank` module.

### Why I chose this

TODO

### Open questions

TODO
