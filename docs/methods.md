# Methods

The citations below are the papers the implementation is taken from, or the literature the heuristic is answering. The collusion score and the numeric cutoffs are not a statistic from one of those papers. They are defined in `ScanConfig` and `src/annotrust/collusion.py`.

## Dawid-Skene consensus

Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Applied Statistics*, 28(1), 20-28.

For each item the model has a latent class. Each annotator has a confusion matrix, rows true class, columns emitted label. The implementation in `src/annotrust/dawid_skene.py` is that EM:

1. Initialize the item posterior from a smoothed count of the observed labels.
2. M-step: class prior is the mean posterior. Each confusion row is the posterior-weighted count of emitted labels, plus `ScanConfig.smoothing` (default 0.05) in every cell, then normalized.
3. E-step: the item posterior is proportional to the prior times the product of the emitting probabilities, in log space (`scipy.special.logsumexp`).
4. Stop when the log-likelihood moves by less than `tol` (default 1e-5), or after `max_iter` (default 100).

The reported consensus label is the posterior mode. Ties take the lowest class index. Class indices are the sorted label strings.

Raykar, Yu, Zhao, Valadez, Florin, Bogoni, and Moy (2010), "Learning from crowds", *Journal of Machine Learning Research* 11, 1297-1322, is the usual modern reference for this family of annotator models. This package does not fit their feature-dependent extension. It fits the original categorical EM.

A majority vote is the initialization, not the consensus. Majority vote does not produce a confusion matrix, and the flip check needs one.

The model assumes annotators are independent given the true class. A colluding pair, and a bloc that outvotes everyone else, both break that. The benchmark records the second case.

## Cohen's kappa and the collusion score

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37-46.

Artstein, R. and Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics*, 34(4), 555-596, is the survey for how kappa is and is not a measure of "true" agreement.

`cohen_kappa` builds the confusion table of two raters, including classes neither rater used (those cells are zero and do not change the value). When the chance term is 1, perfect agreement returns 1 and any other agreement returns 0.

On the items both annotators labeled, with C the Dawid-Skene mode:

- excess = kappa(A, B) - max(kappa(A, C), kappa(B, C))
- joint_wrong = count of items where A and B emitted the same label and that label is not C
- both_differ = count of items where A and B both differ from C
- wrong_agreement = joint_wrong / both_differ, or 0 when both_differ is 0
- score = 0.7 * max(0, excess) + 0.3 * wrong_agreement

A pair is flagged when the overlap is at least 10, kappa is at least 0.75, excess is at least 0.15, joint_wrong is at least 3, and the score is at least 0.25. The score is not a probability. It can exceed 1 when both kappas against the consensus are negative.

Using the max, rather than the mean, is deliberate: a partner who copies an annotator already matching the consensus has an excess near 0. The benchmark's "copying an accurate leader" row is that case, and recall is 0.

## Targeted flips

Xiao, H., Xiao, H., and Eckert, C. (2012). Adversarial label flips attack on support vector machines. *ECAI 2012*.

Paudice, A., Muñoz-González, L., Gyorgy, A., and Lupu, E. C. (2018). Label sanitization against label flipping poisoning attacks. *ECML PKDD 2018 Workshops*.

Steinhardt, J., Koh, P. W., and Liang, P. (2017). Certified defenses for data poisoning attacks. *Advances in Neural Information Processing Systems* 30.

Those papers are about the attack and about training-time defenses. Annotrust does not certify a model. It looks at one annotator's Dawid-Skene confusion row and flags a source class when all of the following hold:

- posterior mass on that source, over items the annotator labeled, is at least 12
- P(target | source) is at least 0.40 and the expected count of those flips is at least 6
- that probability is at least 0.60 of the row's off-diagonal mass
- accuracy on the other classes, weighted by their posterior mass, is at least 0.70

A constant-label spammer has several peaked rows and fails the other-class accuracy check. `tests/test_poison.py` locks that. Severity is high when the rate is at least 0.70, otherwise medium.

## Speed and streaks

Vuurens, J., de Vries, A. P., and Eickhoff, C. (2011). How much spam can you take? An analysis of crowdsourcing results to increase accuracy. *SIGIR 2011 Workshop on Crowdsourcing for Information Retrieval*.

Ipeirotis, P. G., Provost, F., and Wang, J. (2010). Quality management on Amazon Mechanical Turk. *Proceedings of HCOMP*.

Snow, R., O'Connor, B., Jurafsky, D., and Ng, A. Y. (2008). Cheap and fast — but is it good? Evaluating non-expert annotations for natural language tasks. *Proceedings of EMNLP*.

The speed rule is not their classifier. If at least eight inter-arrival gaps exist and at least half of them are under 2 seconds, the annotator gets a low-severity `fast` flag. Timestamps that do not vary are ignored. Numeric values with median above 1e11 are read as unix milliseconds; smaller numbers are seconds. Only the gaps are used.

A streak is the longest run of one label in time order, or in file order when there is no usable timestamp. A long streak with normalized label entropy at most 0.35 is folded into the careless flag when agreement with the consensus is at most 0.85. A spammer who only emits the true majority class, and therefore agrees with the consensus, is not flagged. That limitation is stated in the README.

Normalized entropy is Shannon entropy in bits divided by log2 of the number of classes (`scipy.stats.entropy`).

## Careless annotators

Agreement at or below 0.55, with at least 12 labels, is a careless flag. Severity is high at or below 0.40, otherwise medium. A poisoning flag suppresses the careless flag: the missed labels are explained by the flip. Primary label priority at the same severity is poisoning, then colluding, then careless, then fast.

## Item scores

For each item the report stores the posterior mode, its probability, normalized entropy, the majority label, and the majority share. An item is flagged when at least three annotators labeled it and either the normalized entropy is at least 0.65, or the majority label is not the mode and the mode probability is at least 0.60. The second case at probability at least 0.75 is medium severity. Otherwise a flagged item is low. These flags do not change `--fail-on`.

If `model_prob_<class>` columns exist and match the class set, a model label is the argmax. It is a `model_disagrees` reason when that probability is at least 0.85 and the class is not the consensus. A bare `model_label` with no probabilities is reported the same way, without a confidence.

Northcutt, C. G., Jiang, L., and Chuang, I. L. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research*, 70, 1373-1411. `--cleanlab` calls `cleanlab.rank.get_label_quality_scores` on the consensus labels and the model probabilities when the optional extra is installed. The core scan does not import it.
