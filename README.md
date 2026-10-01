# xling-bci

Does an imagined-speech EEG decoder learn the concept, or the language?

Reanalysis of two public EEG datasets. No data is collected here.

## Status

Early. The pipeline runs end to end on synthetic data and the tests pass.
Neither real dataset has been downloaded yet, so the loaders are written against
the published file layouts and have not been run against actual files. Expect
them to need fixing.

## The question

Imagined-speech decoders are trained and evaluated per subject, and participants
are usually described by one label: the language they speak. It is worth asking
how much that label predicts. If a decoder trained on speakers of one language
fails on speakers of another, is that about the language, or about the person, or
about the amplifier?

Two numbers get reported: how much accuracy drops when the training group speaks
a different language, and how much accuracy varies between individuals who speak
the same one. Comparing those says whether the language label is doing any work.

## The four regimes

| # | Regime | What it isolates |
|---|---|---|
| 1 | Within subject, 5-fold | Ceiling, and a check that the pipeline decodes anything |
| 2 | Within language, leave one subject out | Transfer between people who share a language |
| 3 | Across language, leave one subject out | Same held-out subjects, trained on the other language group |
| 4 | Across dataset, same language | Spanish from one dataset tested on Spanish from the other |

Regimes 2 and 3 call the same function with different training pools. Same
features, same held-out subjects, one variable changed.

Regime 4 is a control. A failure in regime 3 has a boring explanation, which is
that the two groups were recorded differently. If same-language transfer across
two rigs fails just as badly, that is the explanation.

## Data

Neither dataset is in the repo. Download both and point `config.py` at them.

**Directional-word dataset.** Zenodo release accompanying *Scientific Data*
[10.1038/s41597-026-07809-9](https://doi.org/10.1038/s41597-026-07809-9).
22 subjects, 12 native Russian speakers and 10 native Spanish speakers, six
spatial-direction words spoken aloud and imagined, 38 channels at 500 Hz, epochs
from -0.5 to +1.0 s. Loader helpers at
[AvedikEkiz/inner-speech-bci](https://github.com/AvedikEkiz/inner-speech-bci).
Licensed CC BY-NC-ND, so a processed copy of the data cannot be redistributed.

```
directional/
  raw/{russian,spanish}/sub*/      EDF
  preprocessed/{russian,spanish}/  epochs .fif + events .xlsx
  metadata/subject_metadata.json
```

**Nieto et al. 2022.** OpenNeuro `ds003626`. 10 native Spanish speakers, 3
sessions each, 128-channel BioSemi at 1024 Hz, inner / pronounced / visualized
conditions, classes arriba abajo derecha izquierda. Fetch the `derivatives/` with
the scripts at
[N-Nieto/Inner_Speech_Dataset](https://github.com/N-Nieto/Inner_Speech_Dataset).

Regime 4 uses only the four concepts both datasets share. The Russian-only
"next" class is dropped everywhere.

## Running it

```
pip install -r requirements.txt
python tests/test_pipeline.py                      # synthetic, no data needed
python experiments/run.py --data synthetic         # whole pipeline end to end
python experiments/run.py --data real --regime 3
python experiments/run.py --data real --null       # label-shuffle null
```

`--no-align` turns off per-subject re-centering. Worth reporting both ways:
cross-subject EEG transfer without re-centering sits near chance for reasons
that have nothing to do with language.

## Method

Per-trial spatial covariance with OAS shrinkage, whitened to each subject's
Riemannian mean, projected to the tangent space at the identity, then multinomial
logistic regression.

This is the standard simple baseline rather than a deep network, on purpose. It
is hard to beat on samples this size, there is nothing to tune, and it runs on a
laptop in minutes, so a weak transfer number is more likely to be about transfer
than about how long I trained.

Re-centering uses no labels, so applying it to held-out subjects is not leakage.
It also makes the identity the right tangent-space reference, which turns the
projection into a closed form computed once instead of an iterative mean refitted
inside every fold.

## Layout

```
config.py              paths, target sampling rate, window, bandpass
xling/labels.py        concept vocabulary and the per-dataset event maps
xling/datasets.py      loaders for both datasets
xling/harmonize.py     filter, crop, resample, channel selection, montage matching
xling/decode.py        covariances, re-centering, features, the four regimes
xling/report.py        per-subject scores to summary numbers
xling/synthetic.py     stand-in recordings matching each dataset's documented shape
experiments/run.py     runs the regimes, prints a table, writes results/*.json
tests/test_pipeline.py end to end on synthetic data
```

## Known problems

1. **The loaders are unverified.** Written from the datasets' documentation, not
   run against the files.
2. **Nieto's events array column order has changed between releases.**
   `_resolve_nieto_columns` infers it from the documented code sets and raises if
   it cannot, rather than guessing.
3. **Regime 4 matches channels by position, not by name.** The directional set
   uses 10-10 names and Nieto uses BioSemi A1-D32, so there is no name overlap.
   Nearest-electrode matching within 25 mm paired 25 of 38 on the synthetic
   montages. Any regime 4 result needs to say this.
4. **Regime 4 confounds more than hardware.** Amplifier, montage, sampling rate,
   session structure and preprocessing all differ at once. It bounds the
   recording-differences explanation, it does not isolate it.
5. **Synthetic within-subject accuracy sits near ceiling** because each synthetic
   class carries its own frequency. The tests check the ordering across regimes,
   not the values. The synthetic numbers say nothing about inner speech.
6. **Both datasets ship their own preprocessing** and the two pipelines are not
   the same pipeline.
