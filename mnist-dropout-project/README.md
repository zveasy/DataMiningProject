# Dropout on MNIST · Experiment guide

Train a feed-forward digit classifier, vary its hidden-unit dropout rate, and follow the effect from validation loss to individual test errors. The completed study includes 15 runs, 300 epochs of measurements and nine held-out test evaluations. [Results at a glance](../README.md) · [Full experimental summary](presentation/experiment_summary.md)

## Install and verify

Start in `mnist-dropout-project/`. For a fresh environment, use Python 3.11 and install the dependencies:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The prepared workspace already contains the four MNIST files. For a fresh checkout, download the original IDX archive:

```sh
curl -fL https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset -o data/mnist.zip
unzip -o data/mnist.zip '*.idx*-ubyte' -d data/
```

Verify the files from the experiment directory and run the tests:

```sh
shasum -a 256 -c data/SHA256SUMS
python -m pytest -q
```

The checksum command should report four matches. The loader also accepts hyphenated filenames, nested directories and individual gzip files. It validates IDX headers, payload lengths, dimensions, sample counts and label coverage. [Kaggle dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) · [IDX loader](src/dataset.py)

## Run the offline demo

```sh
jupyter lab notebooks/live_demo.ipynb
```

Run all cells to load the three seed-42 checkpoints and draw the first test digit. Rerun **Draw a digit** for the next image. Restart the kernel and run all cells to replay the same sequence. The [experiment notebook](notebooks/dropout_experiment.ipynb) walks through the splits, learning curves, per-class metrics and misclassifications using the saved outputs.

## Reproduce the experiments

With the environment active and the dataset verified, run:

```sh
python -m src.train --device cpu
python -m src.evaluate
```

Training writes the best and final weights, epoch history and timing under `checkpoints/<run>/`. A rerun replaces matching run artifacts. Evaluation reads the best checkpoints and generates the [CSV tables](results/tables/), [300-DPI figures](results/figures/) and [presentation data](presentation/presentation_data.json). To use automatic device selection, omit `--device cpu`. The code tries CUDA, then Apple Silicon MPS, then CPU.

The recorded runs used four CPU threads on macOS arm64 with Python 3.13.5 (Conda), PyTorch 2.11.0, NumPy 1.26.4, pandas 2.2.2 and scikit-learn 1.5.2. Hardware and library versions can affect timings and numerical results.

To execute both notebooks as a final check and write the executed copies to a temporary directory:

```sh
python -m jupyter nbconvert --execute --to notebook --output-dir=/tmp notebooks/dropout_experiment.ipynb notebooks/live_demo.ipynb
```

## What the algorithm does

For paper retention probability p, each hidden activation h receives an independent binary mask r ~ Bernoulli(p). The original masked activation is r × h. At inference the authors activate all units and multiply their outgoing weights by p. PyTorch calls its argument the **drop probability** q = 1 − p. During training it passes (r × h)/p. The expected value conditional on h is h, so evaluation can pass h through unchanged. This expectation applies to the activation. The deterministic deep network only approximates averaging the predictions of the sampled networks.

Flattening occurs inside the model. The 784 input pixels feed 512 ReLU units, then 256 ReLU units, with dropout after each ReLU. A final linear layer produces ten logits z. Cross-entropy for true class y is log(sum(exp(z))) − z[y]. PyTorch computes this from logits with numerical stabilization. Backpropagation differentiates the minibatch mean loss through the sampled network. Gradient descent updates weights in a direction that reduces loss locally. Adam adjusts the step for each weight using running first and second gradient moments. The optional softmax in the demo converts logits into probabilities after training.

## Experimental design

- **Dropout comparison and sensitivity.** Train drop probabilities 0, 0.2 and 0.5 for 20 epochs with Adam, learning rate 0.001 and batch size 128. Use seeds 42, 43 and 44 for the primary comparison. Every rate within a seed starts with identical weights and receives the same shuffle order from a separate random generator. Repeatability is strongest on the same software and device.
- **Data.** Divide original pixel intensities by 255. Preserve N × 1 × 28 × 28 tensors until the model flattens them. Fix a seed-42 stratified split of the original 60,000 development images into 50,000 training and 10,000 validation examples. Keep all 10,000 original test examples for final evaluation.
- **Dataset-size study.** Compare nested stratified cohorts of 5,000, 15,000 and 50,000 training examples at seed 42 and all three rates. Validation examples are identical across runs. Equal epoch counts give smaller cohorts fewer optimizer updates, so the size effect also includes this change in training budget.
- **Metrics.** At each epoch end, compute train and validation cross-entropy and accuracy over each entire split with masks disabled, using the same weights. The generalization gap is training accuracy minus validation accuracy. A positive gap measures the observed advantage on training examples. A small gap can coincide with poor accuracy on both splits. Multiply fractional accuracies and gaps in CSVs by 100 for percentages and percentage points, respectively.
- **Selection.** Retain the lowest-validation-loss epoch and the final epoch separately. Runtime includes epoch-end scoring and checkpoint writing. The fixed-epoch curves show all 20 epochs. The comparison table uses the selected epoch from each run. In `configs/experiment.json`, a positive integer `patience` enables stopping after that many consecutive epochs without a new minimum. The primary setting is null.
- **Prediction analysis.** Evaluate all nine primary best checkpoints on the original test set once training is finished. Select the detailed diagnostic rate by lowest mean validation loss and use prespecified seed 42. Per-class reports and confusion counts are available for every primary model. Confidence is the maximum softmax probability. These scores alone do not establish calibration.
- **Uncertainty.** Tables report the arithmetic mean and sample standard deviation across initialization and shuffle seeds on one fixed split. SD describes run variability. All nine primary models share the same test images. The secondary size study has a single seed.

## Connection to the paper

Sections 4–5 motivate sampling subnetworks and backpropagating through retained activations. Section 6.1.1, Table 2 reports **1.25% MNIST test error** for the authors’ three-hidden-layer, 1,024-unit ReLU dropout network. Section 7.3 shows underfitting at low retention with fixed width. Section 7.4 finds that the benefit also changes with sample size. Appendix A discusses capacity, momentum and max-norm constraints. Appendix B.1 describes input dropout and a final training pass incorporating the validation examples. Our smaller hidden-only MLP uses Adam, a fixed holdout and a much shorter schedule. The results linked above are a controlled course experiment. [Srivastava et al., JMLR 15 (2014), 1929–1958](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

## Repository guide

| Location | Purpose |
|---|---|
| `configs/experiment.json` | Rates, seeds, optimization and cohort settings |
| `src/dataset.py`, `src/model.py` | Binary parsing, shared splits and configurable network |
| `src/train.py`, `src/utils.py` | Optimization, reproducible order, checkpoint selection and loading |
| `src/evaluate.py`, `src/visualize.py` | Held-out metrics, evidence export and 300-DPI plots |
| `checkpoints/<run>/` | Best and final weights, per-epoch CSV and timed configuration |
| `results/tables/`, `results/figures/` | Run-level and aggregate measurements, per-class reports and PNGs |
| `notebooks/dropout_experiment.ipynb` | Guided analysis of saved runs |
| `notebooks/live_demo.ipynb` | Repeated offline inference on reproducibly sampled digits |
| `presentation/` | Slide outline, speaking notes, measured summary and JSON data |
| `tests/` | Parser, split, training, inference, metrics and plotting checks |
