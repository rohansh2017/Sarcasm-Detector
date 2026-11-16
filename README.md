# Sarcasm Detector

Seven classifiers trained on the same headline dataset to answer one question: how much does model complexity buy you when detecting sarcasm in short text? Team project by three Rutgers students.

<!-- TODO: add the course name, and one line on which parts you built -->

## Results

Binary classification (sarcastic or not) on 21,464 training headlines, 716 validation and 966 test.

| Model | Validation accuracy | Test accuracy | Test F1 |
|---|---|---|---|
| **MLP (256, 128) on TF-IDF + features** | **84.6%** | **84.8%** | **0.829** |
| BiLSTM (word + part-of-speech embeddings) | 82.8% | 82.8% | 0.813 |
| CNN | 82.8% | 82.5% | 0.802 |
| SVM | 82.5% | 81.7% | 0.801 |
| Logistic regression | 83.1% | 81.5% | 0.799 |
| Naive Bayes | 79.9% | 80.5% | 0.775 |
| Random forest | 82.0% | 80.3% | 0.786 |

The spread is small: about four points of test accuracy separate the best model from the worst. A plain MLP over TF-IDF and hand-built features beat both sequence models, which suggests that on short headlines the engineered features carry most of the signal.

Numbers are computed from the prediction files in this repo (`*_test_pred*.csv`, `*_valid_pred*.csv`).

## Approach

**Preprocessing** (`preprocessing.ipynb`)
- Lowercased text and normalised whitespace.
- Kept stop words, because their frequency differs between the two classes.
- Kept punctuation as separate tokens, and treated `...` differently from `.`.

**Features** (`feature_engineering.ipynb`)
- TF-IDF vectors.
- Sentiment score and length in words.
- Punctuation counts: exclamation marks, question marks, ellipses, quotes, dashes, and punctuation density.
- Part-of-speech tag counts and tag bigrams.

**Models**
- Classical: logistic regression, naive Bayes, SVM and random forest (scikit-learn).
- MLP: two hidden layers (256, 128) over TF-IDF plus the scaled hand-built features.
- CNN and BiLSTM in PyTorch. The BiLSTM combines 128-dimensional word embeddings with 32-dimensional part-of-speech embeddings and the auxiliary features.

**Evaluation**
- Hyperparameters chosen on the validation set; the test set was used only for the final numbers.

## Repository layout

| Path | What it is |
|---|---|
| `preliminaryAnalysis.ipynb` | Descriptive statistics on the raw data |
| `preprocessing.ipynb`, `feature_engineering.ipynb` | Data pipeline |
| `logitsticRegression.py`, `naiveBayes.py`, `svm.py`, `randomforest.py`, `mlp.py` | Classical models and the MLP |
| `cnn.ipynb`, `bilstm.ipynb` | PyTorch models |
| `stacking.ipynb` | Stacking ensemble experiment |
| `predict_sarcasm.py`, `get_predictions.ipynb` | Run saved models on new data |
| `*_preds.csv`, `*_predictions.csv` | Per-model predictions on each split |

## Run it

```bash
pip install pandas numpy scipy scikit-learn torch joblib
python mlp.py
```

The preprocessed splits and TF-IDF matrices are committed, so each model script runs without repeating the pipeline.
