# wikishop-toxic-comment-classification
NLP project to detect toxic user comments for moderation (F1-focused) using TF-IDF + LinearSVC and BERT embeddings.

## Project overview
Online store **Wikishop** launches a service where users can edit and expand product descriptions (similar to wiki platforms). Users can also leave comments on edits, so the platform needs an automated way to **detect toxic comments** and send them to moderation.

**Goal:** train a binary text classification model with **F1 ≥ 0.75**.

## Tech stack
- **Python** (Jupyter Notebook; kernel used: 3.9.x)
- **pandas**, **numpy**
- **scikit-learn** (Pipeline, GridSearchCV, cross-validation, metrics)
- **nltk** (tokenization, POS tagging, lemmatization)
- **transformers** (BERT tokenizer/model)
- **torch**
- **langdetect**
- **tqdm**
- **matplotlib**, **seaborn**

## Dataset
The notebook loads the dataset from a public CSV:

- `toxic_comments.csv` (loaded from an external source inside the notebook)

Main columns:
- `text` — comment text
- `toxic` — target label (1 = toxic, 0 = non-toxic)

Dataset stats from the notebook:
- **159,292** rows
- **3** columns
- no missing values detected

## Approach
1. **Data checks** (types, duplicates, missing values)
2. **Language detection**: ~97% of texts are English → English preprocessing and `bert-base-uncased` are appropriate.
3. **Text preprocessing**
   - lowercasing, URL/HTML/@/# removal
   - keeping latin chars + basic punctuation
   - POS-aware lemmatization (NLTK)
4. **Train/test split**
   - `train_test_split(test_size=0.2, stratify=y, random_state=42)`
5. **Models**
   - **TF-IDF (word)** + `LinearSVC` (baseline)
   - **TF-IDF (character n-grams 3–4)** + `LinearSVC` (robust to typos/obfuscation)
   - **BERT embeddings ([CLS])** + `LogisticRegression` / `LinearSVC`
     - embeddings were computed on a reduced sample for speed (4,000 train / 1,000 test)

## Results
**Best model:** **TF-IDF (char 3–4) + LinearSVC**


**Classification report (toxic class = 1):**
- precision: **0.7404**
- recall: **0.8373**
- F1: **0.7859**

Baseline (word TF-IDF + LinearSVC) achieved **CV F1 ≈ 0.7777 ± 0.0065**.

## How to run

### System requirements
- Python **3.9+**
- Jupyter Notebook / JupyterLab

### Installation
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install pandas numpy scikit-learn nltk langdetect transformers torch tqdm matplotlib seaborn jupyter
```
### NLTK resources
If you run the notebook for the first time, download NLTK data:

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")
```
