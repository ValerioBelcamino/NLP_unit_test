# Text Comparison Tool

This tool provides functionality to compare two text inputs using various similarity metrics.

## Features

- **BERT Cosine Similarity**: Compares semantic similarity using BERT embeddings
- **ROUGE Scores**: Includes ROUGE-1, ROUGE-2, and ROUGE-L metrics
- **BLEU Score**: Measures precision-oriented similarity
- **BLEU-4 Score (SacreBLEU)**: Provides a more robust BLEU metric with consistent tokenization
- **Jaccard Similarity**: Measures token-level overlap

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install numpy scikit-learn transformers torch rouge nltk sacrebleu
```

## Usage

```python
from text_comparison import TextComparison

# Initialize the comparison class (BERT model is loaded once)
comparator = TextComparison()

# Example texts to compare
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast brown fox leaps over a sleepy canine."

# Compare the texts with all metrics
results = comparator.compare_texts(text1, text2)

# Print the results
print("Similarity Metrics:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")

# Or use individual metrics
bert_similarity = comparator.bert_cosine_similarity(text1, text2)
rouge_scores = comparator.rouge_scores(text1, text2)
bleu = comparator.bleu_score(text1, text2)
bleu4 = comparator.bleu4_score(text1, text2)
jaccard = comparator.jaccard_similarity(text1, text2)
```

## Metrics Explanation

- **BERT Cosine Similarity**: Measures semantic similarity using contextual embeddings (0-1 scale)
- **ROUGE-1**: Overlap of unigrams between texts (0-1 scale)
- **ROUGE-2**: Overlap of bigrams between texts (0-1 scale)
- **ROUGE-L**: Longest common subsequence-based score (0-1 scale)
- **BLEU**: Precision-oriented metric commonly used in translation (0-1 scale)
- **BLEU-4 (SacreBLEU)**: Enhanced BLEU metric that uses consistent tokenization and focuses on 4-grams (0-1 scale)
- **Jaccard**: Token overlap ratio (intersection over union) (0-1 scale)

## Customization

You can customize the BERT model by specifying a different model name when initializing the class:

```python
# Use a different BERT model
comparator = TextComparison(bert_model_name="bert-large-uncased")
```