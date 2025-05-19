import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import warnings
import sacrebleu

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The SacreBLEU.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*")

class TextComparison:
    """
    A class for comparing two text inputs using various similarity metrics.
    The BERT model is initialized once in the constructor to avoid reinstantiating it with each call.
    """
    
    def __init__(self, bert_model_name="bert-base-uncased"):
        """
        Initialize the TextComparison class with a BERT model.
        
        Args:
            bert_model_name (str): The name of the BERT model to use.
        """
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = AutoModel.from_pretrained(bert_model_name)
        
        # Initialize Rouge
        self.rouge = Rouge()
        
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
    
    def _get_bert_embedding(self, text):
        """
        Get BERT embeddings for a given text.
        
        Args:
            text (str): The input text.
            
        Returns:
            numpy.ndarray: The BERT embedding vector.
        """
        # Add special tokens and return tensors
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token embedding as the sentence representation
        # This is a common approach for sentence-level embeddings
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding
    
    def bert_cosine_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts using BERT embeddings.
        
        Args:
            text1 (str): The first text input.
            text2 (str): The second text input.
            
        Returns:
            float: The cosine similarity score (0-1).
        """
        embedding1 = self._get_bert_embedding(text1)
        embedding2 = self._get_bert_embedding(text2)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)  # Convert to native Python float
    
    def rouge_scores(self, text1, text2):
        """
        Calculate ROUGE scores between two texts.
        
        Args:
            text1 (str): The first text input (reference).
            text2 (str): The second text input (hypothesis).
            
        Returns:
            dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        try:
            scores = self.rouge.get_scores(text2, text1)[0]
            # Extract f-measures for each ROUGE metric
            return {
                'rouge1': scores['rouge-1']['f'],
                'rouge2': scores['rouge-2']['f'],
                'rougeL': scores['rouge-l']['f']
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def bleu_score(self, text1, text2):
        """
        Calculate BLEU score between two texts.
        
        Args:
            text1 (str): The first text input (reference).
            text2 (str): The second text input (hypothesis).
            
        Returns:
            float: The BLEU score (0-1).
        """
        # Tokenize the texts
        reference = [nltk.word_tokenize(text1.lower())]
        hypothesis = nltk.word_tokenize(text2.lower())
        
        # Calculate BLEU score with smoothing
        try:
            return sentence_bleu(reference, hypothesis, smoothing_function=self.smoothing)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
            
    def bleu4_score(self, text1, text2):
        """
        Calculate BLEU-4 score between two texts using SacreBLEU.
        
        Args:
            text1 (str): The first text input (reference).
            text2 (str): The second text input (hypothesis).
            
        Returns:
            float: The BLEU-4 score (0-1).
        """
        try:
            # SacreBLEU expects a list of references and a single hypothesis
            references = [text1]
            hypothesis = text2
            
            # Calculate BLEU-4 score using SacreBLEU
            bleu = sacrebleu.corpus_bleu([hypothesis], [[ref] for ref in references])
            
            # Return the score normalized to 0-1 range
            return bleu.score / 100.0
        except Exception as e:
            print(f"Error calculating BLEU-4 score: {e}")
            return 0.0
    
    def jaccard_similarity(self, text1, text2):
        """
        Calculate Jaccard similarity between two texts.
        
        Args:
            text1 (str): The first text input.
            text2 (str): The second text input.
            
        Returns:
            float: The Jaccard similarity score (0-1).
        """
        # Tokenize the texts
        tokens1 = set(nltk.word_tokenize(text1.lower()))
        tokens2 = set(nltk.word_tokenize(text2.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def compare_texts(self, text1, text2):
        """
        Compare two texts using multiple similarity metrics.
        
        Args:
            text1 (str): The first text input.
            text2 (str): The second text input.
            
        Returns:
            dict: A dictionary containing all similarity scores.
        """
        # Calculate all metrics
        bert_similarity = self.bert_cosine_similarity(text1, text2)
        rouge = self.rouge_scores(text1, text2)
        bleu = self.bleu_score(text1, text2)
        bleu4 = self.bleu4_score(text1, text2)
        jaccard = self.jaccard_similarity(text1, text2)
        
        # Combine all metrics into a single dictionary
        results = {
            'bert_cosine_similarity': bert_similarity,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'bleu': bleu,
            'bleu4': bleu4,
            'jaccard': jaccard
        }
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the comparison class
    comparator = TextComparison()
    
    # Example texts
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast brown fox leaps over a sleepy canine."
    
    # Compare the texts
    results = comparator.compare_texts(text1, text2)
    
    # Print the results
    print("Similarity Metrics:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")