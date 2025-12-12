"""Sentiment analysis using HuggingFace transformers."""

from transformers import pipeline
import pandas as pd
from typing import List, Tuple, Dict
import streamlit as st


@st.cache_resource
def load_sentiment_model():
    """
    Load the sentiment analysis model.
    
    Returns:
        Sentiment analysis pipeline
    """
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=False
    )


def analyze_sentiment_batch(texts: List[str], batch_size: int = 32) -> List[Dict[str, any]]:
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts: List of text strings to analyze
        batch_size: Number of texts to process at once
        
    Returns:
        List of dictionaries with 'label' and 'score' keys
        
    Example:
        >>> results = analyze_sentiment_batch(["Great food!", "Terrible service"])
        >>> results[0]['label']
        'POSITIVE'
    """
    if not texts:
        return []
    
    model = load_sentiment_model()
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
    
    # Normalize labels to lowercase
    for result in results:
        label = result['label']
        if label == 'POSITIVE':
            result['label'] = 'positive'
        elif label == 'NEGATIVE':
            result['label'] = 'negative'
        elif label == 'NEUTRAL':
            result['label'] = 'neutral'
        else:
            # Handle any other labels
            result['label'] = label.lower()
    
    return results


def add_sentiment_to_df(df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
    """
    Add sentiment analysis columns to dataframe.
    
    Args:
        df: Dataframe with text column
        text_column: Name of the column containing text to analyze
        
    Returns:
        Dataframe with added 'sentiment' and 'sentiment_confidence' columns
        
    Example:
        >>> df = pd.DataFrame({'cleaned_text': ['Great food!', 'Bad service']})
        >>> result = add_sentiment_to_df(df)
        >>> 'sentiment' in result.columns
        True
    """
    if df.empty or text_column not in df.columns:
        return df.copy()
    
    result_df = df.copy()
    texts = result_df[text_column].tolist()
    
    # Analyze sentiment
    sentiment_results = analyze_sentiment_batch(texts)
    
    # Extract labels and scores
    sentiments = [r['label'] for r in sentiment_results]
    confidences = [r['score'] for r in sentiment_results]
    
    result_df['sentiment'] = sentiments
    result_df['sentiment_confidence'] = confidences
    
    return result_df

