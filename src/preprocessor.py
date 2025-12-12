"""Text cleaning, deduplication, and language detection module."""

import re
import pandas as pd
from typing import List
from langdetect import detect
from rapidfuzz import fuzz
import streamlit as st


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, URLs, emails, and normalizing whitespace.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
        
    Example:
        >>> clean_text("<p>Visit https://example.com or email test@test.com</p>")
        'Visit or email'
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def is_english(text: str) -> bool:
    """
    Detect if text is in English.
    
    Args:
        text: Text string to check
        
    Returns:
        True if text is English, False otherwise
        
    Example:
        >>> is_english("This is a review")
        True
        >>> is_english("Esta es una reseña")
        False
    """
    if not text or len(text.strip()) < 3:
        return False
    
    try:
        lang = detect(text)
        return lang == 'en'
    except Exception:
        # If detection fails, assume it's English to avoid false negatives
        return True


def find_duplicates(texts: List[str], threshold: int = 85) -> List[int]:
    """
    Find duplicate texts using fuzzy matching.
    
    Args:
        texts: List of text strings
        threshold: Similarity threshold (0-100) for considering texts as duplicates
        
    Returns:
        List of indices to remove (keeping first occurrence of duplicates)
        
    Example:
        >>> texts = ["Great food!", "Great food", "Terrible service"]
        >>> find_duplicates(texts, threshold=85)
        [1]  # Second item is duplicate of first
    """
    indices_to_remove = []
    
    for i in range(len(texts)):
        if i in indices_to_remove:
            continue
        
        for j in range(i + 1, len(texts)):
            if j in indices_to_remove:
                continue
            
            # Calculate similarity ratio
            similarity = fuzz.ratio(texts[i].lower(), texts[j].lower())
            
            if similarity >= threshold:
                indices_to_remove.append(j)
    
    return sorted(set(indices_to_remove))


def preprocess_reviews(df: pd.DataFrame, remove_duplicates: bool = True, 
                       filter_english: bool = True, 
                       dedup_threshold: int = 85) -> pd.DataFrame:
    """
    Preprocess review dataframe: clean text, filter English, deduplicate.
    
    Args:
        df: Input dataframe with 'review_text' column
        remove_duplicates: Whether to remove duplicate reviews
        filter_english: Whether to filter non-English reviews
        dedup_threshold: Similarity threshold for duplicate detection (0-100)
        
    Returns:
        Preprocessed dataframe with 'review_text' and 'cleaned_text' columns
        
    Example:
        >>> df = pd.DataFrame({'review_text': ['<p>Great food!</p>', 'Bad service']})
        >>> result = preprocess_reviews(df)
        >>> result.columns
        Index(['review_text', 'cleaned_text'], dtype='object')
    """
    if df.empty:
        return df.copy()
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Clean text
    result_df['cleaned_text'] = result_df['review_text'].apply(clean_text)
    
    # Filter out empty cleaned texts
    result_df = result_df[result_df['cleaned_text'].str.len() > 0]
    
    if result_df.empty:
        return result_df
    
    # Filter English reviews if requested
    if filter_english:
        mask = result_df['cleaned_text'].apply(is_english)
        non_english_count = (~mask).sum()
        if non_english_count > 0:
            st.info(f"Filtered out {non_english_count} non-English review(s)")
        result_df = result_df[mask]
    
    if result_df.empty:
        return result_df
    
    # Remove duplicates if requested
    if remove_duplicates:
        texts = result_df['cleaned_text'].tolist()
        duplicate_indices = find_duplicates(texts, threshold=dedup_threshold)
        
        if duplicate_indices:
            duplicate_count = len(duplicate_indices)
            st.info(f"Removed {duplicate_count} duplicate review(s)")
            result_df = result_df.drop(result_df.index[duplicate_indices]).reset_index(drop=True)
    
    return result_df.reset_index(drop=True)

