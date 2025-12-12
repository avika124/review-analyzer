"""Aspect extraction using Google Gemini API."""

import json
import time
import google.generativeai as genai
import pandas as pd
from typing import List, Dict, Optional, Any
import streamlit as st


def get_gemini_client(api_key: str):
    """
    Initialize and return Gemini client.
    
    Args:
        api_key: Google Gemini API key
        
    Returns:
        Configured Gemini model
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')


def extract_aspects_from_review(review_text: str, model, max_retries: int = 3) -> Dict[str, Any]:
    """
    Extract aspects from a single review using Gemini API.
    
    Args:
        review_text: Review text to analyze
        model: Gemini model instance
        max_retries: Maximum number of retry attempts with exponential backoff
        
    Returns:
        Dictionary with 'aspects' key containing list of aspect dictionaries
        
    Example:
        >>> model = get_gemini_client(api_key)
        >>> result = extract_aspects_from_review("Great food but slow service", model)
        >>> result['aspects'][0]['name']
        'food quality'
    """
    prompt = f"""Analyze this review and extract mentioned aspects with their sentiment.
Return JSON only, no markdown:
{{"aspects": [{{"name": "food quality", "sentiment": "positive", "quote": "exact phrase"}}]}}

Possible aspects: food quality, service, ambiance, price, cleanliness, location, wait time, portion size

Review: {review_text}"""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                # Extract JSON from code block
                lines = response_text.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_json = not in_json
                        continue
                    if in_json or (not response_text.startswith('```json') and not response_text.startswith('```')):
                        json_lines.append(line)
                response_text = '\n'.join(json_lines)
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate structure
            if 'aspects' not in result:
                result = {'aspects': []}
            
            return result
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                st.warning(f"Failed to parse JSON response for review: {str(e)}")
                return {'aspects': []}
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                st.warning(f"Error extracting aspects: {str(e)}")
                return {'aspects': []}
    
    return {'aspects': []}


def extract_aspects_batch(reviews: List[str], api_key: str, 
                         progress_bar=None) -> List[Dict[str, Any]]:
    """
    Extract aspects from multiple reviews with progress updates.
    Note: Results are cached via session state in app.py to avoid redundant API calls.
    
    Args:
        reviews: List of review texts
        api_key: Google Gemini API key
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        List of aspect dictionaries for each review
    """
    if not reviews or not api_key:
        return [{'aspects': []} for _ in reviews]
    
    model = get_gemini_client(api_key)
    results = []
    
    total = len(reviews)
    for i, review in enumerate(reviews):
        if progress_bar:
            progress_bar.progress((i + 1) / total)
        
        aspects = extract_aspects_from_review(review, model)
        results.append(aspects)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return results


def add_aspects_to_df(df: pd.DataFrame, api_key: str, 
                      text_column: str = 'cleaned_text',
                      progress_bar=None) -> pd.DataFrame:
    """
    Add extracted aspects to dataframe.
    
    Args:
        df: Dataframe with text column
        api_key: Google Gemini API key
        text_column: Name of the column containing text to analyze
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        Dataframe with added 'aspects' column containing list of aspect dicts
        
    Example:
        >>> df = pd.DataFrame({'cleaned_text': ['Great food!']})
        >>> result = add_aspects_to_df(df, api_key)
        >>> 'aspects' in result.columns
        True
    """
    if df.empty or text_column not in df.columns:
        return df.copy()
    
    result_df = df.copy()
    reviews = result_df[text_column].tolist()
    
    # Extract aspects
    aspect_results = extract_aspects_batch(reviews, api_key, progress_bar)
    
    result_df['aspects'] = aspect_results
    
    return result_df

