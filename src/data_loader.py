"""CSV ingestion and validation module."""

import pandas as pd
from typing import Optional, Tuple
import streamlit as st


def load_csv(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and validate CSV file with review data.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        Tuple of (dataframe, error_message). If successful, error_message is None.
        If failed, dataframe is None and error_message contains the error.
        
    Example:
        >>> df, error = load_csv(uploaded_file)
        >>> if error:
        ...     st.error(error)
        >>> else:
        ...     st.dataframe(df)
    """
    if file is None:
        return None, "No file uploaded"
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                file.seek(0)  # Reset file pointer
                df = pd.read_csv(file, encoding=encoding)
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            return None, "Could not decode CSV file. Please ensure it's a valid CSV file."
        
        # Validate required columns
        required_columns = ['review_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check if dataframe is empty
        if df.empty:
            return None, "CSV file is empty"
        
        # Check if review_text column has any non-null values
        if df['review_text'].isna().all():
            return None, "review_text column contains no valid data"
        
        # Drop rows where review_text is null
        df = df.dropna(subset=['review_text'])
        
        if df.empty:
            return None, "No valid reviews found after removing null values"
        
        # Ensure review_text is string type
        df['review_text'] = df['review_text'].astype(str)
        
        # Validate optional columns exist and have correct types
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df, None
        
    except pd.errors.EmptyDataError:
        return None, "CSV file is empty"
    except pd.errors.ParserError as e:
        return None, f"Error parsing CSV file: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error loading CSV: {str(e)}"

