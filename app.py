"""Streamlit main application for Review Analyzer."""

import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.preprocessor import preprocess_reviews
from src.sentiment import add_sentiment_to_df
from src.aspect_extractor import add_aspects_to_df
from src.visualizations import (
    create_sentiment_donut_chart,
    create_sentiment_timeline,
    aggregate_aspects,
    create_aspect_bar_chart,
    get_top_quotes_for_aspect
)

# Page configuration
st.set_page_config(
    page_title="Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'aspect_df' not in st.session_state:
    st.session_state.aspect_df = None


def main():
    """Main application function."""
    st.title("📊 AI-Powered Review Analyzer for Local Businesses")
    st.markdown("Upload customer reviews to analyze sentiment and extract key aspects.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should contain 'review_text' column. Optional: 'rating', 'date', 'source'"
        )
        
        # API key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key for aspect extraction",
            value=st.session_state.get('api_key', '')
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        # Processing options
        st.subheader("Processing Options")
        remove_duplicates = st.checkbox("Remove duplicates", value=True)
        filter_english = st.checkbox("Filter English only", value=True)
        dedup_threshold = st.slider("Duplicate similarity threshold", 70, 100, 85)
        
        # Process button
        process_button = st.button("🔄 Process Reviews", type="primary", use_container_width=True)
    
    # Main content area
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file to get started.")
        st.markdown("""
        ### Expected CSV Format
        
        Your CSV file should contain at least a `review_text` column. Optional columns:
        - `rating`: Numeric rating (1-5)
        - `date`: Date of review
        - `source`: Source of review (e.g., "google", "yelp")
        
        ### Sample Data
        
        You can use the sample data in `data/sample_reviews.csv` to test the application.
        """)
        return
    
    # Load and validate CSV
    df, error = load_csv(uploaded_file)
    
    if error:
        st.error(f"❌ Error loading CSV: {error}")
        return
    
    st.success(f"✅ Loaded {len(df)} reviews from CSV")
    
    # Show raw data preview
    with st.expander("📋 Preview Raw Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Process reviews
    if process_button or st.session_state.processed_df is None:
        with st.spinner("Processing reviews..."):
            # Preprocessing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/3: Preprocessing reviews...")
            progress_bar.progress(0.1)
            processed_df = preprocess_reviews(
                df,
                remove_duplicates=remove_duplicates,
                filter_english=filter_english,
                dedup_threshold=dedup_threshold
            )
            
            if processed_df.empty:
                st.error("❌ No reviews remaining after preprocessing. Please check your data.")
                return
            
            status_text.text(f"Step 2/3: Analyzing sentiment for {len(processed_df)} reviews...")
            progress_bar.progress(0.5)
            
            # Sentiment analysis
            processed_df = add_sentiment_to_df(processed_df)
            
            status_text.text("Step 3/3: Extracting aspects (this may take a while)...")
            progress_bar.progress(0.7)
            
            # Aspect extraction (only if API key is provided)
            if api_key:
                try:
                    aspect_progress = st.progress(0)
                    processed_df = add_aspects_to_df(
                        processed_df,
                        api_key,
                        progress_bar=aspect_progress
                    )
                    aspect_progress.empty()
                except Exception as e:
                    st.warning(f"⚠️ Aspect extraction failed: {str(e)}. Showing sentiment-only results.")
                    processed_df['aspects'] = [[] for _ in range(len(processed_df))]
            else:
                st.warning("⚠️ No API key provided. Skipping aspect extraction.")
                processed_df['aspects'] = [[] for _ in range(len(processed_df))]
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            # Store in session state
            st.session_state.processed_df = processed_df
            
            # Aggregate aspects
            if api_key:
                st.session_state.aspect_df = aggregate_aspects(processed_df)
            
            st.success(f"✅ Processed {len(processed_df)} reviews successfully!")
    
    # Get processed data
    processed_df = st.session_state.processed_df
    aspect_df = st.session_state.aspect_df if api_key else None
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Aspect Analysis", "📋 Raw Data"])
    
    with tab1:
        st.header("Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reviews", len(processed_df))
        
        with col2:
            if 'sentiment' in processed_df.columns:
                positive_count = (processed_df['sentiment'] == 'positive').sum()
                st.metric("Positive Reviews", positive_count)
        
        with col3:
            if 'sentiment' in processed_df.columns:
                negative_count = (processed_df['sentiment'] == 'negative').sum()
                st.metric("Negative Reviews", negative_count)
        
        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        sentiment_fig = create_sentiment_donut_chart(processed_df)
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Sentiment timeline
        if 'date' in processed_df.columns:
            st.subheader("Sentiment Over Time")
            timeline_fig = create_sentiment_timeline(processed_df)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.info("No valid date data available for timeline visualization.")
    
    with tab2:
        st.header("Aspect Analysis")
        
        if not api_key:
            st.warning("⚠️ Please provide a Gemini API key in the sidebar to enable aspect extraction.")
        elif aspect_df is None or aspect_df.empty:
            st.info("No aspects extracted yet. Please process reviews with an API key.")
        else:
            # Aspect bar chart
            st.subheader("Aspect Frequency")
            aspect_fig = create_aspect_bar_chart(aspect_df)
            st.plotly_chart(aspect_fig, use_container_width=True)
            
            # Detailed aspect breakdown
            st.subheader("Aspect Details")
            
            for _, row in aspect_df.iterrows():
                aspect_name = row['name']
                count = row['count']
                positive_ratio = row['positive_ratio']
                
                with st.expander(f"**{aspect_name.title()}** - {count} mentions ({positive_ratio:.1%} positive)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Mentions", count)
                    
                    with col2:
                        st.metric("Positive", row['positive_count'])
                    
                    with col3:
                        st.metric("Negative", row['negative_count'])
                    
                    # Top quotes
                    st.markdown("#### Top Quotes")
                    quotes = get_top_quotes_for_aspect(processed_df, aspect_name, top_n=3)
                    
                    if quotes:
                        for i, quote_data in enumerate(quotes, 1):
                            sentiment_emoji = {
                                'positive': '✅',
                                'negative': '❌',
                                'neutral': '➖'
                            }.get(quote_data['sentiment'], '➖')
                            
                            st.markdown(
                                f"{i}. {sentiment_emoji} *\"{quote_data['quote']}\"* "
                                f"({quote_data['sentiment'].capitalize()})"
                            )
                    else:
                        st.info("No quotes available for this aspect.")
    
    with tab3:
        st.header("Raw Data")
        
        # Search functionality
        search_term = st.text_input("🔍 Search reviews", "")
        
        # Filter dataframe
        display_df = processed_df.copy()
        
        if search_term:
            mask = display_df['cleaned_text'].str.contains(search_term, case=False, na=False)
            display_df = display_df[mask]
            st.info(f"Found {len(display_df)} reviews matching '{search_term}'")
        
        # Prepare display dataframe
        display_columns = ['review_text', 'cleaned_text', 'sentiment', 'sentiment_confidence']
        
        if 'rating' in display_df.columns:
            display_columns.insert(2, 'rating')
        if 'date' in display_df.columns:
            display_columns.insert(3, 'date')
        if 'source' in display_df.columns:
            display_columns.insert(4, 'source')
        
        # Format aspects for display
        if 'aspects' in display_df.columns:
            display_df['aspects_display'] = display_df['aspects'].apply(
                lambda x: ', '.join([a.get('name', '') for a in x if isinstance(a, dict)]) if isinstance(x, list) else ''
            )
            display_columns.append('aspects_display')
        
        # Select only available columns
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data as CSV",
            data=csv,
            file_name="processed_reviews.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

