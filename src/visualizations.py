"""Plotly chart generation for dashboard visualizations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict
from collections import Counter, defaultdict


def create_sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a donut chart showing sentiment distribution.
    
    Args:
        df: Dataframe with 'sentiment' column
        
    Returns:
        Plotly figure object
        
    Example:
        >>> df = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral']})
        >>> fig = create_sentiment_donut_chart(df)
        >>> fig.show()
    """
    if df.empty or 'sentiment' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    sentiment_counts = df['sentiment'].value_counts()
    
    # Map sentiment to colors
    color_map = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    colors = [color_map.get(sent, '#95a5a6') for sent in sentiment_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index.str.capitalize(),
        values=sentiment_counts.values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        height=400
    )
    
    return fig


def create_sentiment_timeline(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a line chart showing sentiment over time.
    
    Args:
        df: Dataframe with 'date' and 'sentiment' columns
        
    Returns:
        Plotly figure object or None if date column doesn't exist
        
    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=10),
        ...     'sentiment': ['positive'] * 5 + ['negative'] * 5
        ... })
        >>> fig = create_sentiment_timeline(df)
    """
    if df.empty or 'date' not in df.columns or df['date'].isna().all():
        return None
    
    # Filter out null dates
    df_with_dates = df.dropna(subset=['date']).copy()
    
    if df_with_dates.empty:
        return None
    
    # Group by date and sentiment
    df_with_dates['date'] = pd.to_datetime(df_with_dates['date'])
    df_with_dates['date'] = df_with_dates['date'].dt.date
    
    timeline = df_with_dates.groupby(['date', 'sentiment']).size().reset_index(name='count')
    
    # Pivot for easier plotting
    timeline_pivot = timeline.pivot(index='date', columns='sentiment', values='count').fillna(0)
    
    fig = go.Figure()
    
    color_map = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    for sentiment in timeline_pivot.columns:
        fig.add_trace(go.Scatter(
            x=timeline_pivot.index,
            y=timeline_pivot[sentiment],
            mode='lines+markers',
            name=sentiment.capitalize(),
            line=dict(color=color_map.get(sentiment, '#95a5a6'), width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Reviews",
        hovermode='x unified',
        height=400
    )
    
    return fig


def aggregate_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate aspects from dataframe into summary statistics.
    
    Args:
        df: Dataframe with 'aspects' column containing lists of aspect dicts
        
    Returns:
        Dataframe with aspect statistics: name, count, positive_count, negative_count, 
        neutral_count, positive_ratio
        
    Example:
        >>> df = pd.DataFrame({
        ...     'aspects': [
        ...         [{'name': 'food quality', 'sentiment': 'positive'}],
        ...         [{'name': 'food quality', 'sentiment': 'negative'}]
        ...     ]
        ... })
        >>> result = aggregate_aspects(df)
        >>> result[result['name'] == 'food quality']['count'].iloc[0]
        2
    """
    if df.empty or 'aspects' not in df.columns:
        return pd.DataFrame()
    
    aspect_stats = defaultdict(lambda: {'count': 0, 'positive': 0, 'negative': 0, 'neutral': 0, 'quotes': []})
    
    for aspects_list in df['aspects']:
        if not isinstance(aspects_list, list):
            continue
        
        for aspect in aspects_list:
            if not isinstance(aspect, dict):
                continue
            
            name = aspect.get('name', '').lower().strip()
            if not name:
                continue
            
            sentiment = aspect.get('sentiment', '').lower()
            quote = aspect.get('quote', '')
            
            aspect_stats[name]['count'] += 1
            aspect_stats[name][sentiment] = aspect_stats[name].get(sentiment, 0) + 1
            
            if quote:
                aspect_stats[name]['quotes'].append({
                    'quote': quote,
                    'sentiment': sentiment
                })
    
    # Convert to dataframe
    rows = []
    for name, stats in aspect_stats.items():
        total = stats['count']
        positive = stats.get('positive', 0)
        negative = stats.get('negative', 0)
        neutral = stats.get('neutral', 0)
        
        positive_ratio = positive / total if total > 0 else 0
        
        rows.append({
            'name': name,
            'count': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_ratio': positive_ratio
        })
    
    result_df = pd.DataFrame(rows)
    
    if not result_df.empty:
        result_df = result_df.sort_values('count', ascending=False)
    
    return result_df


def create_aspect_bar_chart(aspect_df: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart showing aspect frequency colored by sentiment ratio.
    
    Args:
        aspect_df: Dataframe from aggregate_aspects() function
        
    Returns:
        Plotly figure object
        
    Example:
        >>> aspect_df = aggregate_aspects(df)
        >>> fig = create_aspect_bar_chart(aspect_df)
        >>> fig.show()
    """
    if aspect_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No aspects found", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Create color scale based on positive ratio
    colors = aspect_df['positive_ratio'].apply(
        lambda x: '#2ecc71' if x >= 0.6 else '#e74c3c' if x <= 0.4 else '#f39c12'
    )
    
    fig = go.Figure(data=[go.Bar(
        x=aspect_df['count'],
        y=aspect_df['name'],
        orientation='h',
        marker=dict(color=colors),
        text=aspect_df['count'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Positive Ratio: %{customdata:.1%}<extra></extra>',
        customdata=aspect_df['positive_ratio']
    )])
    
    fig.update_layout(
        title="Aspect Frequency (colored by sentiment ratio)",
        xaxis_title="Number of Mentions",
        yaxis_title="Aspect",
        height=max(400, len(aspect_df) * 40),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def get_top_quotes_for_aspect(df: pd.DataFrame, aspect_name: str, 
                              top_n: int = 3) -> List[Dict[str, str]]:
    """
    Get top representative quotes for a specific aspect.
    
    Args:
        df: Dataframe with 'aspects' and 'sentiment_confidence' columns
        aspect_name: Name of the aspect to extract quotes for
        top_n: Number of quotes to return
        
    Returns:
        List of quote dictionaries with 'quote', 'sentiment', and 'confidence' keys
        
    Example:
        >>> quotes = get_top_quotes_for_aspect(df, 'food quality', top_n=3)
        >>> len(quotes) <= 3
        True
    """
    quotes = []
    
    for idx, row in df.iterrows():
        aspects_list = row.get('aspects', [])
        if not isinstance(aspects_list, list):
            continue
        
        confidence = row.get('sentiment_confidence', 0.5)
        
        for aspect in aspects_list:
            if not isinstance(aspect, dict):
                continue
            
            name = aspect.get('name', '').lower().strip()
            if name == aspect_name.lower().strip():
                quote = aspect.get('quote', '')
                sentiment = aspect.get('sentiment', 'neutral')
                
                if quote:
                    quotes.append({
                        'quote': quote,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
    
    # Sort by confidence (prioritize high-confidence sentiments)
    quotes.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Prioritize positive and negative over neutral
    positive_negative = [q for q in quotes if q['sentiment'] in ['positive', 'negative']]
    neutral = [q for q in quotes if q['sentiment'] == 'neutral']
    
    # Return top N, prioritizing positive/negative
    result = positive_negative[:top_n]
    if len(result) < top_n:
        result.extend(neutral[:top_n - len(result)])
    
    return result[:top_n]

