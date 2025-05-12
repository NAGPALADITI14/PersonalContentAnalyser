import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# from datapreprocessing import DataPreprocessor
from ChromePreprocessing import ChromePreprocessor
from YoutubePreprocessing import YouTubePreprocessor
from yt_analysis import YtAnalyser
from chrome_analysis import ChromeAnalyser
from IntellectualNutrition import IntellectualNutrition
# Set page configuration
st.set_page_config(
    page_title="Personal Content Consumption Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data preprocessor
yt_preprocessor = YouTubePreprocessor()
chrome_preprocessor = ChromePreprocessor()

yt_analysis = YtAnalyser()
ch_analysis = ChromeAnalyser()
nutrition = IntellectualNutrition()


# Improved caching for better performance
@st.cache_data(ttl=3600)
def load_data_file(file_obj, file_type):
    """Load and preprocess a single data file."""
    if file_obj is None:
        return None
    
    temp_filename = f"temp_{file_type}"
    with open(temp_filename, "wb") as f:
        f.write(file_obj.getbuffer())
    
    if file_type == "youtube_watch":
        data = yt_preprocessor.preprocess_youtube_watch_history(temp_filename)
    elif file_type == "youtube_search":
        data = yt_preprocessor.preprocess_youtube_search_history(temp_filename)
    elif file_type == "chrome_history":
        data = chrome_preprocessor.preprocess_chrome_history(temp_filename)
    elif file_type == "chrome_bookmarks":
        data = chrome_preprocessor.preprocess_chrome_bookmarks(temp_filename)
    
    # Clean up temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
         
    return data



def main():
    """Main function to run the Streamlit app."""
    # Set up sidebar
    st.sidebar.title("Personal Content Consumption Analyzer")
    st.sidebar.image("https://img.icons8.com/clouds/100/000000/data-analytics.png")
    
    # Create navigation first, before file uploaders
    st.sidebar.header("Analysis Type")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type", 
        ["Overview", "YouTube Analysis", "Chrome Analysis", "Intellectual Nutrition"]
    )
    
    # Initialize dictionaries to store data
    data = {}
    
    # Show specific file uploaders based on selected analysis
    st.sidebar.header("Upload Your Data")
    
    if analysis_type == "Overview" or analysis_type == "YouTube Analysis" or analysis_type == "Intellectual Nutrition":
        st.sidebar.markdown("**YouTube Data Files:**")
        youtube_watch_file = st.sidebar.file_uploader("Upload YouTube Watch History (HTML)", type=["html"], key="yt_watch")
        youtube_search_file = st.sidebar.file_uploader("Upload YouTube Search History (HTML)", type=["html"], key="yt_search")
        
        # Load YouTube data only if files are uploaded
        with st.spinner("Loading YouTube data..."):
            if youtube_watch_file:
                data['youtube_watch'] = load_data_file(youtube_watch_file, "youtube_watch")
            if youtube_search_file:
                data['youtube_search'] = load_data_file(youtube_search_file, "youtube_search")
    
    if analysis_type == "Overview" or analysis_type == "Chrome Analysis" or analysis_type == "Intellectual Nutrition":
        st.sidebar.markdown("**Chrome Data Files:**")
        chrome_history_file = st.sidebar.file_uploader("Upload Chrome History (JSON)", type=["json"], key="chrome_hist")
        chrome_bookmarks_file = st.sidebar.file_uploader("Upload Chrome Bookmarks (HTML)", type=["html"], key="chrome_book")
        
        # Load Chrome data only if files are uploaded
        with st.spinner("Loading Chrome data..."):
            if chrome_history_file:
                data['chrome_history'] = load_data_file(chrome_history_file, "chrome_history")
            if chrome_bookmarks_file:
                data['chrome_bookmarks'] = load_data_file(chrome_bookmarks_file, "chrome_bookmarks")
    
    # Show analysis based on selection
    if analysis_type == "Overview":
        show_overview(data)
    elif analysis_type == "YouTube Analysis":
        yt_analysis.show_youtube_analysis(data)
    elif analysis_type == "Chrome Analysis":
        # show_chrome_analysis(data)
        ch_analysis.show_chrome_analysis(data)
    elif analysis_type == "Intellectual Nutrition":
        # st.info("Intellectual Nutrition feature coming soon!")
        nutrition.show_intellectual_nutrition(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Personal Content Consumption Analyzer** helps you understand your 
        digital footprint and content consumption habits.
        """
    )

def show_overview(data):
    """Display overview of all data."""
    st.title("📊 Personal Content Consumption Analyzer")
    
    if not data:
        show_getting_started()
        return
    
    # Show data overview stats with progress indicator
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'youtube_watch' in data:
            yt_watch_count = len(data['youtube_watch'])
            st.metric("YouTube Videos Watched", f"{yt_watch_count:,}")
        else:
            st.metric("YouTube Videos Watched", "No data")
    
    with col2:
        if 'youtube_search' in data:
            yt_search_count = len(data['youtube_search'])
            st.metric("YouTube Searches", f"{yt_search_count:,}")
        else:
            st.metric("YouTube Searches", "No data")
    
    with col3:
        if 'chrome_history' in data:
            chrome_visit_count = len(data['chrome_history'])
            st.metric("Chrome Visits", f"{chrome_visit_count:,}")
        else:
            st.metric("Chrome Visits", "No data")
    
    with col4:
        if 'chrome_bookmarks' in data:
            bookmark_count = len(data['chrome_bookmarks'])
            st.metric("Chrome Bookmarks", f"{bookmark_count:,}")
        else:
            st.metric("Chrome Bookmarks", "No data")
    
    # Timeline summary - more efficient approach
    st.subheader("Timeline Summary")
    
    # Get timeline data with efficient caching
    timeline_data = []
    
    for source, df in data.items():
        if df is not None and 'timestamp' in df.columns and not df.empty:
            source_data = df[['timestamp']].copy()
            source_data['source'] = source

            # Ensure timestamp is datetime
            source_data['timestamp'] = pd.to_datetime(source_data['timestamp'], errors='coerce')

            # Drop NaT values if any after coercion
            source_data = source_data.dropna(subset=['timestamp'])

            # Remove timezone info if present
            source_data['timestamp'] = source_data['timestamp'].dt.tz_localize(None)

            timeline_data.append(source_data)


    if timeline_data:
        with st.spinner("Generating timeline..."):
            timeline_df = pd.concat(timeline_data)
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(timeline_df['timestamp']):
                timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
            
            # Add date column
            timeline_df['date'] = timeline_df['timestamp'].dt.date
            
            # Count records by date and source - use efficient groupby
            timeline_counts = timeline_df.groupby(['date', 'source']).size().reset_index(name='count')
            
            # Only show last 30 days of data if available
            if len(timeline_counts['date'].unique()) > 30:
                max_date = timeline_counts['date'].max()
                min_date = max_date - timedelta(days=30)
                timeline_counts = timeline_counts[timeline_counts['date'] >= min_date]
            
            # Create timeline chart with better performance config
            fig = px.line(timeline_counts, x='date', y='count', color='source',
                        title='Digital Activity Timeline',
                        labels={'date': 'Date', 'count': 'Activity Count', 'source': 'Data Source'},
                        color_discrete_map={
                            'youtube_watch': '#FF0000',
                            'youtube_search': '#FF5733',
                            'chrome_history': '#4285F4',
                            'chrome_bookmarks': '#0F9D58'
                        })
            
            fig.update_layout(
                xaxis_title="Date", 
                yaxis_title="Number of Activities",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timeline data not available. Please upload files with timestamp information.")
    
    # Only show category distribution if we have categories
    has_categories = False
    for source, df in data.items():
        if df is not None and 'category' in df.columns and not df.empty:
            has_categories = True
            break
    
    if has_categories:
        # Overall Content Distribution with efficient processing
        st.subheader("Overall Content Category Distribution")
        with st.spinner("Processing content categories..."):
            # Combine all category data more efficiently
            category_data = []
            
            for source, df in data.items():
                if df is not None and 'category' in df.columns and not df.empty:
                    # Extract just the category column for efficiency
                    source_data = df[['category']].copy()
                    source_data['source'] = source
                    category_data.append(source_data)
            
            if category_data:
                category_df = pd.concat(category_data)
                
                # Get category distribution - more efficient approach
                category_counts = category_df['category'].value_counts().nlargest(10).reset_index()
                category_counts.columns = ['category', 'count']
                
                # Calculate percentages
                total = category_counts['count'].sum()
                category_counts['percentage'] = category_counts['count'] / total * 100
                
                # Create pie chart
                fig = px.pie(category_counts, values='count', names='category',
                            title='Top 10 Content Categories',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    # Quick Analysis Summary - more efficient
    if data:
        st.header("Quick Analysis Summary")
        
        st.markdown("#### Key Insights")
        insights = []
        
        # Add insights based on available data
        if 'youtube_watch' in data and not data['youtube_watch'].empty:
            # Most watched YouTube channels
            if 'channel' in data['youtube_watch'].columns:
                top_channels = data['youtube_watch']['channel'].value_counts().head(3)
                if not top_channels.empty:
                    channels_text = ", ".join([f"{channel}" for channel in top_channels.index.tolist()])
                    insights.append(f"📺 Your top YouTube channels: {channels_text}")
        
        if 'chrome_history' in data and not data['chrome_history'].empty:
            # Most visited domains
            if 'domain' in data['chrome_history'].columns:
                top_domains = data['chrome_history']['domain'].value_counts().head(3)
                if not top_domains.empty:
                    domains_text = ", ".join([f"{domain}" for domain in top_domains.index.tolist()])
                    insights.append(f"🌐 Your most visited websites: {domains_text}")
            
            # Top categories - only if we have categories
            if 'category' in data['chrome_history'].columns:
                top_categories = data['chrome_history']['category'].value_counts().head(3)
                if not top_categories.empty:
                    categories_text = ", ".join([f"{cat}" for cat in top_categories.index.tolist()])
                    insights.append(f"📊 Your top content categories: {categories_text}")
        
        # Time of day insights - efficient approach
        time_of_day_data = []
        for source, df in data.items():
            if df is not None and 'time_of_day' in df.columns and not df.empty:
                source_data = df[['time_of_day']].copy()
                time_of_day_data.append(source_data)
        
        if time_of_day_data:
            time_of_day_df = pd.concat(time_of_day_data)
            peak_time = time_of_day_df['time_of_day'].value_counts().index[0]
            insights.append(f"⏰ Your peak activity time: {peak_time}")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.info("Upload more data to see insights.")
        
        # Highlight areas for further exploration
        st.markdown("#### Explore Further")
        st.markdown("""
        - Check the **YouTube Analysis** option for detailed video and search patterns
        - Visit the **Chrome Analysis** option for web browsing habits and bookmarks
        - Review your **Intellectual Nutrition** for content consumption balance
        - Get personalized **Recommendations** to diversify your content consumption
        """)

def show_getting_started():
    """Show getting started instructions when no data is uploaded."""
    st.warning("Please upload your data files using the sidebar to begin the analysis.")
    st.markdown("""
    ### Getting Started
    
    1. Export your Google Takeout data
       - Go to [Google Takeout](https://takeout.google.com/)
       - Select YouTube History and Chrome data
       - Download your archive
    
    2. Extract the archive and locate the following files:
       - YouTube Watch History (HTML)
       - YouTube Search History (HTML)
       - Chrome History (JSON)
       - Chrome Bookmarks (HTML)
    
    3. Select an analysis type and upload the relevant files using the sidebar
    
    4. Explore your content consumption patterns and receive personalized recommendations
    """)
    
    st.markdown("---")
    st.subheader("Features of this Analyzer:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📺 **YouTube Analysis**
            - Watch history patterns
            - Search trends
            - Channel preferences
            - Topic diversity
        
        - 🌐 **Chrome Analysis**
            - Browsing patterns
            - Website categories
            - Time distribution
            - Bookmark analysis
        """)
    
    with col2:
        st.markdown("""
        - 🧠 **Intellectual Nutrition**
            - Content consumption balance
            - Bias identification
            - Topic diversity score
            - Usage pattern visualization
        
        - 💡 **Smart Recommendations**
            - Quality content suggestions
            - Underrepresented viewpoints
            - Balanced content diet
            - Time optimization tips
        """)

if __name__ == "__main__":
    main()