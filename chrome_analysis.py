import streamlit as st
import pandas as pd
import plotly.express as px
from ChromePreprocessing import ChromePreprocessor
import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter

class ChromeAnalyser:
    def __init__(self):
        """Initialize the Chrome Analyzer with a preprocessor."""
        self.processor = ChromePreprocessor()
    
    @st.cache_data
    def generate_wordcloud(_self,_data, column):
        """Generate a wordcloud from text data."""
        if _data is None or column not in _data.columns or _data.empty:
            return None
        
        # Combine all text data
        text = ' '.join([str(item) for item in _data[column].dropna()])
        
        if not text or len(text) < 10:  # Check if there's enough text
            return None
        
        # Generate wordcloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            contour_width=3,
            colormap='viridis'
        ).generate(text)
        
        # Convert to image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Convert plot to base64 encoded image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    @st.cache_data
    def compute_time_patterns(_self, df, activity_type):
        """Compute time patterns for a dataframe with timestamp column."""
        if df is None or 'timestamp' not in df.columns or df.empty:
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a copy to avoid modifying the original
        time_df = df[['timestamp']].copy()
        time_df['type'] = activity_type
        
        # Extract time components
        time_df['hour'] = time_df['timestamp'].dt.hour
        time_df['day_of_week'] = time_df['timestamp'].dt.day_name()
        time_df['month'] = time_df['timestamp'].dt.month_name()
        time_df['date'] = time_df['timestamp'].dt.date
        
        # Create time bins for time of day analysis
        time_df['time_of_day'] = pd.cut(
            time_df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)']
        )
        
        return time_df

    def show_chrome_analysis(self, data):
        """Display Chrome-specific analysis."""
        st.title("🌐 Chrome Browsing Analysis")
        
        # Check if Chrome data is available
        if not data:
            st.warning("Please upload Chrome history and/or bookmarks files to see this analysis.")
            return
        
        if 'chrome_history' not in data and 'chrome_bookmarks' not in data:
            st.warning("No Chrome data found. Please upload Chrome history and/or bookmarks files.")
            return
        
        tabs = st.tabs(["Browsing History", "Bookmarks", "Domain Analysis", "Time Patterns"])
        
        # Browsing History Tab
        with tabs[0]:
            if 'chrome_history' in data and not data['chrome_history'].empty:
                st.header("Browsing History Analysis")
                history_df = data['chrome_history']
                
                # Summary statistics
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_visits = len(history_df)
                    st.metric("Total Page Visits", f"{total_visits:,}")
                
                with col2:
                    if 'domain' in history_df.columns:
                        unique_domains = history_df['domain'].nunique()
                        st.metric("Unique Domains", f"{unique_domains:,}")
                    else:
                        st.metric("Unique Domains", "N/A")
                
                with col3:
                    if 'timestamp' in history_df.columns and not history_df['timestamp'].isnull().all():
                        try:
                            date_range = (history_df['timestamp'].max() - history_df['timestamp'].min()).days
                            st.metric("Date Range", f"{date_range} days")
                        except:
                            st.metric("Date Range", "N/A")
                    else:
                        st.metric("Date Range", "N/A")
                
                # Domain analysis
                if 'domain' in history_df.columns:
                    st.subheader("Domain Analysis")
                    with st.spinner("Analyzing domain data..."):
                        top_domains = history_df['domain'].value_counts().head(15).reset_index()
                        top_domains.columns = ['domain', 'count']
                        
                        fig = px.bar(
                            top_domains,
                            x='domain',
                            y='count',
                            color='count',
                            color_continuous_scale='blues',
                            labels={'domain': 'Domain', 'count': 'Visit Count'},
                            title='Top 15 Most Visited Domains'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Title analysis with wordcloud
                st.subheader("Content Analysis")
                with st.spinner("Generating content wordcloud..."):
                    if 'title' in history_df.columns:
                        # Sample data for wordcloud to improve performance
                        if len(history_df) > 1000:
                            sample_df = history_df.sample(1000)
                        else:
                            sample_df = history_df
                        
                        wordcloud = self.generate_wordcloud(sample_df, 'title')
                        
                        if wordcloud:
                            st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{wordcloud}' style='max-width: 100%;'></div>", unsafe_allow_html=True)
                            st.caption("Word cloud generated from page titles")
                        else:
                            st.info("Not enough data to generate a word cloud.")
                    else:
                        st.info("Title data not available for wordcloud generation.")
                
                # Most visited pages
                st.subheader("Most Visited Pages")
                if len(history_df) > 0 and 'title' in history_df.columns and 'url' in history_df.columns:
                    with st.spinner("Finding most visited pages..."):
                        page_counts = history_df.groupby(['title', 'url']).size().reset_index(name='count')
                        page_counts = page_counts.sort_values('count', ascending=False).head(10)
                        
                        for i, (_, row) in enumerate(page_counts.iterrows()):
                            st.markdown(f"**{i+1}.** [{row['title']}]({row['url']}) - Visited {row['count']} times")
                else:
                    st.info("No history data available.")
                
                # Search queries if available
                st.subheader("Search Queries")
                if 'search_query' in history_df.columns:
                    search_queries = history_df.dropna(subset=['search_query'])
                    if not search_queries.empty:
                        query_counts = search_queries['search_query'].value_counts().head(15)
                        
                        fig = px.bar(
                            x=query_counts.index,
                            y=query_counts.values,
                            labels={'x': 'Search Query', 'y': 'Count'},
                            title='Most Common Search Queries',
                            color=query_counts.values,
                            color_continuous_scale='greens'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Search query wordcloud
                        wordcloud = self.generate_wordcloud(search_queries, 'search_query')
                        if wordcloud:
                            st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{wordcloud}' style='max-width: 100%;'></div>", unsafe_allow_html=True)
                            st.caption("Word cloud generated from search queries")
                    else:
                        st.info("No search queries found in the history data.")
                else:
                    st.info("Search query data not available.")
            else:
                st.info("No Chrome browsing history found. Please upload history file.")
        
        # Bookmarks Tab
        with tabs[1]:
            if 'chrome_bookmarks' in data and not data['chrome_bookmarks'].empty:
                st.header("Bookmarks Analysis")
                bookmarks_df = data['chrome_bookmarks']
                
                # Summary statistics
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    total_bookmarks = len(bookmarks_df)
                    st.metric("Total Bookmarks", f"{total_bookmarks:,}")
                
                with col2:
                    if 'domain' in bookmarks_df.columns:
                        unique_domains = bookmarks_df['domain'].nunique()
                        st.metric("Unique Domains", f"{unique_domains:,}")
                    else:
                        st.metric("Unique Domains", "N/A")
                
                # Folder structure analysis
                if 'folder' in bookmarks_df.columns:
                    st.subheader("Bookmark Folders")
                    folder_counts = bookmarks_df['folder'].value_counts().head(10)
                    
                    fig = px.bar(
                        x=folder_counts.index,
                        y=folder_counts.values,
                        labels={'x': 'Folder', 'y': 'Count'},
                        title='Top 10 Bookmark Folders',
                        color=folder_counts.values,
                        color_continuous_scale='purples'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Domain analysis
                if 'domain' in bookmarks_df.columns:
                    st.subheader("Domain Analysis")
                    domain_counts = bookmarks_df['domain'].value_counts().head(15)
                    
                    fig = px.bar(
                        x=domain_counts.index,
                        y=domain_counts.values,
                        labels={'x': 'Domain', 'y': 'Count'},
                        title='Most Bookmarked Domains',
                        color=domain_counts.values,
                        color_continuous_scale='oranges'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Title wordcloud
                st.subheader("Bookmark Content")
                with st.spinner("Generating bookmark wordcloud..."):
                    if 'title' in bookmarks_df.columns:
                        wordcloud = self.generate_wordcloud(bookmarks_df, 'title')
                        
                        if wordcloud:
                            st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{wordcloud}' style='max-width: 100%;'></div>", unsafe_allow_html=True)
                            st.caption("Word cloud generated from bookmark titles")
                        else:
                            st.info("Not enough data to generate a word cloud.")
                    else:
                        st.info("Title data not available for wordcloud generation.")
                
                # Show bookmark folders and contents
                st.subheader("Bookmark Organization")
                if 'folder' in bookmarks_df.columns and 'title' in bookmarks_df.columns and 'url' in bookmarks_df.columns:
                    folders = bookmarks_df['folder'].unique()
                    
                    for folder in sorted(folders):
                        folder_bookmarks = bookmarks_df[bookmarks_df['folder'] == folder]
                        with st.expander(f"{folder} ({len(folder_bookmarks)} bookmarks)"):
                            for _, row in folder_bookmarks.iterrows():
                                st.markdown(f"[{row['title']}]({row['url']})")
                else:
                    st.info("Folder structure data not available.")
            else:
                st.info("No Chrome bookmarks found. Please upload bookmarks file.")
        
        # Domain Analysis Tab
        with tabs[2]:
            st.header("Domain Analysis")
            
            combined_domains = []
            
            # Collect domain data from both history and bookmarks
            if 'chrome_history' in data and not data['chrome_history'].empty and 'domain' in data['chrome_history'].columns:
                history_domains = data['chrome_history'][['domain', 'url']].copy()
                history_domains['source'] = 'History'
                combined_domains.append(history_domains)
            
            if 'chrome_bookmarks' in data and not data['chrome_bookmarks'].empty and 'domain' in data['chrome_bookmarks'].columns:
                bookmark_domains = data['chrome_bookmarks'][['domain', 'url']].copy()
                bookmark_domains['source'] = 'Bookmarks'
                combined_domains.append(bookmark_domains)
            
            if combined_domains:
                all_domains_df = pd.concat(combined_domains)
                
                # Top domains overall
                st.subheader("Top Domains Overall")
                domain_counts = all_domains_df['domain'].value_counts().head(15)
                
                fig = px.bar(
                    x=domain_counts.index,
                    y=domain_counts.values,
                    labels={'x': 'Domain', 'y': 'Count'},
                    title='Most Frequent Domains in Browsing History and Bookmarks',
                    color=domain_counts.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare domains between history and bookmarks
                st.subheader("Domain Comparison: History vs. Bookmarks")
                
                domain_source_counts = all_domains_df.groupby(['domain', 'source']).size().unstack(fill_value=0)
                
                if domain_source_counts.shape[1] == 2:  # If we have both history and bookmarks
                    # Get top domains by total count
                    top_domains = domain_source_counts.sum(axis=1).sort_values(ascending=False).head(10).index
                    comparison_df = domain_source_counts.loc[top_domains].reset_index()
                    
                    # Melt for plotting
                    comparison_df_melted = pd.melt(
                        comparison_df, 
                        id_vars='domain', 
                        var_name='source', 
                        value_name='count'
                    )
                    
                    fig = px.bar(
                        comparison_df_melted,
                        x='domain',
                        y='count',
                        color='source',
                        barmode='group',
                        labels={'domain': 'Domain', 'count': 'Count', 'source': 'Source'},
                        title='Top 10 Domains: History vs. Bookmarks'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Domain categorization if available
                if 'category' in all_domains_df.columns:
                    st.subheader("Domain Categories")
                    category_counts = all_domains_df['category'].value_counts()
                    
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title='Distribution of Domain Categories',
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Domain network visualization placeholder
                st.subheader("Domain Network")
                st.info("This feature would show a network visualization of domains and their relationships.")
                
                # TLD Analysis
                st.subheader("Top-Level Domain (TLD) Analysis")
                
                # Extract TLDs from domains
                all_domains_df['tld'] = all_domains_df['domain'].apply(
                    lambda x: x.split('.')[-1] if isinstance(x, str) and '.' in x else 'unknown'
                )
                
                tld_counts = all_domains_df['tld'].value_counts().head(10)
                
                fig = px.pie(
                    values=tld_counts.values,
                    names=tld_counts.index,
                    title='Top-Level Domain Distribution',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No domain data available. Please upload Chrome history or bookmarks files.")
        
        # Time Patterns Tab
        with tabs[3]:
            st.header("Time Patterns")
            
            # Process time patterns from history data
            history_time_df = None
            if 'chrome_history' in data and not data['chrome_history'].empty and 'timestamp' in data['chrome_history'].columns:
                history_time_df = self.compute_time_patterns(data['chrome_history'], 'Browsing')
            
            if history_time_df is not None:
                # Hourly distribution
                st.subheader("Hourly Browsing Distribution")
                hourly_counts = history_time_df.groupby('hour').size().reset_index(name='count')
                
                fig = px.line(
                    hourly_counts,
                    x='hour',
                    y='count',
                    labels={'hour': 'Hour of Day', 'count': 'Visit Count'},
                    title='Hourly Distribution of Browser Activity',
                    markers=True
                )
                fig.update_layout(
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                    xaxis_title="Hour of Day (24-hour format)",
                    yaxis_title="Number of Page Visits"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Time of day distribution
                st.subheader("Time of Day Distribution")
                time_of_day_counts = history_time_df.groupby('time_of_day').size().reset_index(name='count')
                
                fig = px.bar(
                    time_of_day_counts, 
                    x='time_of_day', 
                    y='count',
                    color='count',
                    color_continuous_scale='blues',
                    title='Browsing Activity by Time of Day',
                    labels={'time_of_day': 'Time of Day', 'count': 'Visit Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Day of week distribution
                st.subheader("Day of Week Distribution")
                
                # Set proper day of week order
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                history_time_df['day_of_week'] = pd.Categorical(history_time_df['day_of_week'], categories=day_order, ordered=True)
                
                dow_counts = history_time_df.groupby('day_of_week').size().reset_index(name='count')
                
                fig = px.bar(
                    dow_counts,
                    x='day_of_week',
                    y='count',
                    color='count',
                    color_continuous_scale='blues',
                    title='Browsing Activity by Day of Week',
                    labels={'day_of_week': 'Day of Week', 'count': 'Visit Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Daily activity over time
                st.subheader("Daily Activity Over Time")
                
                daily_counts = history_time_df.groupby('date').size().reset_index(name='count')
                daily_counts['date'] = pd.to_datetime(daily_counts['date'])
                daily_counts = daily_counts.sort_values('date')
                
                fig = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    title='Daily Browsing Activity Over Time',
                    labels={'date': 'Date', 'count': 'Visit Count'}
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Page Visits"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Heat map of activity by hour and day of week
                st.subheader("Activity Heatmap by Hour and Day")
                
                heatmap_data = history_time_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
                
                # Ensure all days are in the correct order
                heatmap_pivot = heatmap_pivot.reindex(day_order)
                
                fig = px.imshow(
                    heatmap_pivot,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Visit Count"),
                    # x=[str(i) for i in range(24)],
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    color_continuous_scale="blues",
                    title="Browsing Activity Heatmap by Hour and Day"
                )
                fig.update_layout(
                    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No timestamp data found in your Chrome history.")