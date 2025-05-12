import streamlit as st
import pandas as pd
import plotly.express as px
from YoutubePreprocessing import YouTubePreprocessor
processor = YouTubePreprocessor()

class YtAnalyser:
    @st.cache_data
    def compute_time_patterns(df, activity_type):
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
        
        # Create time bins for time of day analysis
        time_df['time_of_day'] = pd.cut(
            time_df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)']
        )
        
        return time_df

    @st.cache_data
    def analyze_topics(df, text_column):
        """Analyze topics in a dataframe with text data."""
        if df is None or text_column not in df.columns or df.empty:
            return None, None, None
        
        # Only pass necessary columns to reduce memory usage
        required_cols = [text_column]
        if 'title' in df.columns:
            required_cols.append('title')
        if 'url' in df.columns:
            required_cols.append('url')
        
        subset_df = df[required_cols].copy()
        
        return processor.analyze_topic_diversity(subset_df, text_column)

    def show_youtube_analysis(self,data):
        """Display YouTube-specific analysis with improved performance."""
        st.title("📺 YouTube Analysis")
        
        # Check if YouTube data is available
        if not data:
            st.warning("Please upload YouTube history files to see this analysis.")
            return
        
        if 'youtube_watch' not in data and 'youtube_search' not in data:
            st.warning("No YouTube data found. Please upload YouTube watch history and/or search history files.")
            return
        
        tabs = st.tabs(["Watch History", "Search History", "Topic Analysis", "Time Patterns"])
        
        # Watch History Tab - improved performance
        with tabs[0]:
            if 'youtube_watch' in data and not data['youtube_watch'].empty:
                st.header("Watch History Analysis")
                watch_df = data['youtube_watch']
                
                # Summary statistics - quick to compute
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_videos = len(watch_df)
                    st.metric("Total Videos Watched", f"{total_videos:,}")
                
                with col2:
                    if 'channel' in watch_df.columns:
                        print(watch_df['channel'].isnull().sum(), "null values out of", len(watch_df))
                        print(watch_df['channel'].dropna().unique())
                        unique_channels = watch_df['channel'].nunique()
                        st.metric("Unique Channels", f"{unique_channels:,}")

                    else:
                        st.metric("Unique Channels", "N/A")
                
                with col3:
                    if 'timestamp' in watch_df.columns and not watch_df['timestamp'].isnull().all():
                        try:
                            date_range = (watch_df['timestamp'].max() - watch_df['timestamp'].min()).days
                            st.metric("Date Range", f"{date_range} days")
                        except:
                            st.metric("Date Range", "N/A")
                    else:
                        st.metric("Date Range", "N/A")
                
                # Channel analysis - only if data is available and with spinners
                if 'channel' in watch_df.columns:
                    st.subheader("Channel Analysis")
                    with st.spinner("Analyzing channel data..."):
                        # Top channels - limit to 10 for better performance
                        # top_channels = watch_df['channel'].value_counts().head(10)
                        
                        # fig = px.bar(
                        #     x=top_channels.index,
                        #     y=top_channels.values,
                        #     labels={'x': 'Channel', 'y': 'Video Count'},
                        #     title='Top 10 Most Watched Channels',
                        #     color=top_channels.values,
                        #     color_continuous_scale='reds'
                        # )
                        # fig.update_layout(xaxis_tickangle=-45)
                        # st.plotly_chart(fig, use_container_width=True)
                        top_channels = watch_df['channel'].value_counts().head(10).reset_index()
                        top_channels.columns = ['channel', 'count']

                        fig = px.bar(
                            top_channels,
                            x='channel',
                            y='count',
                            color='count',
                            color_continuous_scale='reds',
                            labels={'channel': 'Channel', 'count': 'Video Count'},
                            title='Top 10 Most Watched Channels'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Content analysis - with performance spinner
                st.subheader("Content Analysis")
                with st.spinner("Generating content wordcloud..."):
                    if 'processed_title' in watch_df.columns:
                        # Sample data for wordcloud to improve performance
                        if len(watch_df) > 1000:
                            sample_df = watch_df.sample(1000)
                        else:
                            sample_df = watch_df
                            
                        wordcloud = processor.generate_wordcloud(sample_df, 'processed_title')
                        
                        if wordcloud:
                            st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{wordcloud}' style='max-width: 100%;'></div>", unsafe_allow_html=True)
                            st.caption("Word cloud generated from video titles")
                        else:
                            st.info("Not enough data to generate a word cloud.")
                    else:
                        st.info("Processed title data not available for wordcloud generation.")
                
                # Most watched videos - more efficient processing
                st.subheader("Most Watched Videos")
                if len(watch_df) > 0 and 'title' in watch_df.columns and 'url' in watch_df.columns:
                    with st.spinner("Finding most watched videos..."):
                        # More efficient groupby operation
                        video_counts = watch_df.groupby(['title', 'url']).size().reset_index(name='count')
                        video_counts = video_counts.sort_values('count', ascending=False).head(10)
                        
                        for i, (_, row) in enumerate(video_counts.iterrows()):
                            st.markdown(f"**{i+1}.** [{row['title']}]({row['url']}) - Watched {row['count']} times")
                else:
                    st.info("No watch history data available.")
            else:
                st.info("No YouTube watch history found. Please upload watch history file.")
        
        # Search History Tab - with performance improvements
        with tabs[1]:
            if 'youtube_search' in data and not data['youtube_search'].empty:
                st.header("Search History Analysis")
                search_df = data['youtube_search']
                
                # Summary statistics
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    total_searches = len(search_df)
                    st.metric("Total Searches", f"{total_searches:,}")
                
                with col2:
                    if 'timestamp' in search_df.columns and not search_df['timestamp'].isnull().all():
                        try:
                            date_range = (search_df['timestamp'].max() - search_df['timestamp'].min()).days
                            st.metric("Date Range", f"{date_range} days")
                        except:
                            st.metric("Date Range", "N/A")
                    else:
                        st.metric("Date Range", "N/A")
                
                # Search term analysis - with performance improvements
                st.subheader("Search Term Analysis")
                
                # Most common search terms
                if 'query' in search_df.columns:
                    with st.spinner("Analyzing search terms..."):
                        search_counts = search_df['query'].value_counts().head(15)
                        
                        fig = px.bar(
                            x=search_counts.index,
                            y=search_counts.values,
                            labels={'x': 'Search Term', 'y': 'Count'},
                            title='Most Common Search Terms',
                            color=search_counts.values,
                            color_continuous_scale='oranges'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Search term wordcloud - with sampling for performance
                with st.spinner("Generating search wordcloud..."):
                    if 'query' in search_df.columns:
                        # Sample data for wordcloud to improve performance
                        if len(search_df) > 1000:
                            sample_df = search_df.sample(1000)
                        else:
                            sample_df = search_df
                            
                        wordcloud = processor.generate_wordcloud(sample_df, 'query')
                        
                        if wordcloud:
                            st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{wordcloud}' style='max-width: 100%;'></div>", unsafe_allow_html=True)
                            st.caption("Word cloud generated from search queries")
                        else:
                            st.info("Not enough data to generate a word cloud.")
                    else:
                        st.info("Query data not available for wordcloud generation.")
                
                # Recent searches - limit to 10 for performance
                st.subheader("Recent Searches")
                if 'timestamp' in search_df.columns and 'query' in search_df.columns and len(search_df) > 0:
                    with st.spinner("Finding recent searches..."):
                        recent_searches = search_df.sort_values('timestamp', ascending=False).head(10)
                        
                        for i, (_, row) in enumerate(recent_searches.iterrows()):
                            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['timestamp']) else "Unknown date"
                            st.markdown(f"**{i+1}.** \"{row['query']}\" - {timestamp}")
                else:
                    st.info("No timestamp data available for searches.")
            else:
                st.info("No YouTube search history found. Please upload search history file.")
        
        # Topic Analysis Tab - with lazy loading approach
        with tabs[2]:
            st.header("Topic Analysis")
            
            # Only perform topic analysis when the tab is selected
            if st.checkbox("Generate Topic Analysis", key="run_topic_analysis", value=False):
                with st.spinner("Analyzing topics... this may take a moment"):
                    # Combine watch and search data for topic analysis
                    topic_data = []
                    
                    if 'youtube_watch' in data and not data['youtube_watch'].empty:
                        if 'processed_title' in data['youtube_watch'].columns:
                            # Only use necessary columns
                            cols_to_use = ['processed_title', 'title']
                            if 'url' in data['youtube_watch'].columns:
                                cols_to_use.append('url')
                            topic_data.append(data['youtube_watch'][cols_to_use])
                    
                    if 'youtube_search' in data and not data['youtube_search'].empty:
                        if 'processed_query' in data['youtube_search'].columns:
                            # Rename processed_query to processed_title for consistency
                            search_topic = data['youtube_search'][['processed_query', 'query']].copy()
                            search_topic.columns = ['processed_title', 'title']
                            search_topic['url'] = None
                            topic_data.append(search_topic)
                    
                    if topic_data:
                        combined_topic_df = pd.concat(topic_data)
                        
                        # Use cached function for topic clustering
                        cluster_terms, cluster_distribution, clustered_data = YtAnalyser.analyze_topics(
                            combined_topic_df, 'processed_title'
                        )
                        
                        if cluster_terms and cluster_distribution:
                            # Show topic distribution chart
                            topic_dist_df = pd.DataFrame({
                                'Topic': list(cluster_distribution.keys()),
                                'Percentage': [v * 100 for v in cluster_distribution.values()]
                            })
                            
                            fig = px.pie(
                                topic_dist_df,
                                values='Percentage',
                                names='Topic',
                                title='YouTube Content Topic Distribution',
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show topic keywords
                            st.subheader("Topic Keywords")
                            for topic, terms in cluster_terms.items():
                                with st.expander(f"{topic} ({topic_dist_df[topic_dist_df['Topic']==topic]['Percentage'].values[0]:.1f}%)"):
                                    st.write(", ".join(terms))
                        else:
                            st.info("Not enough data for topic clustering analysis.")
                        
                        # Show keyword frequency if available
                        if 'keywords' in combined_topic_df.columns:
                            st.subheader("Keyword Frequency")
                            
                            # Flatten all keywords
                            all_keywords = [kw for sublist in combined_topic_df['keywords'].dropna() for kw in sublist if isinstance(sublist, list)]
                            
                            if all_keywords:
                                keyword_counts = pd.Series(all_keywords).value_counts().head(20)
                                
                                fig = px.bar(
                                    x=keyword_counts.index,
                                    y=keyword_counts.values,
                                    labels={'x': 'Keyword', 'y': 'Frequency'},
                                    title='Top 20 Keywords in YouTube Content',
                                    color=keyword_counts.values,
                                    color_continuous_scale='purples'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No keywords extracted from content.")
                        else:
                            st.info("Keyword data not available.")
                    else:
                        st.info("Not enough data for topic analysis. Please upload YouTube history files.")
            else:
                st.info("Check the box above to generate topic analysis. Note: This may take a few moments to process.")
        
        # Time Patterns Tab - with efficient time data calculation
        with tabs[3]:
            st.header("Time Patterns")
            
            # Combine watch and search data for time analysis using cached function
            watch_time_df = None
            search_time_df = None
            
            if 'youtube_watch' in data and not data['youtube_watch'].empty:
                watch_time_df = YtAnalyser.compute_time_patterns(data['youtube_watch'], 'Watch')
            
            if 'youtube_search' in data and not data['youtube_search'].empty:
                search_time_df = YtAnalyser.compute_time_patterns(data['youtube_search'], 'Search')
            
            if watch_time_df is not None or search_time_df is not None:
                with st.spinner("Analyzing time patterns..."):
                    time_data = []
                    if watch_time_df is not None:
                        time_data.append(watch_time_df)
                    if search_time_df is not None:
                        time_data.append(search_time_df)
                    
                    combined_time_df = pd.concat(time_data)
                    
                    # Hourly distribution
                    st.subheader("Hourly Activity Distribution")
                    hourly_counts = combined_time_df.groupby(['hour', 'type']).size().reset_index(name='count')
                    
                    fig = px.line(
                        hourly_counts,
                        x='hour',
                        y='count',
                        color='type',
                        labels={'hour': 'Hour of Day', 'count': 'Activity Count', 'type': 'Activity Type'},
                        title='Hourly Distribution of Activity Types')
                    
                    # Add better axis labels
                    fig.update_layout(
                        xaxis_title="Hour of Day (24-hour format)",
                        yaxis_title="Number of Activities",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time of day distribution
                    st.subheader("Time of Day Distribution")
                    time_of_day_counts = combined_time_df.groupby(['time_of_day', 'type']).size().reset_index(name='count')
                    
                    fig = px.bar(
                        time_of_day_counts, 
                        x='time_of_day', 
                        y='count', 
                        color='type',
                        barmode='group',
                        title='Activity Distribution by Time of Day',
                        labels={'time_of_day': 'Time of Day', 'count': 'Activity Count', 'type': 'Activity Type'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Day of week distribution
                    st.subheader("Day of Week Distribution")
                    
                    # Set proper day of week order
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    combined_time_df['day_of_week'] = pd.Categorical(combined_time_df['day_of_week'], categories=day_order, ordered=True)
                    
                    dow_counts = combined_time_df.groupby(['day_of_week', 'type']).size().reset_index(name='count')
                    
                    fig = px.bar(
                        dow_counts,
                        x='day_of_week',
                        y='count',
                        color='type',
                        barmode='group',
                        title='Activity Distribution by Day of Week',
                        labels={'day_of_week': 'Day of Week', 'count': 'Activity Count', 'type': 'Activity Type'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid timestamp data found in your YouTube watch or search history.")

