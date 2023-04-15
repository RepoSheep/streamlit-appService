# import altair as alt
# import pandas as pd
# import streamlit as st
# from vega_datasets import data

# st.set_page_config(
#     page_title="Time series annotations", page_icon="⬇", layout="centered"
# )


# @st.experimental_memo
# def get_data():
#     source = data.stocks()
#     source = source[source.date.gt("2004-01-01")]
#     return source


# @st.experimental_memo(ttl=60 * 60 * 24)
# def get_chart(data):
#     hover = alt.selection_single(
#         fields=["date"],
#         nearest=True,
#         on="mouseover",
#         empty="none",
#     )

#     lines = (
#         alt.Chart(data, title="Evolution of stock prices")
#         .mark_line()
#         .encode(
#             x="date",
#             y="price",
#             color="symbol",
#             # strokeDash="symbol",
#         )
#     )

#     # Draw points on the line, and highlight based on selection
#     points = lines.transform_filter(hover).mark_circle(size=65)

#     # Draw a rule at the location of the selection
#     tooltips = (
#         alt.Chart(data)
#         .mark_rule()
#         .encode(
#             x="yearmonthdate(date)",
#             y="price",
#             opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
#             tooltip=[
#                 alt.Tooltip("date", title="Date"),
#                 alt.Tooltip("price", title="Price (USD)"),
#             ],
#         )
#         .add_selection(hover)
#     )

#     return (lines + points + tooltips).interactive()


# st.title("⬇ Time series annotations")

# st.write("Give more context to your time series using annotations!")

# col1, col2, col3 = st.columns(3)
# with col1:
#     ticker = st.text_input("Choose a ticker (⬇💬👇ℹ️ ...)", value="⬇")
# with col2:
#     ticker_dx = st.slider(
#         "Horizontal offset", min_value=-30, max_value=30, step=1, value=0
#     )
# with col3:
#     ticker_dy = st.slider(
#         "Vertical offset", min_value=-30, max_value=30, step=1, value=-10
#     )

# # Original time series chart. Omitted `get_chart` for clarity
# source = get_data()
# chart = get_chart(source)

# # Input annotations
# ANNOTATIONS = [
#     ("Mar 01, 2008", "Pretty good day for GOOG"),
#     ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
#     ("Nov 01, 2008", "Market starts again thanks to..."),
#     ("Dec 01, 2009", "Small crash for GOOG after..."),
# ]

# # Create a chart with annotations
# annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
# annotations_df.date = pd.to_datetime(annotations_df.date)
# annotations_df["y"] = 0
# annotation_layer = (
#     alt.Chart(annotations_df)
#     .mark_text(size=15, text=ticker, dx=ticker_dx, dy=ticker_dy, align="center")
#     .encode(
#         x="date:T",
#         y=alt.Y("y:Q"),
#         tooltip=["event"],
#     )
#     .interactive()
# )

# # Display both charts together
# st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)

import streamlit as st
st.set_page_config(page_title="Bridging the Gap - Matching Jobs to Skills", layout="wide")
# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"

import pandas as pd
# import plotly.express as px
import ast
import sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher

def getAreaOfTraing(input):
    course_list = ast.literal_eval(input)
    try:
        description = course_list[0]['description'].lower()
        return description
    except:
        return ""

dirpath = 'C:/Users/user/Desktop/EBA5006 - Big Data Analytics/4. Practice Module (Group 4)/2. Data/20_script/'

@st.cache_data
def load_data():
    course_data1 = pd.read_csv(dirpath + 'course_extractedskill.csv', header=0)
    course_data2 = pd.read_csv(dirpath + 'course_extractedskill2.csv', header=0)
    course_data = pd.concat([course_data1, course_data2])
    course_data = course_data.reset_index()
    
    course_data['area_of_training'] = course_data['areaOfTrainings'].apply(getAreaOfTraing)
    col_subset_course = ['referenceNumber','title','area_of_training','SkillsTaxonomy']
    course_subset_data = course_data[col_subset_course] 
    course_subset_data = course_subset_data.dropna()
    course_subset_data['combined_features'] = course_subset_data['title'].str.lower() + ", " + course_subset_data['SkillsTaxonomy']
    course_subset_data['SkillsTaxonomy_length'] = course_subset_data['SkillsTaxonomy'].apply(lambda x: len(x))
    course_subset_data = course_subset_data[course_subset_data['SkillsTaxonomy_length'] >= 2]   # remove courses with less than 2 characters, skills names are usually more than 3 characters
    course_subset_data.drop(['SkillsTaxonomy_length'], axis=1, inplace=True) ; course_subset_data
    course_subset_data['combined_features'] = course_subset_data['title'] + ", " + course_subset_data['SkillsTaxonomy']
        
    jobs_data = pd.read_csv(dirpath + 'withDescSkills_extractedSkills.csv'
                            , header=0
                            , sep=';'
                            , quotechar='@'
                            , quoting=2
                            , encoding='utf-8')
    
    col_subset_jobs = ['jobPostId','title','skills_extracts']
    jobs_subset_data = jobs_data[col_subset_jobs]
    jobs_subset_data = jobs_subset_data.dropna() ; jobs_subset_data
    jobs_subset_data['skills_extracts_length'] = jobs_subset_data['skills_extracts'].apply(lambda x: len(x))
    jobs_subset_data = jobs_subset_data[jobs_subset_data['skills_extracts_length'] >= 2 | jobs_subset_data['skills_extracts'].str.contains('r')]   
    jobs_subset_data["combined_features"] = jobs_subset_data["title"].str.lower() + ", "+ jobs_subset_data["skills_extracts"]
    jobs_subset_data.drop(['skills_extracts_length'], axis=1, inplace=True) ; jobs_subset_data
        
    return course_subset_data, jobs_subset_data

course_subset_data, jobs_subset_data = load_data()


title = "Course Recommendation Engine"
st.title(title)

st.write("First of all, welcome! This is the place where you can input the jobs of your interest and it will return the relevant courses based on the typical skills required.")
st.markdown("##")

# add search panel and search button
# main_layout, search_layout = st.columns([10, 1])
# options = main_layout.multiselect('Which job are you interested in?', jobs_subset_data["title"].unique())
# show_recommended_job_btn = search_layout.button("search")

user_input = st.text_input("Enter job here 👇.  For example, 'Data Scientist'",)

if user_input:
        st.write("You entered: ", user_input)

# add widgets on sidebar
recommended_course_num = st.sidebar.slider("Recommended course number", min_value=5, max_value=10)
# show_score = st.sidebar.checkbox("Show score")


# Find similar jobs based on the input job
# Function to calculate string similarity ratio
def similarity_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Create a subset of dataframe for job title
df_title = pd.DataFrame(jobs_subset_data['title'])

# Function to find similar job titles
def find_similar_job_titles(df_title, job_title, similarity_threshold):
    similar_job_titles = []
    count_df = 0
    for title in df_title['title']:
        similarity = similarity_ratio(job_title.lower(), title.lower())
        if similarity >= similarity_threshold:
            similar_job_titles.append(count_df)
        count_df += 1
    return similar_job_titles

# Find similar job titles for a given job title
job_title = user_input
similarity_threshold = 0.7  # Adjust the similarity threshold as needed
similar_titles = find_similar_job_titles(df_title, job_title, similarity_threshold)

sim_jobs_subset_data = pd.DataFrame(columns = jobs_subset_data.columns)

if len(similar_titles) > 0:
    for title in similar_titles:
        new_row = pd.Series(jobs_subset_data.iloc[title], name=-1)
        sim_jobs_subset_data = sim_jobs_subset_data.append(new_row)
        sim_jobs_subset_data.index += 1
    sim_jobs_subset_data = sim_jobs_subset_data.sort_index()
else:
    st.write("No similar job titles found.")


# Create the recommender object
tfidf = TfidfVectorizer(ngram_range=(1, 4)) # up to 4-grams, all combinations of the 4 words
tfidf.fit(course_subset_data['combined_features'])  # fit the vectorizer to the corpus
tfidf_matrix = tfidf.transform(course_subset_data['combined_features'])
tfidf_cosine_sim = cosine_similarity(tfidf_matrix)

for ngram in range(1,4):
    CV_job = CountVectorizer(ngram_range=(ngram, 3))
    CV_X = CV_job.fit_transform(sim_jobs_subset_data['skills_extracts'])

    # Get the feature names (words)
    feature_names = CV_job.get_feature_names_out()

    # Sum the occurrences of each word
    word_counts = CV_X.sum(axis=0)

    # Convert the word counts to a list
    word_counts = word_counts.tolist()[0]

    # Create a dictionary of word counts with word as key and count as value
    word_count_dict = dict(zip(feature_names, word_counts))

    # Sort the word count dictionary by value (count) in descending order
    sorted_word_count_dict = dict(sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True))

    # Get the top 10 most frequent words
    top_10_ngram_words = list(sorted_word_count_dict.keys())[:10]

    #query = "data engineering, big data"
    skills_query = ', '.join(top_10_ngram_words)
    
    skills_vector = tfidf.transform([skills_query]) # transform the query into a vector (sparse matrix)
    skills_sim = cosine_similarity(skills_vector, tfidf_matrix).flatten() # calculate the similarity between the query and all the courses, flatten the matrix into a vector

    top_10_indices = skills_sim.argsort()[::-1][:recommended_course_num]
    st.write("\nRecommended courses for:\n {question} \n".format(question=skills_query))
    k = 0
    recommendations = pd.DataFrame()
    for i in top_10_indices:
        course = course_subset_data.iloc[i]
        recommendations = recommendations.append(course[['title', 'referenceNumber', 'combined_features']])
        k += 1
        # st.write('({})'.format(k),':',course['title'] ,':',course['SkillsTaxonomy'], )
    
    recommendations.columns = ['Course Title', 'Course Reference Number', 'Course Skills']
    
    recommendations.reset_index(drop=True, inplace=True)
    recommendations.index = np.arange(1, len(recommendations) + 1)
    
    st.write(recommendations)
    st.markdown("##")
