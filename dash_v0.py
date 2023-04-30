import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd

#set plotly template
px.defaults.template = "ggplot2"

# Load IMDb Top 250 movies data
df = pd.read_csv("IMDB Top 250 Movies.csv")

# Define a function to convert the run_time column to minutes
def convert_to_minutes(time_str):
    hours = 0
    minutes = 0
    
    if 'h' in time_str:
        hours = int(time_str.split('h')[0].strip())
    
    if 'm' in time_str:
        minutes = int(time_str.split('m')[0].split()[-1].strip())
    
    total_minutes = (hours * 60) + minutes
    return total_minutes

# Apply the conversion function to the run_time column
df['run_time'] = df['run_time'].apply(convert_to_minutes)

#split genre column into multiple columns, save them to the same dataframe
df1 = df["genre"].str.split(",", expand = True)
df1.columns = ["genre1", "genre2", "genre3"]
df2 = pd.concat([df, df1], axis = 1)
df2 = df2.drop(["genre"], axis = 1)

#top directors in the top 250 movies
top_directors = df2["directors"].value_counts().head(20)

# Create a DataFrame for treemap visualization
director_data = pd.DataFrame({"Director": top_directors.index, "Count": top_directors.values})

# Sort directors by count in descending order for better visualization
director_data = director_data.sort_values(by="Count", ascending=False)

# Decade Analysis
df2["decade"] = df["year"] // 10 * 10

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "IMDb Top 250 Movies"

# Function to generate Ratings Distribution figure
def generate_ratings_figure(filtered_df):
    fig_ratings = px.histogram(filtered_df, x="rating", hover_data=["rating"])
    fig_ratings.update_layout(
        title="Rating Distribution",
        xaxis_title="Rating",
        yaxis_title="Count",
        barmode="overlay",  # Overlay both histograms
        bargap=0.04,  # Gap between bars
        showlegend=False  # Hide the legend
    )
    fig_ratings.update_traces(
        marker_color="rgba(50, 100, 200, 0.7)",
        hovertemplate="<b>Rating:</b> %{x}<br>" +
                      "<b>Count:</b> %{y}<br>",
        opacity=0.7  # Adjust the opacity of the bars
    )
    return fig_ratings

#function to generate genre figure
def generate_genre_figure(filtered_df):
    genre_counts = filtered_df["genre1"].value_counts().head(5)
    
    fig_genre = px.bar(genre_counts, 
                       x=genre_counts.index, 
                       y=genre_counts.values, 
                       color=genre_counts.index,
                       hover_data=[genre_counts.index, genre_counts.values],
                       #labels={'x':'Genre', 'y':'Count'}
                       )#text=genre_counts.values)
    
    fig_genre.update_traces(
        hovertemplate="<b>Genre:</b> %{x}<br>" +
                      "<b>Count:</b> %{y}<extra></extra>",
        opacity=0.7  # Adjust the opacity of the bars
    )

    fig_genre.update_layout(
        title="Genre Distribution",
        xaxis_title="",
        yaxis_title="Count",
        hovermode='closest',  # Show hover information for closest bar
        legend_title_text='Genre',  # Set legend title
        bargap=0.01  # Gap between bars
    )
    return fig_genre

#function to generate top rated movies figure
def generate_top_movies_figure(filtered_df):
    top_rated_movies = filtered_df.groupby(["genre1", "name"])["rating"].nlargest(3).reset_index()
    # Assign unique colors to each genre
    genres = top_rated_movies["genre1"].unique()
    colors = px.colors.qualitative.Alphabet[:len(genres)]
    color_map = {genre: color for genre, color in zip(genres, colors)}

    fig_top_movies = px.bar(top_rated_movies, 
                            x="genre1", 
                            y="rating",
                            color="genre1",
                            color_discrete_map=color_map,  # Assign unique colors to each genre
                            hover_data=["name"])
    
    fig_top_movies.update_traces(hovertemplate="<b>Genre:</b> %{x}<br>" +
                                                "<b>Rating:</b> %{y}<br>" +
                                                "<b>Movie:</b> %{customdata}<extra></extra>",
                                                opacity=0.7)  # Adjust the opacity of the bars
    
    fig_top_movies.update_layout(
        title="Top Rated Movies per Genre",
        legend_title_text="Genre",
        xaxis_title="",
        yaxis_title="Rating",
        hovermode='closest',  # Show hover information for closest bar
        bargap=0.02  # Gap between bars
    )
    return fig_top_movies

#function to generate genre percentage figure
def generate_genre_percent_figure(filtered_df):
    genre_counts = filtered_df["genre1"].value_counts()
    fig_genre_percent = px.pie(genre_counts, 
                               values=genre_counts.values, 
                               names=genre_counts.index,
                               hover_data=["genre1", genre_counts.values])
    fig_genre_percent.update_traces(
        hovertemplate="<br>".join([
            "Genre: %{label}",
            "Count: %{value}",
            "%{percent}",
        ]))
    fig_genre_percent.update_layout(
        title="Percentage of Movies per Genre",
    )
    return fig_genre_percent

#function to generate yearly ratings figure
def generate_yearly_ratings_figure(filtered_df):
    yearly_ratings = filtered_df.groupby("year")["rating"].mean().reset_index()
    fig_yearly_ratings = px.line(yearly_ratings, x="year", y="rating")

    # Format hover data
    fig_yearly_ratings.update_traces(
        hovertemplate="Year: %{x}<br>Average Rating: %{y:.2f}")
    
    fig_yearly_ratings.update_layout(
        title="Average Movie Ratings Over the Years",
        xaxis_title="Year",
        yaxis_title="Average Rating",
    )
    return fig_yearly_ratings

# Group the DataFrame by year and genre and count the number of movies
#movies_by_genre_year = df2.groupby(['year', 'genre1']).size().reset_index(name='Count')

#function to generate yearly genre figure
def generate_yearly_genre_figure(filtered_df):
    movies_by_genre_year = filtered_df.groupby(['year', 'genre1']).size().reset_index(name='Count')

    fig_yearly_genre = px.scatter(movies_by_genre_year, x='year', y='Count', color='genre1',
                 labels={'Year': 'Year', 'Count': 'Number of movies'},
                 title='Movies Released by Genre Over the Years',
                 hover_data=['genre1', 'Count'],
                 symbol='genre1')

    fig_yearly_genre.update_traces(mode='markers',
                                   marker=dict(size=10),
                                   hovertemplate="<b>Year:</b> %{x}<br>" +
                                                 "<b>Genre:</b> %{customdata[0]}<br>" +
                                                 "<b>Count:</b> %{y}<extra></extra>")
    
    fig_yearly_genre.update_layout(
        legend=dict(title='Genre'),
        hovermode='closest' # Show hover information for closest point)
    )
    return fig_yearly_genre

#function to generate fig_top_dir_map
def generate_top_dir_map_figure(filtered_df):
    top_directors = filtered_df["directors"].value_counts().head(20)
    director_data = pd.DataFrame({"Director": top_directors.index, "Count": top_directors.values})
    director_data = director_data.sort_values(by="Count", ascending=False)

    fig_top_dir_map = go.Figure(go.Treemap(
        labels=director_data["Director"],
        parents=["" for _ in director_data["Director"]],
        values=director_data["Count"],
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>No. Movies: %{value}",
        hoverlabel=dict(namelength=0),  # Remove trace name from hover label
    ))
    fig_top_dir_map.update_layout(
        title="Top Rated Directors",
        autosize=True,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    return fig_top_dir_map


# Movie Duration Distribution==========???????????
fig_duration = px.histogram(df, x="run_time", nbins=20)
fig_duration.update_layout(
    title="Movie Duration Distribution",
    xaxis_title="Duration (minutes)",
    yaxis_title="Count"
)

#function to generate fig_genre_decade
def generate_genre_decade_figure(filtered_df):
    movies_by_genre_decade = filtered_df.groupby(['decade', 'genre1']).size().reset_index(name='Count')
    fig_genre_decade = px.scatter(movies_by_genre_decade, x='decade', y='Count', color='genre1', size='Count',
                 labels={'genre1': 'Genre', 'decade': 'Decade', 'Count': 'Number of Movies'},
                 title='Number of Movies Released by Genre (by Decade)',
                 text='Count')
    fig_genre_decade.update_traces(mode='markers', marker=dict(size=8))
    return fig_genre_decade

#function to generate fig_wordcloud
def generate_wordcloud_figure(filtered_df):
    wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(" ".join(filtered_df["name"]))

    fig_wordcloud = px.imshow(wordcloud, template=None)
    fig_wordcloud.update_xaxes(visible=False).update_yaxes(visible=False)
    fig_wordcloud.update_layout(
        title="Word Cloud of Movie Titles",
        hovermode=False,  # Disable hover interactions
        )
    return fig_wordcloud

#generate genre list
genre_list = ['All Genres'] + list(df2['genre1'].sort_values().unique())
#generate year list
year_list = ['All Years'] + list(df2['year'].sort_values(ascending=False).unique())

# Set up the title style
title_style = {
    'text-align': 'center',
    'fontSize': '40px',
    'padding': '20px',
    'background-color': '#2E3E4E',
    'color': '#FFFFFF',
    'margin': 'auto',  # Center the title horizontally
    'max-width': '1200px',  # Limit the width of the title
}
# Set up the container style
container_style = {
    'margin': 'auto',  # Center the container horizontally
    'max-width': '1200px',  # Limit the width of the container
}
# Set up the card style
card_style = {
    'margin-bottom': '20px',
}
# Set up the dropdown style
dropdown_style = {
    'width': '200px',  # Set the width of the dropdown
    'margin': '0 10px',  # Add some margin between dropdowns
    'display': 'inline-block',  # Display the dropdowns inline
    'vertical-align': 'middle',  # Align the dropdowns vertically
    'background-color': '#F0F0F0',
    'color': '#333333'
}
#setup label style
label_style = {
    'font-size': '18px',
    'font-weight': 'bold',
    'color': '#333333',  # gray
    'text-align': 'center'
}

# Create the layout
app.layout = html.Div([
    html.H1("IMDB Top 250 Movies Dashboard", style=title_style),
    html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Label('Select Genre:', style=label_style),
                                dcc.Dropdown(
                                    id='genre-dropdown',
                                    options=[{'label': i, 'value': i} for i in genre_list],
                                    value=genre_list[0],
                                    clearable=False,
                                    style=dropdown_style,  # Apply the dropdown style
                                ),
                            ]
                        )
                    ),
                    width=6#,
                    #style={'margin': 'auto'}  # Center align the column
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Label('Select Year:', style=label_style),
                                dcc.Dropdown(
                                    id='year-dropdown',
                                    options=[{'label': i, 'value': i} for i in year_list],
                                    value=year_list[0],
                                    clearable=False,
                                    style=dropdown_style,  # Apply the dropdown style
                                ),
                            ]
                        )
                    ),
                    width=6,
                    style={'margin-bottom': '20px'}  # Center align the column
                ),
            ]
        ),
        style={'text-align': 'center', 'margin-bottom': '20px'}  # Center align the div
    ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='generate_yearly_genre_figure') #graph 1
        ], width=6),
        dbc.Col([
            dcc.Graph(id='generate_ratings_figure') #graph 2
        ], width=6)
    ],
    style=container_style,
    ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='generate_genre_percent_figure') #graph 3
        ], width=6),
        dbc.Col([
            dcc.Graph(id='generate_top_movies_figure') #graph 4
        ], width=6)
    ],
    style=container_style,
    ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='generate_yearly_ratings_figure') #graph 5
        ], width=6),
        dbc.Col([
            dcc.Graph(id='generate_genre_figure') #graph 6
        ], width=6)
    ],
    style=container_style,
    ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='generate_top_dir_map_figure') #graph 7
        ], width=6),
        dbc.Col([
            dcc.Graph(id='generate_wordcloud_figure') #graph 8
        ], width=6)
    ],
    style=container_style,
    ),
])

@app.callback(
    dash.Output(component_id='generate_yearly_genre_figure', component_property='figure'),
    dash.Output(component_id='generate_ratings_figure', component_property='figure'),
    dash.Output(component_id='generate_genre_percent_figure', component_property='figure'),
    dash.Output(component_id='generate_top_movies_figure', component_property='figure'),
    dash.Output(component_id='generate_yearly_ratings_figure', component_property='figure'),
    dash.Output(component_id='generate_genre_figure', component_property='figure'),
    #dash.Output(component_id='generate_genre_decade_figure', component_property='figure'),
    dash.Output(component_id='generate_top_dir_map_figure', component_property='figure'),
    dash.Output(component_id='generate_wordcloud_figure', component_property='figure'),
    dash.Input(component_id='genre-dropdown', component_property='value'),
    dash.Input(component_id='year-dropdown', component_property='value')
)
def update_figures(genre_value, year_value):
    # Filtering data based on selected genre and year
    #filtered_df = df2[df2['genre1'].isin([genre_value]) & df2['year'].isin([int(year_value)])]
    filtered_df = df2  # Assuming df2 is your original dataframe
    
    if genre_value != "All Genres":
        filtered_df = filtered_df[filtered_df['genre1'] == genre_value]
    
    if year_value != "All Years":
        filtered_df = filtered_df[filtered_df['year'] == year_value]

    # updating ratgins_figure
    fig_ratings = generate_ratings_figure(filtered_df)

    # updating genre_figure
    fig_genre = generate_genre_figure(filtered_df)

    # updating yearly_ratings_figure
    fig_yearly_ratings = generate_yearly_ratings_figure(filtered_df)

    # updating yearly_genre_figure
    fig_yearly_genre = generate_yearly_genre_figure(filtered_df)

    # Updating top_dir_map_figure
    fig_top_dir_map = generate_top_dir_map_figure(filtered_df)

    # Updating genre_decade_figure
    #fig_genre_decade = generate_genre_decade_figure(filtered_df)

    # Updating wordcloud_figure
    fig_wordcloud = generate_wordcloud_figure(filtered_df)

    # Updating top movies graph
    fig_top_movies = generate_top_movies_figure(filtered_df)
    
    # Updating genre percentage graph
    fig_genre_percent = generate_genre_percent_figure(filtered_df)
    
    return (fig_yearly_genre,
            fig_ratings,
            fig_genre_percent,
            fig_top_movies,
            fig_yearly_ratings,
            fig_genre,     
            fig_top_dir_map,
            fig_wordcloud)


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
 
