import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Output, State, Input, Event

categories = ['Handphones', 'Laptops', 'Desktops']
tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])

def generateTable(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
                    
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Dashboard',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),

    html.Div(children='''
        Dash: A web application framework for Python.''', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div([
            dcc.Input(id='review', value='Enter Review', type='text'),
            html.Button(id='submit-button', type='submit', children='Submit'),
            html.Div(id='output_div')
    ]),

    # html.Div(children = [
    #     html.H4(children='Output'),
    #     generateTable(tableView)
    # ]),

    dcc.Graph(
        id='sentiment analysis',
        figure={ 
            'data' : [
                go.Bar(
                {'x': categories, 'y': [20, 14, 23], 'name': 'positive sentiment'}
                ),
                go.Bar(
                    {'x': categories, 'y': [-15, -7, -13], 'name': 'negative sentiment'}
                )
            ],
            'layout': {
                'barmode': 'group',
                'title': 'Dash Data Visualization',
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    ),

    dcc.Graph(
        id='sentiment analysis time series analysis (Laptop)',
        figure={
            'data' : [
                go.Scatter(
                {'x': [1,2,3], 'y': [10, 8, 2], 'name': 'positive sentiment'}
                ),
                go.Scatter(
                    {'x': [1,2,3], 'y': [3, 7, 12], 'name': 'negative sentiment'}
                )
            ],
            'layout': {
                'barmode': 'group',
                'title': 'Time Series Sentiment Analysis(Laptop)',
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
    # dcc.Graph(
    #     id='example-graph-2',
    #     figure={
    #         'data': [
    #             {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
    #             {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
    #         ],
    #         'layout': {
    #             'title': 'Dash Data Visualization',
    #             'plot_bgcolor': colors['background'],
    #             'paper_bgcolor': colors['background'],
    #             'font': {
    #                 'color': colors['text']
    #             }
    #         }
    #     }
    # )
])

# tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])
@app.callback(Output('output_div', 'children'),
            [],
            [State('review', 'value')],
            [Event('submit-button', 'click')]
)

def update_output(review):
    # tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])
    # run review through sentiment
    sentiment = runModelSentiment(review)
    # # run review through category classification 
    category = runModelCategoryClassification(review)

    # print('table view: ')
    # print(tableView.columns)
    
    row = pd.Series([review, category, sentiment], index=tableView.columns)
    view = tableView.append(row, ignore_index = True)
    print(row)
    return generateTable(view)

def runModelSentiment(input_value):
    return 'Positive'


def runModelCategoryClassification(input_value):
    return 'test class'

if __name__ == '__main__':
    app.run_server(debug=True)