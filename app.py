import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Output, State, Input
import dash_table

categories = ['Handphones', 'Laptops', 'Desktops']
tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])
displaySentByCategoryX = []
displaySentByCategoryY = []

def generateTable(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        style={
            'background': '#111111',
            'text': '#7FDBFF',
            'align': 'center'
        }
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

     # drop down list to select which category do you want to look at
    dcc.Dropdown(
        id='category dropdown',
        options=[{'label':category, 'value':category} for category in categories],
        value=categories[0]
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
            [Input('submit-button', 'n_clicks')],
            [State('review', 'value')]
            # [State('review', 'value')]
            # [Event('submit-button', 'click')]
)

def update_output(n_clicks, review):
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
    table = dash_table.DataTable(
        data=view.to_dict('rows'),
        columns=[{'id': c, 'name': c} for c in view.columns],
        style_as_list_view=True,
        style_cell={'padding': '5px'},
        style_header={
            'backgroundColor': '#111111',
            'fontWeight': 'bold',
            'color': '#7FDBFF'
        },
        style_cell_conditional=[
        {
            'backgroundColor': '#111111',
            'if': {'column_id': c},
            'textAlign': 'left',
            'color': '#7FDBFF'
        } for c in view.columns
    ],
    )
    return table

def runModelSentiment(input_value):
    if (input_value == 'test'):
        return 'negative'
    return 'positive'


def runModelCategoryClassification(input_value):
    return 'test class'

if __name__ == '__main__':
    app.run_server(debug=True)