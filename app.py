import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Output, State, Input
import dash_table
import base64
import datetime
import io
from datetime import datetime
from sentimentAnalysisUtil import stemmed_words,removeStopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


categories = ['all','cameras', 'laptops', 'mobile phone']
tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])
brands = ['sony', 'nokia','samsung']
allBrands = []
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
        children='Text-Mining Dashboard',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    
    html.Div([
        html.H2(
        children='Enter Review:',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),   
        dcc.Input(id='review', value='', type='text'),
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Div(id='output_div')
    ],
        style = {
            'textAlign': 'center'
        }
    ),

    # html.Div(children = [
    #     html.H4(children='Output'),
    #     generateTable(tableView)
    # ]),
    html.Div([
        html.H2(
            children='Upload CSV File: ',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),   
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'color': 'white'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-data-upload')
    ]),
    # dcc.Dropdown(
    #     id='category_dropdown_bar',
    #     options=[{'label':category, 'value':category} for category in categories],
    #     value=categories[0]
    # ),

    # dcc.Graph(
    #     id='sentiment analysis by category',
        # figure={ 
        #     'data' : [
        #         go.Bar(
        #         {'x': categories, 'y': [20, 14, 23, 40], 'name': 'positive sentiment'}
        #         ),
        #         go.Bar(
        #             {'x': categories, 'y': [-15, -7, -13, -20], 'name': 'negative sentiment'}
        #         )
        #     ],
        #     'layout': {
        #         'barmode': 'group',
        #         'title': 'Dash Data Visualization',
        #         'plot_bgcolor': colors['background'],
        #         'paper_bgcolor': colors['background'],
        #         'font': {
        #             'color': colors['text']
        #         }
        #     }
        # }
    # ),

     # drop down list to select which category do you want to look at
    html.Div([
        html.H2(
            children = 'Select Your Category: ',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
        dcc.Dropdown(
            id='category_dropdown',
            options=[{'label':category, 'value':category} for category in categories],
            value=categories[0]
        )
    ]),

    dcc.Graph(
        id='sentiment analysis time series analysis (Laptop)',
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
@app.callback(Output('sentiment analysis time series analysis (Laptop)', 'figure'),
            [Input('category_dropdown', 'value')],
)

def update_graph(selected_dropdown_value):
    main_df = pd.read_csv("data/preprocessed_reviewinfo.csv", index_col=False)
    main_df['new_Date'] = pd.to_datetime(main_df["Date"], format = '%B %d, %Y')
    if selected_dropdown_value !="all":
        main_df= main_df[main_df['category']== selected_dropdown_value]
    positive_df = main_df.loc[main_df['polarity'] == 1].groupby(main_df.new_Date).agg('count')['Date'].reset_index()
    negative_df = main_df.loc[main_df['polarity'] == 0].groupby(main_df.new_Date).agg('count')['Date'].reset_index()

    return {
        'data' : [
                 {'x':positive_df['new_Date'], 'y': positive_df['Date'] ,'name':'positive sentiment'}
                ,{'x':negative_df['new_Date'], 'y': negative_df['Date'] ,'name':'negative sentiment'}
                
            ],
         'layout': {
                'barmode': 'group',
                'title': 'Time Series Sentiment Analysis (' +selected_dropdown_value+")",
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
        }
    }

# tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])
@app.callback(Output('output_div', 'children'),
            [Input('submit-button', 'n_clicks')],
            [State('review', 'value')]
            # [State('review', 'value')]
            # [Event('submit-button', 'click')]
)

def update_output(n_clicks, review):
    # tableView = pd.DataFrame(columns=['review', 'category', 'sentiment'])
    if review == '':
        return
    # run review through sentiment
    sentiment = runModelSentiment(review)
    # # run review through category classification 
    category = runModelCategoryClassification(review)

    # print('table view: ')
    # print(tableView.columns)
    
    row = pd.Series([review, category, sentiment], index=tableView.columns)
    view = tableView.append(row, ignore_index = True)
    print(row)
    table = html.Div([
        dash_table.DataTable(
            data=view.to_dict('rows'),
            columns=[{'id': c, 'name': c} for c in view.columns],
            style_as_list_view=True,
            # n_fixed_columns=3,
            style_cell={
                'padding': '5px',
                'backgroundColor': '#111111',
                'textAlign': 'left',
                'color': '#7FDBFF',
                
            },
            style_header={
                'backgroundColor': '#111111',
                'fontWeight': 'bold',
                'color': '#7FDBFF',
                'maxWidth': '180px'
            },
            style_cell_conditional=[
            {
                'backgroundColor': '#111111',
                'if': {'column_id': c},
                'textAlign': 'left',
                'color': '#7FDBFF'
            } for c in view.columns
            ],
            style_table={
                'maxHeight': '300'
            },
        )
    ])
    return table

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            print('df:')
            print(df.columns)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    positive = {}
    negative = {}
    # intiate the categories count in both positive and negative dict
    for category in categories:
        positive[category] = 0
        negative[category] = 0
    
    positiveReviews=[]
    negativeReviews=[]
    # allBrands={}

    # for brand in brands:
    #     allBrands[brand]={}
    #     allBrands[brand]['positive'] = pd.DataFrame(columns=['Review'])
    #     allBrands[brand]['negative'] = pd.DataFrame(columns=['Review'])
    
    
    for row in df.iterrows():
        review = row[1]['Content']
        sentiment = runModelSentiment(review)
        category = runModelCategoryClassification(review)
        if sentiment == 'positive':
            currentNumInCategory = positive[category]
            positive[category] = currentNumInCategory + 1
            positiveReviews.append(review)
            # updatedReview = review.lower()
            # for brand in brands:
            #     b = allBrands[brand]
            #     if brand in updatedReview:
            #         b['positive'].append(pd.Series([review], index=b['positive'].columns),ignore_index = True)
            # print('positive')
        else:
            currentNumInCategory = negative[category]
            negative[category] = currentNumInCategory - 1
            negativeReviews.append(review)
            # updatedReview = review.lower()
            # for brand in brands:
            #     b = allBrands[brand]
            #     if brand in updatedReview:
            #         b['negative'].append(pd.Series([review], index=b['negative'].columns),ignore_index = True)
            # print('negative')
    # print('all brands: ')
    # print(allBrands['sony']['positive'])
    numSampleReviews = 5
    positiveReviewsToBeDisplayed = pd.DataFrame(columns=['Review'])
    negativeReviewsToBeDisplayed = pd.DataFrame(columns=['Review'])

    # displaying entered number of Sample Reviews for both positive and negative sentiments
    for review in positiveReviews[0:numSampleReviews]:
        row = pd.Series([review], index=positiveReviewsToBeDisplayed.columns)
        positiveReviewsToBeDisplayed = positiveReviewsToBeDisplayed.append(row,ignore_index = True)
    
    for review in negativeReviews[0:numSampleReviews]:
        row = pd.Series([review], index=negativeReviewsToBeDisplayed.columns)
        negativeReviewsToBeDisplayed = negativeReviewsToBeDisplayed.append(row,ignore_index = True)

    #Analysing features of positive and negative reviews
    
    positiveReviews = removeStopwords(positiveReviews)
    negativeReviews = removeStopwords(negativeReviews)
    positive_features_df = []
    negative_features_df = []

    if len(positiveReviews) > 0:
        #for positive reviews
        count_vect_pos = CountVectorizer(max_features=5000, lowercase=True, ngram_range=(1,2))
        vectorizer_matrix_pos = count_vect_pos.fit_transform(positiveReviews)
        tfidf_transformer_pos = TfidfTransformer(use_idf=True, smooth_idf=True)
        tfidf_pos = tfidf_transformer_pos.fit_transform(vectorizer_matrix_pos)
        # df = pd.DataFrame(tfidf.toarray(), columns = count_vect.get_feature_names())
        # print(df)
        weights_pos = np.asarray(tfidf_pos.mean(axis=0)).ravel().tolist()
        weights_df_pos = pd.DataFrame({'term': count_vect_pos.get_feature_names(), 'weight': weights_pos})
        positive_features_df =(weights_df_pos.sort_values(by='weight', ascending=False).head(20))
    # print('positive: ')
    # print(positive_features_df)    

    if len(negativeReviews) > 0:
        #for negative reviews
        count_vect_neg = CountVectorizer(max_features=5000, lowercase=True, ngram_range=(1,2))
        vectorizer_matrix_neg = count_vect_neg.fit_transform(negativeReviews)
        tfidf_transformer_neg = TfidfTransformer(use_idf=True, smooth_idf=True)
        tfidf_neg = tfidf_transformer_neg.fit_transform(vectorizer_matrix_neg)
        # df = pd.DataFrame(tfidf.toarray(), columns = count_vect.get_feature_names())
        # print(df)
        weights_neg = np.asarray(tfidf_neg.mean(axis=0)).ravel().tolist()
        weights_df_neg = pd.DataFrame({'term': count_vect_neg.get_feature_names(), 'weight': weights_neg})
        negative_features_df =(weights_df_neg.sort_values(by='weight', ascending=False).head(20)) 

    categoryFeatures_df = getCategoryFeatures(numSampleReviews)
    # print('negative: ')
    # print(negative_features_df)

    # print("Positive Features:")
    # print(positive_features_df)

    # print("Negative Features:")
    # print(negative_features_df)

    # dictToReturn = {}
    # dictToReturn['positive'] = positive
    # dictToReturn['negative'] = negative

    return positive,negative,positive_features_df,negative_features_df,positiveReviewsToBeDisplayed,negativeReviewsToBeDisplayed, categoryFeatures_df, allBrands
    # return html.Div([
    #     html.H5(filename),
    #     html.H6(datetime.datetime.fromtimestamp(date)),

    #     dash_table.DataTable(
    #         data=df.to_dict('rows'),
    #         columns=[{'name': i, 'id': i} for i in df.columns]  
    #     html.Hr(),  # horizontal line

    #     # For debugging, display the raw contents provided by the web browser
    #     html.Div('Raw Content'),
    #     html.Pre(contents[0:200] + '...', style={
    #         'whiteSpace': 'pre-wrap',
    #         'wordBreak': 'break-all'
    #     })
    # ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_upload(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
        parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        # print(children)
        return generateDisplay(children[0][0],children[0][1], children[0][2], children[0][3],children[0][4],children[0][5],children[0][6],children[0][7])

def generateDisplay(positive,negative,positive_features_df,negative_features_df,positiveReviewsToBeDisplayed,negativeReviewsToBeDisplayed,categoryFeatures_df,allBrands):
    positive_values = [positive[category] for category in categories]
    negative_values = [negative[category] for category in categories]
    # print(positive)
    # print('---------------------')
    # print(negative)
    # return 'nothing'

    return html.Div([
            dcc.Graph(
                id='sentiment analysis by category',
                figure = {
                    'data' : [
                        go.Bar(
                        {'x': categories[1:], 'y': positive_values[1:], 'name': 'positive sentiment'}
                        ),
                        go.Bar(
                            {'x': categories[1:], 'y': negative_values[1:], 'name': 'negative sentiment'}
                        )
                    ],
                    'layout': {
                        'barmode': 'group',
                        'title': 'Sentiment Distribution by Category',
                        'plot_bgcolor': colors['background'],
                        'paper_bgcolor': colors['background'],
                        'font': {
                            'color': colors['text']
                        }
                    }
                }
            ),
            html.Div([
                html.H2(
                    children='Sample Positive Reviews',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                }),

                dash_table.DataTable(
                    data=positiveReviewsToBeDisplayed.to_dict('rows'),
                    columns=[{'id': c, 'name': c} for c in positiveReviewsToBeDisplayed.columns],
                    style_as_list_view=True,
                    # n_fixed_columns=3,
                    style_cell={
                        'padding': '5px',
                        'backgroundColor': '#111111',
                        'textAlign': 'left',
                        'color': '#7FDBFF',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'backgroundColor': '#111111',
                        'fontWeight': 'bold',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_table={
                        'maxHeight': '500',
                        'overflowY': 'scroll'
                    }
                    # style_cell_conditional=[
                    # {
                    #     'backgroundColor': '#111111',
                    #     'if': {'column_id': c},
                    #     'textAlign': 'left',
                    #     'color': '#7FDBFF'
                    # } for c in positive_features_df.columns
                    # ]
                )
            ]), 
             html.Div([
                html.H2(
                    children='Sample Negative Reviews',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                }),
                dash_table.DataTable(
                    data=negativeReviewsToBeDisplayed.to_dict('rows'),
                    columns=[{'id': c, 'name': c} for c in negativeReviewsToBeDisplayed.columns],
                    style_as_list_view=True,
                    # n_fixed_columns=3,
                    style_cell={
                        'padding': '5px',
                        'backgroundColor': '#111111',
                        'textAlign': 'left',
                        'color': '#7FDBFF',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'backgroundColor': '#111111',
                        'fontWeight': 'bold',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_table={
                        'maxHeight': '500',
                        'overflowY': 'scroll'
                    }
                    # style_cell_conditional=[
                    # {
                    #     'backgroundColor': '#111111',
                    #     'if': {'column_id': c},
                    #     'textAlign': 'left',
                    #     'color': '#7FDBFF'
                    # } for c in positive_features_df.columns
                    # ]
                )
            ]),
            # data table for positive features
            html.Div([
                html.H2(
                    children='Positive Features',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                }),
                dash_table.DataTable(
                    data=positive_features_df.to_dict('rows'),
                    columns=[{'id': c, 'name': c} for c in positive_features_df.columns],
                    style_as_list_view=True,
                    # n_fixed_columns=3,
                    style_cell={
                        'padding': '5px',
                        'backgroundColor': '#111111',
                        'textAlign': 'left',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_header={
                        'backgroundColor': '#111111',
                        'fontWeight': 'bold',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_table={
                        'maxHeight': '500',
                        'overflowY': 'scroll'
                    }
                    # style_cell_conditional=[
                    # {
                    #     'backgroundColor': '#111111',
                    #     'if': {'column_id': c},
                    #     'textAlign': 'left',
                    #     'color': '#7FDBFF'
                    # } for c in positive_features_df.columns
                    # ]
                )
            ], 
                style = {'display':'inline-block', 'width': '50%'}
            ), 
             html.Div([
                html.H2(
                    children='Negative Features',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                }),
                dash_table.DataTable(
                    data=negative_features_df.to_dict('rows'),
                    columns=[{'id': c, 'name': c} for c in negative_features_df.columns],
                    style_as_list_view=True,
                    # n_fixed_columns=3,
                    style_cell={
                        'padding': '5px',
                        'backgroundColor': '#111111',
                        'textAlign': 'left',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_header={
                        'backgroundColor': '#111111',
                        'fontWeight': 'bold',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_table={
                        'maxHeight': '500',
                        'overflowY': 'scroll'
                    }
                    # style_cell_conditional=[
                    # {
                    #     'backgroundColor': '#111111',
                    #     'if': {'column_id': c},
                    #     'textAlign': 'left',
                    #     'color': '#7FDBFF'
                    # } for c in positive_features_df.columns
                    # ]
                )
            ], 
                style = {'display':'inline-block', 'width': '50%'}
            ),
            html.Div([
                html.H2(
                    children='Sample Category Reviews',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                }),

                dash_table.DataTable(
                    data=categoryFeatures_df.to_dict('rows'),
                    columns=[{'id': c, 'name': c} for c in categoryFeatures_df.columns],
                    style_as_list_view=True,
                    # n_fixed_columns=3,
                    style_cell={
                        'padding': '5px',
                        'backgroundColor': '#111111',
                        'textAlign': 'left',
                        'color': '#7FDBFF',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'backgroundColor': '#111111',
                        'fontWeight': 'bold',
                        'color': '#7FDBFF',
                        'maxWidth': '180px'
                    },
                    style_table={
                        'maxHeight': '500',
                        'overflowY': 'scroll'
                    }
                    # style_cell_conditional=[
                    # {
                    #     'backgroundColor': '#111111',
                    #     'if': {'column_id': c},
                    #     'textAlign': 'left',
                    #     'color': '#7FDBFF'
                    # } for c in positive_features_df.columns
                    # ]
                )
            ]),
            # html.Div([
            #     html.H2(
            #         children = 'Select Your Brand: ',
            #         style={
            #             'textAlign': 'center',
            #             'color': colors['text']
            #         }
            #     ),
            #     dcc.Dropdown(
            #         id='brand_dropdown',
            #         options=[{'label':brand, 'value':brand} for brand in brands],
            #         value=brands[0]
            #     ),
            #     html.Div(id='brand_table')
            # ])
        ])

# @app.callback(Output('brand_table', 'children'),
#             [Input('brand_dropdown', 'value')],
# )

# def getBrandTable(input_value):
#     return html.Div([
#                 html.H2(
#                     children='Positive Reviews in selected brand',
#                     style={
#                         'textAlign': 'center',
#                         'color': colors['text']
#                 }),

#                 dash_table.DataTable(
#                     data=allBrands[input_value]['positive'].to_dict('rows'),
#                     columns=[{'id': c, 'name': c} for c in allBrands[input_value][positive].columns],
#                     style_as_list_view=True,
#                     # n_fixed_columns=3,
#                     style_cell={
#                         'padding': '5px',
#                         'backgroundColor': '#111111',
#                         'textAlign': 'left',
#                         'color': '#7FDBFF',
#                         'maxWidth': '180px'
#                     },
#                     style_header={
#                         'backgroundColor': '#111111',
#                         'fontWeight': 'bold',
#                         'color': '#7FDBFF',
#                         'maxWidth': '180px'
#                     },
#                     style_table={
#                         'maxHeight': '500',
#                         'overflowY': 'scroll'
#                     }
#                     # style_cell_conditional=[
#                     # {
#                     #     'backgroundColor': '#111111',
#                     #     'if': {'column_id': c},
#                     #     'textAlign': 'left',
#                     #     'color': '#7FDBFF'
#                     # } for c in positive_features_df.columns
#                     # ]
#                 )
#             ])

def getCategoryFeatures(numSampleReviews):
    from gensim import corpora

    import sklearn
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split,ShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import model_selection, naive_bayes, svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    import numpy as np
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import svm
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    import pickle
    import pandas as pd
    import re
    import statistics
    import random

    #Load the saved feature extraction 
    classifier_saved = open("model_classification/FeatureExtraction.pickle", "rb") #binary read
    classifier = pickle.load(classifier_saved)
    classifier_saved.close()

    # Extract the key features and put into dataframe

    list_1 = classifier.most_informative_features(numSampleReviews)
    df_important_features = pd.DataFrame(columns=['Feature','Cat1_Cat0','Ratio_1'])

    for (fname, fval) in list_1:
        cpdist = classifier._feature_probdist
        
        def labelprob(l):
            return cpdist[l, fname].prob(fval)

        labels = sorted(
            [l for l in classifier._labels if fval in cpdist[l, fname].samples()],
            key=labelprob
        )
        
        if len(labels) == 1:
            continue
        l0 = labels[0]
        l1 = labels[-1]
        if cpdist[l0, fname].prob(fval) == 0:
            ratio = 'INF'
        else:
            ratio = round(cpdist[l1, fname].prob(fval) / cpdist[l0, fname].prob(fval), 1)
            fname = fname.replace('contains(','')
            fname = fname.replace(')','')        
            df_important_features.loc[len(df_important_features)] = [fname, l1+" : "+l0, 
                                                    str(ratio)+" : 1.0"]
    print('category features: ')
    print(df_important_features)
    return df_important_features

def runModelSentiment(input_value):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    import pickle
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    import numpy as np
    from sentimentAnalysisUtil import stemmed_words,removeStopwords

    #filename = 'model_sentiment/logistic_regression_model.pk'
    #filename = 'model_sentiment/nb_model.pk
    filename = 'model_sentiment/svm_model.pk'
    tfidf = pickle.load(open('model_sentiment/tfidf_trans.pk','rb'))
    count = pickle.load(open('model_sentiment/count_vert.pk','rb'))
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))

    prediction = model.predict(tfidf.transform(count.transform([input_value])))

    # print('predicting sentiment')
    if (prediction[0] == 0):
        return 'negative'
    return 'positive'


def runModelCategoryClassification(input_value):
    import sklearn
    from sklearn.model_selection import train_test_split,ShuffleSplit
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import model_selection, naive_bayes, svm

    import numpy as np
    from sklearn import svm
    import pickle
    import pandas as pd

    # Load the classifier
    classifier_saved = open("model_classification/CategoryClassifier.pickle", "rb")
    classifier = pickle.load(classifier_saved)
    classifier_saved.close()

    #Load the saved classifier 
    classifier_saved = open("model_classification/TFIDF_Reviews_Category.pickle", "rb") #binary read
    TFIDF_vect = pickle.load(classifier_saved)
    classifier_saved.close()

    #process input
    review_holder = np.array([input_value])
    review_holder_1 = pd.Series(review_holder)

    # Transform reviews into TFIDF
    test_TFIDF = TFIDF_vect.transform(review_holder_1)
    prediction = classifier.predict(test_TFIDF)

    # print('predicting category')
    if prediction[0] == 0:
        return "cameras"
    elif prediction[0] == 1:
        return "laptops"
    else:
        return "mobile phone"

    # import sklearn
    # from sklearn.model_selection import train_test_split,ShuffleSplit
    # from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn import model_selection, naive_bayes, svm

    # import numpy as np
    # from sklearn import svm
    # import pickle
    # import pandas as pd

    # # Load the classifier
    # classifier_saved = open("model_classification/CategoryClassifier.pickle", "rb")
    # classifier = pickle.load(classifier_saved)
    # classifier_saved.close()

    # #Load the saved classifier 
    # classifier_saved = open("model_classification/TFIDF_Reviews_Category.pickle", "rb") #binary read
    # TFIDF_vect = pickle.load(classifier_saved)
    # classifier_saved.close()

    # #process input
    # review_holder = np.array([input_value])
    # review_holder_1 = pd.Series(review_holder)

    # # Transform reviews into TFIDF
    # review_TFIDF = TFIDF_vect.transform(review_holder_1)
    # prediction = classifier.predict(review_TFIDF)

    # print('predicting category')
    # if prediction[0] == 0:
    #     return 'cameras'
    # elif prediction[0] == 1:
    #     return 'laptops'
    # else:
    #     return 'mobile phone'




if __name__ == '__main__':
    app.run_server(debug=True)