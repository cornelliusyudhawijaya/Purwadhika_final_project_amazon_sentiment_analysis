import pandas as pd
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.plotly as py
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import nltk
from src.SupportFunction import show_word, show_word_tfidf, stops, meaningful_word_specific

#Data Amazon

dfAmazon = pickle.load(open('amazon_5_percent_complete.sav', 'rb'))
dfAmazonPlot = dfAmazon

#Model
model_title = pickle.load(open('model_predict_amazon_title.sav', 'rb'))
model_review = pickle.load(open('model_predict_amazon_review.sav', 'rb'))
model_combination = pickle.load(open('model_predict_amazon_combination.sav', 'rb'))

#Generate Table

def generate_table(dataframe, pagesize = 10):
    return dt.DataTable(
                id='table-multicol-sorting',
                columns=[{"name": i, "id": i} for i in list(dataframe.columns) ],
                pagination_settings={'current_page': 0,'page_size': pagesize},
                pagination_mode='be',
                style_table={'overflowX': 'scroll'},
                style_cell={'minWidth': '0px', 'maxWidth': '180px',
                            'whiteSpace': 'no-wrap',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            },
                css=[{'selector': '.dash-cell div.dash-cell-value',
                      'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                    }],
                sorting='be',
                sorting_type='multi',
                sorting_settings=[]
                )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
server = app.server



app.title = 'Dashboard Amazon Sentiment'

app.layout = html.Div([ 
    html.H1('Dashboard Amazon Sentiment Prediction'),
    html.H3('''
        Created by: Cornellius Yudha Wijaya
        ''')
    ,
    dcc.Tabs(id="tabs", value='tab-all', children=[ #Table
        dcc.Tab(label='Data amazon', value='tab-1', children=[
          html.Div([  
            html.Div([
                    html.P('Sentiment: '),
                    dcc.Dropdown(
                        id='filtersentiment',
                        options=[i for i in [{ 'label': 'All', 'value': '' },
                                            { 'label': 'Positive', 'value': 'Positive' },
                                            { 'label': 'Neutral', 'value': 'Neutral' },
                                            { 'label': 'Negative', 'value': 'Negative' }
                                            ]],
                        value=''
                    )
                ], className='col-4'),
            html.Div([
                    html.P('Rating: '),
                    dcc.Dropdown(
                        id='filterrating',
                        options=[i for i in [{ 'label': 'All', 'value': '' },
                                            { 'label': '1', 'value': 1 },
                                            { 'label': '2', 'value': 2 },
                                            { 'label': '3', 'value': 3 },
                                            { 'label': '4', 'value': 4 },
                                            { 'label': '5', 'value': 5 },
                                            ]],
                        value=''
                    )
                ], className='col-4')
          ], className = 'row'),
            html.Br(), 
            html.Div([
                html.Div([
                    html.Br(),
                    html.Button('Search', id='buttonsearch', style=dict(width='100%'))
                ], className='col-2')
            ], className='row'),
            html.Br(),html.Br(),html.Br(),
            html.Div([
                html.Div([
                    html.P('Max Rows : '),
                    dcc.Input(
                        id='filterrowstable',
                        type='number',
                        value=10,
                        style=dict(width='100%')
                    )
                ], className='col-1')
            ], className='row'),
            html.Center([
                html.H2('Data Amazon', className='title'),
                html.Div(id='tablediv', children = generate_table(dfAmazon))
            ])
        ]),
        dcc.Tab(label='EDA Plots', value='tab-2', children =[ #EDA Plot
        html.Div(children=[
                html.Div(children=[
                        html.P('Kind :'), 
                        dcc.Dropdown(
                        id='jenisplotcategory',
                        options=[{'label': i, 'value': i} for i in ['Hist','Bar','Box','Violin']],
                        value='Hist'
                        )
                    ], className = 'col-3'),
                    html.Div(children=[
                        html.P('Analysis :'), 
                        dcc.Dropdown(
                        id='xplotcategory',
                        options=[{'label': i, 'value': i} for i in ['Sentiment','Rating']],
                        value='Sentiment'
            )
                    ], className = 'col-3'),
                    html.Div(children=[
                        html.P('Number :'), 
                        dcc.Dropdown(
                        id='yplotcategory',
                        options=[{'label': i, 'value': i} for i in ['Title_length', 'Review_length', 'Title_sentence_wo_punc', 'Review_sentence_wo_punc']],
                        value='Title_length'
            )
                    ], className = 'col-3'),
                    html.Div([
                    html.P('Stats : '),
                    dcc.Dropdown(
                        id='statsplotcategory',
                        options=[i for i in [{ 'label': 'Mean', 'value': 'mean' },
                                            { 'label': 'Standard Deviation', 'value': 'std' },
                                            { 'label': 'Count', 'value': 'count' },
                                            { 'label': 'Min', 'value': 'min' },
                                            { 'label': 'Max', 'value': 'max' },
                                            { 'label': '25th Percentiles', 'value': '25%' },
                                            { 'label': 'Median', 'value': '50%' },
                                            { 'label': '75th Percentiles', 'value': '75%' }]],
                        value='mean',
                        disabled=False
                    )
                ], className='col-3')
                ], className = 'row'),
        html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
            dcc.Graph(
                id='categorygraph'    
            ) 
        ]),
        dcc.Tab(label='Word plots', value ='tab-3', children = [ #Word Generator Plot
            html.Div(children = [
                html.Div(children = [
                    html.P('Sentiment: '),
                    dcc.Dropdown(
                        id='sentimentword',
                        options=[i for i in [{ 'label': 'All', 'value': '' },
                                            { 'label': 'Positive', 'value': 'Positive' },
                                            { 'label': 'Neutral', 'value': 'Neutral' },
                                            { 'label': 'Negative', 'value': 'Negative' }
                                            ]],
                        value=''
                    )
                ], className='col-3'),
                html.Div(children = [
                    html.P('Type: '),
                    dcc.Dropdown(
                        id = 'typeword',
                        options = [{'label' : 'Count', 'value': 'Count'},
                                   {'label': 'TF-IDF', 'value': 'TF-IDF'}],
                        value = 'Count'            
                    )
                ], className ='col-3'),
                html.Div(children = [
                    html.P('Direction: '),
                    dcc.Dropdown(
                        id = 'directionword',
                        options = [{'label' : 'Highest', 'value': 'Highest'},
                                   {'label': 'Lowest', 'value': 'Lowest'}],
                        value = 'Highest'
                    )
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Category: '),
                    dcc.Dropdown(
                        id = 'categoryword',
                        options = [{'label' : 'Title', 'value': 'Title_meaningful'},
                                   {'label': 'Review', 'value': 'Review_meaningful'}],
                        value = 'Title_meaningful'
                    )
                ], className = 'col-3')
                ], className = 'row'),
                html.Br(), html.Br(), 
            html.Div([
                html.Div(children = [
                    html.P('Gram: '),
                    dcc.Dropdown(
                        id = 'gramword',
                        options = [{'label' : 'Unigram', 'value': 1},
                                   {'label': 'Bigram', 'value': 2}],
                        value = 1
                    )
                ], className = 'col-3'),
                html.Div(children=[
                    html.P('Number: '),
                    dcc.Input(
                        id='numberword',
                        type='number',
                        value=20,
                        style=dict(width='100%')
                    )
                ], className='col-1'),
                html.Div([
                    html.Br(),
                    html.Button('Search', id='buttonword', style=dict(width='100%'))
                ], className='col-2')
            ], className='row'),
            html.Br(),
            dcc.Graph(
                id='wordgraph'    
            ) 
        ]),
        dcc.Tab(label = 'Sentiment Prediction Model', value = 'tab=4', children=[
            html.Div([
                html.Div([
                    html.Center([
                        html.H2('Amazon Sentiment Prediction Model', className='title')])
                     ]),
                    html.Br(),
                html.Div([        
                    html.P('Insert Title: '),
                    dcc.Input(
                        id ='titlepredict',
                        type='text',
                        placeholder='Enter the title..',
                        value = '',
                        style=dict(width='100%')
                    )], className = 'col-12 row'),
                    html.Br(),
                    html.Div([
                        html.P('Insert Review: '),
                    dcc.Textarea(
                    id = 'reviewpredict',
                    placeholder='Enter the review..',
                    value='',
                    style={'width': '100%'} 
                        )], className = 'col-12 row'
                    ),
                    html.Br(),
                    html.Div([
                        html.Button('Predict', id='buttonpredict', style=dict(width='100%'))
                    ], className='col-2 row'),
                    html.Div([
                        html.Center([],id ='outputpredict')
                    ])]
                )]
                )
                ],
    style = {
                'fontFamily': 'system-ui'
    }, content_style = {
        'fontFamily': 'Arial',
        'borderBottom': '1px solid #d6d6d6',
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'padding': '44px'
    }) 
], style ={
        'maxWidth': '1200px',
        'margin': '0 auto'
    })

#Table Callback

@app.callback(
    Output('table-multicol-sorting', "data"),
    [Input('table-multicol-sorting', "pagination_settings"),
     Input('table-multicol-sorting', "sorting_settings")])
def callback_sorting_table(pagination_settings, sorting_settings):
    if len(sorting_settings):
        dff = dfAmazon.sort_values(
            [col['column_id'] for col in sorting_settings],
            ascending=[
                col['direction'] == 'asc'
                for col in sorting_settings
            ],
            inplace=False
        )
    else:
        # No sort is applied
        dff = dfAmazon

    return dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ].to_dict('rows')
    

@app.callback(
    Output(component_id='tablediv', component_property='children'),
    [Input('buttonsearch', 'n_clicks'),
    Input('filterrowstable', 'value')],
    [State('filtersentiment', 'value'),
    State('filterrating', 'value')
    ]
)
def update_table(n_clicks,maxrows, sentiment, rating):
    global dfAmazon
    dfAmazon = pickle.load(open('amazon_5_percent_complete.sav', 'rb'))
    
    if (sentiment != '') & (rating != ''):
        dfAmazon = dfAmazon[(dfAmazon['Sentiment'] == sentiment) & (dfAmazon['Rating'] == rating)]    
    elif (sentiment != ''):
        dfAmazon = dfAmazon[dfAmazon['Sentiment'] == sentiment]
    elif (rating != ''): 
        dfAmazon = dfAmazon[dfAmazon['Rating'] == rating]

    return generate_table(dfAmazon, pagesize=maxrows)  

#callback EDA plots

listGoFunc ={
    'Bar': go.Bar,
    'Box' : go.Box,
    'Violin' : go.Violin
}

def generateValuePlot(xplot,yplot,stats = 'mean'):
    return{
            'x': {
            'Bar': dfAmazonPlot[xplot].unique(),
            'Box': dfAmazonPlot[xplot],
            'Violin': dfAmazonPlot[xplot]
                },
            'y': {
            'Bar': dfAmazonPlot.groupby(xplot)[yplot].describe()[stats],    
            'Box': dfAmazonPlot[yplot],
            'Violin': dfAmazonPlot[yplot]
            }
            }

@app.callback(
    Output(component_id= 'categorygraph', component_property = 'figure'),
    [Input(component_id='jenisplotcategory', component_property = 'value'),
    Input(component_id='xplotcategory', component_property = 'value'),
    Input(component_id='yplotcategory', component_property = 'value'),
    Input(component_id='statsplotcategory', component_property='value')]
)
def callback_update_category_graph(jenisplot, xplot, yplot, stats):
    dfAmazonPlot = pickle.load(open('amazon_5_percent_complete.sav', 'rb'))
    if jenisplot != 'Hist':
        newDict = dict(
            layout = go.Layout(
                title = '{} Plot Amazon'.format(jenisplot),
                xaxis = {"title": f'{xplot}'},
                yaxis = {'title': f'{yplot}'},
                boxmode = 'group',
                violinmode ='group'
            ), 
            data = [ 
                listGoFunc[jenisplot](
                x = generateValuePlot(xplot,yplot)['x'][jenisplot],
                y = generateValuePlot(xplot,yplot, stats)['y'][jenisplot]
                )]
        )
        return newDict 
    else:
        if xplot == 'Sentiment':
            color = ['red', 'blue', 'green']
            newDict = dict(
                data=[
                    go.Histogram(
                        x=dfAmazonPlot[dfAmazonPlot['Sentiment'] == i][yplot],
                        name= i,
                        marker=dict(
                            color=j, 
                            opacity = 0.7
                        )
                    )
                for i,j in zip(dfAmazonPlot['Sentiment'].unique(), color)],
                layout=go.Layout(
                    title='Histogram {} Amazon'.format(yplot),
                    xaxis=dict(title=yplot),
                    yaxis=dict(title='Count'),
                    height=400, width=1000
                )
            )   
            return newDict
        elif xplot == 'Rating':
            color = ['red', 'green', 'blue', 'yellow', 'brown']
            newDict = dict(
                data=[
                    go.Histogram(
                        x=dfAmazonPlot[dfAmazonPlot['Rating'] == i][yplot],
                        name= str(i),
                        marker=dict(
                            color=j,
                            opacity = 0.7
                        )
                    )
                for i, j in zip(dfAmazonPlot['Rating'].unique(), color)],
                layout=go.Layout(
                    title='Histogram {} Amazon'.format(yplot),
                    xaxis=dict(title=yplot),
                    yaxis=dict(title='Count'),
                    height=400, width=1000
                )
            )   
            return newDict

@app.callback(
    Output(component_id='statsplotcategory', component_property='disabled'),
    [Input(component_id='jenisplotcategory', component_property='value')]
)
def update_disabled_stats(jenisplot):
    if(jenisplot == 'Bar') :
        return False
    return True

# Callback word plot
@app.callback(
    Output(component_id='wordgraph', component_property='figure'),
    [Input('buttonword', 'n_clicks')],
    [State('sentimentword', 'value'),
    State('typeword', 'value'),
    State('directionword','value'),
    State('numberword','value'),
    State('categoryword', 'value'),
    State('gramword', 'value')
    ]
)
def word_graph(n_clicks, sentiment, type, direction, number, category,gram):
    dfAmazonPlot =  pickle.load(open('amazon_5_percent_complete.sav', 'rb'))
    if type == 'Count':
        data = show_word(sentiment, category, gram, direction, number, df = dfAmazonPlot)
    elif type == 'TF-IDF':
        data = show_word_tfidf(gram, sentiment, category, direction, number, df = dfAmazonPlot)
    figure_word = dict(
        data = [go.Bar(
        y=data['y'],
        x=data['x']
    )],

    layout = go.Layout(
        title="{} {} {}-gram {} words of the {} {}".format(number, direction, gram, type, sentiment, category),
        yaxis=dict(
            ticklen=8
        )
    ))
    return figure_word    

# Callback Prediciton Model

@app.callback(
    Output(component_id='outputpredict', component_property='children'),
    [Input('buttonpredict', 'n_clicks')],
    [State('titlepredict', 'value'),
     State('reviewpredict', 'value')]
)
def amazon_predict(n_clicks, title_text, review_text):
    predict_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    title_text = str(title_text)
    review_text = str(review_text)
    if (title_text == '') and (review_text == ''):
        return html.H3('Please fill either the title or the review text box')
    elif (title_text != '') and (review_text == ''):
        title = model_title.predict([title_text])[0]
        title_prob = model_title.predict_proba([title_text])[0, predict_dict[title]]
        return [html.H3('This title is classified as {} Sentiment text with probability {}%'.format(title, round(title_prob*100,2)))]    
    elif (title_text == '') and (review_text != ''):
        review = model_review.predict([review_text])[0]
        review_prob = model_review.predict_proba([review_text])[0, predict_dict[review]]
        return [html.H3('This review is classified as {} Sentiment text with probability {}%'.format(review, round(review_prob*100,2)))]
    elif (title_text != '') and (review_text != ''): 
        title = model_title.predict([title_text])[0]
        title_prob = model_title.predict_proba([title_text])[0, predict_dict[title]]
        review = model_review.predict([review_text])[0]
        review_prob = model_review.predict_proba([review_text])[0, predict_dict[review]]
        combination_text = ' '.join([title_text, review_text])
        combination = model_combination.predict([combination_text])[0]
        combination_prob = model_combination.predict_proba([combination_text])[0, predict_dict[combination]]
        return [html.H3('The Title is classified as {} Sentiment text with probability {}%'.format(title, round(title_prob*100,2))),
                html.H3('The Review is classified as {} Sentiment text with probability {}%'.format(review, round(review_prob*100,2))),
                html.H3('Combination of the Title and The Review is classified as {} Sentiment text with probability {}%'.format(combination, round(combination_prob*100)))]
if __name__ == '__main__':
    app.run_server(debug=True)