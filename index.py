import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output,State
from app import app, server

from apps import home,explore,telco_customer_churn

app.layout = html.Div([
  dcc.Location(id='url', refresh=False),
   html.Div(id='page-content'),
])

# links method
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/explore':
        return explore.layout
    elif pathname == '/apps/telco_customer_churn':
        return telco_customer_churn.layout
    # elif pathname == '/apps/home':
    #     return home.layout
    # elif pathname == '/apps/tweet_analysis':
    #     return tweet_analysis.layout
    # elif pathname == '/apps/topic_modeling':
    #     return topic_modeling.layout
    else:
        return home.layout


if __name__ == '__main__':
    app.run_server(debug=False)