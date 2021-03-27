# import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output,State

from app import app, server



layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Telco Customer Churn", active=True,href="/apps/telco_customer_churn")),
        # dbc.NavItem(dbc.NavLink("Explore", active=True,href="/apps/explore")),
        # dbc.NavItem(dbc.NavLink("Clean", active=True,href="#")),
        # dbc.NavItem(dbc.NavLink("Analyse", active=True,href="#")),
        # dbc.NavItem(dbc.NavLink("Model", active=True, href="#"))
    ], 
    brand="Qaml",
    brand_href="/apps/home",
    color="primary",
    dark=True,
    style={'margin-bottom': '2px'}

),#end navigation

	#body
	 html.Div(
    [

  

    
    #1.
        dbc.Row(
            [
                dbc.Col(html.Div([
                  # html.H6("Analyse your data"),
                  ] 
                	),
			style={
            'margin-top': '30px'
            },
                	md=0),
   #2.
                      dbc.Col(html.Div([
                    html.H6("Home") , 
                    ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=12),
   #3. doughnut_pie_chart_with_center
                       dbc.Col(html.Div(
              [
                # html.H6("Tweet Analysis") , 
                   ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=0),

            ]
        ),

# 4. 
        dbc.Row(
            [
                        dbc.Col(html.Div(
     
                  ),
                  md=4),

    #5. 
                   dbc.Col(html.Div(
     
                  ),
                  md=4),

    # 6
                         dbc.Col(html.Div( 

                  ),
                  md=4),
            ]
        ),
     

       
        # footer
 		dbc.Row(
            [
                dbc.Col(html.Div("@galaxydataanalytics "),
                	style={
            'margin-top': '2px',
            'text-align':'center',
            'backgroundColor': 'rgba(120,120,120,0.2)'
            },
                 md=12)
            ]
        ),
        #end footer
    ],
        style={
            'padding-left': '3px',
            'padding-right': '3px'
            },
)
	#end body

	],
	fluid=True
	)
