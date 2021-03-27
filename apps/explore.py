# import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_uploader as du
from dash.dependencies import Input, Output,State
import pandas as pd
import pathlib
import uuid

from app import app, server


PATH=pathlib.Path(__file__).parent
# TELCO_CHURN_FILE_UPLOADS_DATA_PATH=PATH.joinpath("../datasets/telco_churn_file_uploads").resolve()
# du.configure_upload(app, TELCO_CHURN_FILE_UPLOADS_DATA_PATH, use_upload_id=False)


layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Explore", active=True,href="/apps/explore")),
        dbc.NavItem(dbc.NavLink("Clean", active=True,href="#")),
        dbc.NavItem(dbc.NavLink("Analyse", active=True,href="#")),
        dbc.NavItem(dbc.NavLink("Model", active=True, href="#"))
    ], 
    brand="Qaml",
    brand_href="/apps/home",
    color="primary",
    dark=True,
    style={'margin-bottom': '2px'},


),#end navigation

	#body
	 html.Div(
    [

  

    
    #1.
        dbc.Row(
            [
                dbc.Col(html.Div([                  
                   du.Upload( id='upload-file',
                        max_file_size=2,  # 2 Mb max file size
                        filetypes=['csv'],
                        # upload_id=uuid.uuid1(),  # Unique session id
                        text='Drag and Drop a File Here to upload!',
                        text_completed='File Sucessfully Uploaded: ',
                        # cancel_button=True, 
                        # pause_button=True,
                           ),
                   # dbc.Button("Submit", color="primary", active=True, className="mr-1",id="load-data"),
                  ] 
                	),
			style={
            'margin-top': '30px'
            },
                	md=4),
   #2.
                      dbc.Col(html.Div([
                    html.Div(id='output-stats'),
                   
                    
                    ]
                  ),
      style={
            'margin-top': '30px',
            'font-size':'20px'
            },
                  md=4),
   #3. 
                       dbc.Col(html.Div(
              [
                # html.H6("Tweet Analysis") , 
                   ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),

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



# @app.callback(
#     Output('output-stats', 'children'),
#     [Input('upload-file', 'isCompleted')],
#     [State('upload-file', 'fileNames'),
#      State('upload-file', 'upload_id')
#      ],
# )

# def callback_on_completion(iscompleted, filenames, upload_id):
#   # data=[str(x) for x in filenames]
#   # return filenames
  
#   file=str(filenames).replace("['","").replace("']","")
#   df=pd.read_csv(TELCO_CHURN_FILE_UPLOADS_DATA_PATH.joinpath(file))
#   # print(df.head())
#   return "# Records : "+str(df.shape[0])+" | # Attributes : "+str(df.shape[1])

