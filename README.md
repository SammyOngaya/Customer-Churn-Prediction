### <a href="https://qaml.herokuapp.com/apps/telco_customer_churn" target="blank">Telco Customer Churn</a>
This repo is based on the <a href="https://qaml.herokuapp.com/apps/telco_customer_churn" target="blank">Telco Customer Churn </a>use-case. In this use-case we use machine learning to predict wether a an existing customer will churn or not.
We begin by collecting and cleaning the data. We then explore the data to find the insights and the distribution of the data. We preprocess and perform feature engineering of the data.
Once we have the correct features we proceed to machine learning modeling where we split our data into training, validation and testing sets. We train various models and compare there performance. <br>

We perform hyperparameter tunning using Grid Search approach to optimize our model and find the optimal hyperparameters for each algorithm. After training the model with best hyerparameters we evaluate the models 
using various model evaluation techniques (Accuracy, Precision, Recall, F1-Score and UAC ROC). We save the best performing model based on the evaluation metric and use it for deployment.
In deployment we create a dash web app and deploy our model on heroku server. We create a user interface where the model can be used to predict with single or batch data upload and prediction.
The web app has three web interface; The data Exploration section, The Model training and Feature selection section and The Model Prediction and results presentation section.
The project is hosted here <a href="https://qaml.herokuapp.com/apps/telco_customer_churn" target="blank">Telco Customer Churn ML Project</a>
1. ![Telco Customer Churn Data Exploration](https://raw.githubusercontent.com/SammyOngaya/Customer-Churn-Prediction/master/Notebooks/Churn%20Models/Telco%20Customer%20Churn%20Data%20Exploration.PNG)

2. ![Telco Customer Churn Modeling](https://raw.githubusercontent.com/SammyOngaya/Customer-Churn-Prediction/master/Notebooks/Churn%20Models/Telco%20Customer%20Churn%20ML%20Modeling.PNG)


3. ![Telco Customer Churn Model Prediction and Results Presentation](https://github.com/SammyOngaya/Customer-Churn-Prediction/blob/master/Notebooks/Churn%20Models/Telco%20Customer%20Churn%20Model%20Prediction.PNG?raw=true)
