### AWS Machine Learning Solutions Lab - Machine Learning Engineer Work Sample ####

### Description ###

A data scientist on your team has developed a model to predict stock market opening values (i.e. stock
price at market open) for various cryptocurrencies. Before this model can be deployed to production, the team
must first build a POC (proof of concept) deployment and test all the components. The team wants to test this 
model by submitting test data to a RESTful endpoint. 

As a machine learning engineer on the team, you are tasked with taking the modeling code (and resultant 
model objects) the team has produced and deploying it to an API endpoint such that new predictions 
can be generated. The pipeline you develop is meant to be a simple POC, with plans to expand the scale pending
successful modeling results. 

The team has developed a simple frontend to visualize the model predictions, and it is expected that your 
API will provide predictions back in the correct format, such that the frontend can consume them.

The team has sent their work over to you, in an attached zip file. In that file you will find the following 
items:

- `crypto_dirty.ipynb`: The data scientist's modeling code
- `crypto_train.csv`: Training data used to build the model
- `model_out/model.h5`: Pre-trained Keras model object
- `model_out/scaler.pkl`: Additional required modeling objects
- `simple_frontend.py`: The code for the frontend that will consume predictions

### Production Scaling ###

Your goal with this POC system is to produce a pipeline that generates predictions on testing data. However,
a production system typically includes many additional items. It is **not** required that you include these items 
in your delivered code, but it **is** required that you create a plan to address them. The team needs to know
how you intend to take your POC system and scale it. 

Your plan should address the following requirements:

- Each prediction must be auditable: must be able to assess the state of the data and ETL code that led to a prediction
- A pipeline for model retraining at an established frequency (i.e. weekly, monthly)
- Each model retrain must be auditable
- The team must be able to rollback a model if it proves problematic
- All training data should be sourced from your pipeline
- Predictions will occur daily at stock market close
- The team must be able to trigger new predictions at will
- The prediction pipeline must be able to automatically scale with increased data volume and/or increased user 
prediction requests

To help you plan for the production pipeline, the team has gathered some requirements from engineering:

- There will be up to 10,000 unique cryptocurrency symbols in the data
- Daily data uploads will contain only one day of data (i.e. one observation) per symbol
- The model will be retrained weekly
- Monthly data volume can be up to 1TB of data

### Instructions ###

Take the code in `crypto_dirty.ipynb` and deploy it to an accessible endpoint. To expedite the process, pre-trained
model objects are available to you - `model.h5` and `scaler.pkl`. You are free to re-train these objects, but that
isn't necessary for this work sample. 

As this initial deployment is a POC, it is not expected that you have built a particularly robust pipeline. However, 
the team will require that you develop a plan for how to scale what you create to a production-ready level.

Please provide the following items back to the team:

- `test_url`: an S3 URL that the team can upload a testing dataset to
- `pred_url`: an S3 URL of the testing set predictions
- A diagram of the POC deployment architecture (rough diagram is fine)
- A plan for scaling the POC deployment to production 
- All code used to achieve the POC architecture 

To test your solution, the team will upload a csv, formatted in the same way as `crypto_filt.csv`, to
the provided `test_url`. The visualization frontend will then be pointed to `pred_url` and it is expected
that the frontend will be able to seamlessly read in the predictions. 

**NOTE:** Accuracy of the model is not your concern, you are only required to build the pipeline to 
generate predictions.

### Requirements ###

The team has a couple of requirements for the production system that you should consider while building your
solution:

- Production system will be near-real-time, and require a fast response on uploaded test set
- The solution should be highly available and fault tolerant
- The production pipeline should be auditable
- Where it makes sense, cost should be considered

The plan for scaling should be AT MOST two pages long, minus any figures or tables. Please keep all figures, tables, 
etc. in an appendix, which has no length requirement.

**It is not required that you deploy your solution on the AWS cloud.** You must be able to clearly describe your solution
to the team, regardless of which cloud/tools you use.

### Evaluation ###

The team will be assessing your infrastructure along the following criteria:

- Cleanliness, efficiency, and interpretability of all code used
- Use of appropriate tools/services in architecture
- Efficiency of architecture
- Feasibility of scaling plan provided
- Where applicable, cost of POC (and proposed scaling plan) will be considered

The team also expects that the frontend will be able to properly visualize the predictions generated on
the testing dataset.

### Miscellaneous ###

The team has provided a list of things to keep in mind as you build:

- Data scientists aren't known for producing the cleanest or most efficient code
- Simpler is usually better
- Solutions can be iterated upon; be sure to put enough careful thought into your plan for scaling

`simple_frontend.py` can be launched via `bokeh serve --show simple_frontend.py`. When run locally, the code 
expects predictions to be stored in a foldered labeled `data`, in a file labeled `crypto_test_preds.csv`. 
The team's frontend will accept a URL in place of `crypto_test_preds.csv`, but you are free to use 
`simple_frontend.py` to help test and debug your solution locally.
