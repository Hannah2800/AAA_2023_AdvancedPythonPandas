# methods defined for Task4 - Predictive

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
import math
import pandas as pd

#################################################################################################################################

# method for creating the categorical und numerical features:

def cat_and_num(spatial_feature):
    
    categoric = ['start_stamp', spatial_feature,'dayOfWeek','start_time_month','start_time_day','start_time_week',
             'isHoliday','description','isRushhour', 'season']
    numeric = ['temperature_celsius','wind_speed','wind_direction','humidity','pressure']
    
    return categoric, numeric

#################################################################################################################################
     
# method for transforming the days and season variable to integer:

def dayandseas_to_int(df):
    
    df['dayOfWeek'] = df['dayOfWeek'].replace(['Monday', 'Tuesday','Wednesday','Thursday',
                                                                  'Friday','Saturday','Sunday'],[0,1,2,3,4,5,6])
    df['season'] = df['season'].replace(['summer', 'winter','spring','autumn'],[0,1,2,3])

    return df

#################################################################################################################################

# method for creating a pipeline:

# function for normalize numeric and encode categorical features and for create pipeline
def pipeline_for_prediction(categoric, numeric, model):
    
    numeric_transformer = Pipeline(steps=[("standard_scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical scaler", numeric_transformer, numeric),
            ("one hot encoder", categorical_transformer, categoric),
        ]
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )
    return pipeline

#################################################################################################################################

# method for creating scores based on a specific dataframe and feature (depended variable):

# function for getting different scores for a model
def get_prediction_scores(y_true, y_predicted, df, depend_feat):
    print("MODEL SCORES:")
    print(f"MAE: {metrics.mean_absolute_error(y_true, y_predicted): .3f}")
    print(f"MSE: {metrics.mean_squared_error(y_true, y_predicted): .3f}")
    print(f"RMSE: {math.sqrt(metrics.mean_squared_error(y_true, y_predicted)): .3f}")
    print(f"Accuracy:", round((1-(metrics.mean_absolute_error(y_true, y_predicted)/df[depend_feat].mean()))*100,2),
          "%")
    print(f"R2: {100 * metrics.r2_score(y_true, y_predicted): .3f} %")
    print(f"Max Residual Error: {metrics.max_error(y_true, y_predicted): .3f}")

#################################################################################################################################

# method whole prediction:

# function for creating pipeline and fitting model (created by the pipeline), predict and printing scores
def pipeline_fit_predict(reg, categoric, numeric, x_train, y_train, x_val, y_val, df, depend_feat):
    pipeline = pipeline_for_prediction(categoric, numeric, reg)
    pipeline.fit(x_train, y_train)
    y_predict = pipeline.predict(x_val)
    get_prediction_scores(y_val, y_predict, df, depend_feat)

#################################################################################################################################
    
# method for performing RandomizedSearchCV for hyperparameter tuning:

# function for finding the best hyperparameter by using RandomizedSearchCV and RepeatedStratifiedKFold
"""parameter:
   - pipeline: used pipeline for grid search (the pipeline contains the model)
   - x_val: data set (features) used for grid search
   - y_val: data set (target value) used for grid search
   - model_par: parameters for which the grid search is done
   - score: used score measure 
   - n_iter: how often grid search will be done
   - n_repeats: how often the data set is randomly splitted (by using the same random hyperparameter) in n_splits
   - n_splits: number of splits in RepeatedStratifiedKFold
   - verbose: getting information during the grid search
"""

def find_best_hyperparameters(pipeline, x_val, y_val, model_par, score, n_iter = 50,  
                                   n_repeats=3, n_splits=5, n_jobs=1, verbose=True): #  n_repeats=3
    
    print(f"Running grid search for the model based on {score}")
    grid_pipeline = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=model_par,
        n_jobs=n_jobs,
        n_iter=n_iter,
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42),
        scoring=score,
        random_state=42,
        verbose=verbose,
    )
    grid_pipeline.fit(x_val, y_val)
    print(f"Best {score} Score was: {grid_pipeline.best_score_}")
    print("The best hyper parameters for the model are:")
    print(grid_pipeline.best_params_)

################################################################################################################################# 

# method for performing a split for train, test and validation set:

# default: 30% test, 20% validation, 30% training (and 70% train_val)
def train_val_test(df, testsize=0.3, valsize=0.2):
    
    #split the data set in (1-testsize)% training set and testsize% testing set
    x_train, x_test, y_train, y_test = train_test_split(df.drop('numOfTaxis_area', axis=1)
                                                    , df['numOfTaxis_area'], 
                                                    test_size=testsize,random_state=42)

    # save the combination of training and validation set in extra variables
    x_train_val = x_train
    y_train_val = y_train

    #split the training data set to achieve a 50-20-30 split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(valsize/(1-testsize)), 
                                                      random_state = 42)
    
    return x_train_val, y_train_val, x_train, y_train, x_val, y_val, x_test, y_test 

#################################################################################################################################

# method for sampling data, when its already splittet:

# x_set and y_set are the already splitted datasets, which should be sampled
def sampling_already_splitted(x_set, y_set, dependend, num_samples, random_state=42):
    
    y_val_df = y_set.to_frame() # y_set is series, so turn it to frame
    df = pd.concat([x_set, y_val_df], axis=1) # concat to whole df
    df_sample = df.sample(num_samples, random_state=random_state) # sample the df
    y_val = df_sample[dependend] # again split in y and x set
    x_val = df_sample.drop(columns = dependend)
    
    return x_val, y_val









