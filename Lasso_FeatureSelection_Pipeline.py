#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:48:29 2021

@author: marcinswierczewski
"""
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin



class LassoPipelineProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, savedfeatures_names = [], current_features = []):
        '''
        Constructor for Lasso pipeline class. 
        It has to inherit from BaseEstimator and TransformerMixin to 
        create bespoke estimator, for sklearn's pipeline.
        All estimators in sklearn have to be derived from BaseEstimator

        Parameters
        ----------
        savedfeatures_names : list, string.
            This list can be established by the LassoFeatureSelectionPipeline
            class, after chosing appropariate penalty value.
            Just save the list you get from LassoFeatureSelectionPipeline.model(penalty)
            and pass it as the argument here.
            
        current_features : list, string
            Name of all columns in the dataset.

        Returns
        -------
        Used to be fitted into the sklearn pipeline.

        '''
        self.savedfeatures_names = savedfeatures_names
        self.current_features = current_features
        
    def fit (self, X, y=None):
        # Fits the model with all saved names
        self.all_savedfeatures_names = self.savedfeatures_names
        return self
    
    def transform (self, X):
        # copying only names that for given penalty were not shrinked by lasso 
        X = pd.DataFrame(data=X, columns = self.current_features)
        output = X.copy()
        output = output[self.savedfeatures_names]
        return output
    
    def get_support(self):
        # chosen features getter
        return self.savedfeatures_names
        
        

class LassoFeatureSelectionPipeline:
    def __init__(self, data, test,currentpredyear, *categorical):
        '''
        Constructor for lasso feature selection. 
        When calling object.model() it calls combination of penalties
        and displays accuracy metrics and number of features.
        When topvalues greater than 0, it returns chosen features. 

        Parameters
        ----------
        data : DataFrame.
            Training data.
        test : DataFrame
            Test data.
        currentpredyear : Int
            Currently depreciated, to be removed in v0.2
        *categorical : list,str
            Exclude passed categorical columns.

        Returns
        -------
        None.

        '''
        self.data = data
        self.test = test
        self.categorical = categorical
        self.currentpredyear = currentpredyear
        print(categorical)
        
    def model(self, c_runs = 0):
        '''
        This function runs lasso model utilising it's sparsity. Lasso would shrink 
        and remove the coefficients and push weights to 0, therefore eliminating
        some variables.
        
        In default setting (model()), the function will print a table with
        penalty, number of features and accuracy (AUC) metric. 

        ----------
        c_runs : Int
            The default is 0. That will make many combinations of penalties 
            being applied in the lasso model.
            If this is set to a value greater than 0, model will only run once,
            with stated penalty. 

        Returns
        -------
        If c_runs == 0: returned is a table with C, AUC and no of features.
        If c_runs > 0: returned is a list of names of chosen variables (str)
        '''
        data = self.data.loc[:, ~self.data.columns.isin(self.categorical)]
        test = self.test.loc[:, ~self.test.columns.isin(self.categorical)]
        
        y_train  = data[['Class']]
        y_test    = test[['Class']]
        new_data = data.drop(labels=['Class'], axis=1)
        new_data2  = test.drop(labels=['Class'], axis=1)
        initcolnames = new_data.columns

        transform = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # fill Missing value
                ('scaler', StandardScaler()),
                # think about transforming data 
            ])
        X_train = transform.fit_transform(new_data)
        # if c_runs is chosen lasso does not test and only runs picked c value
        # and returns the feature names
        if c_runs == 0:
            cs = [0.001, 0.005, 0.008, 0.01,0.02,0.05,0.07,0.08,0.1,0.15,0.20,0.30,0.50]
        else:
            cs = [c_runs]
        AUC = []
        features_no = []
        features_return = []
        print("Running feature engineering....")
        for C_ in cs:
            select = LogisticRegression(penalty='l1',C=C_,random_state=7,solver='liblinear')
            select.fit(X_train, y_train)
            
            df_picked = pd.DataFrame(zip(select.coef_[0],initcolnames) , 
                                     columns = ['Coef','Name'])
            df_picked = df_picked[df_picked['Coef'] != 0]
            
            # df_picked = pd.DataFrame(X_train ,columns = initcolnames)
            # selected_feat = df_picked.columns[(select.get_support())]
            selected_feat = list(df_picked['Name'].values)
            features_no.append(len(selected_feat))
            if c_runs != 0:
                features_return = selected_feat
                
            # that was only used for stock project, bc of different dates 
            # yearpred = self.currentpredyear + 1
            # yearpred = str(yearpred)
            # yearpredtest = self.currentpredyear + 2
            # yearpredtest = str(yearpredtest)
            # selected_feat2 = [l.replace(yearpred, yearpredtest) for l in selected_feat]
            
            selected_feat2 = selected_feat
            
            X_testtemp = new_data2.copy()
            X_testtemp = X_testtemp[selected_feat2]
            X_test = transform.fit_transform(X_testtemp)
            select.fit(X_test,y_test)
            aucscore = np.mean(cross_val_score(select , X_test, y_test,
                                    cv=7, scoring='roc_auc'))
            AUC.append(aucscore)
            
            if C_ == 0.15:
                print("50%")
            
        datamerge = []

        for t in zip(cs,AUC,features_no):
            datamerge.append(t)

        returndata = pd.DataFrame(datamerge,columns=['C','AUC',
                                                     'Number of features'])
        if c_runs == 0:
            return returndata
        else:
            return features_return

        
            



        
        
        
        
        
        
        
        
        
        
        
    
