#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:06:05 2021

@author: marcinswierczewski
"""
import os
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

class RandomForestProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, savedfeatures_names = [], current_features = []):
        '''
        Constructor for Random Forest pipeline class. 
        It has to inherit from BaseEstimator and TransformerMixin to 
        create bespoke estimator, for sklearn's pipeline.
        All estimators in sklearn have to be derived from BaseEstimator

        Parameters
        ----------
        savedfeatures_names : list, string.
            This list can be established by the RandomForestFeatureSelectionPipeline
            class, after chosing appropariate number of features, based on 
            AUC displayed by that class.
            Just save the list you get from 
            RandomForestFeatureSelectionPipeline.model(topvalues = n)
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
        # copying only names that for given number of estimators, were
        # picked based on Random Forest results
        X = pd.DataFrame(data=X, columns = self.current_features)
        output = X.copy()
        output = output[self.savedfeatures_names]
        return output
    
    def get_support(self):
        # chosen feature's getter
        return self.savedfeatures_names
        
        

class RandomForestFeatureSelectionPipeline:
    def __init__(self, data, test,n_estimators = 200, max_depth = None):
        '''
        Constructor for Random Forest feature selection. 
        When calling object.model() it calls combination of number of features
        and displays accuracy metrics and number of features.
        When topvalues greater than 0, it returns chosen features. 

        Parameters
        ----------
        data : DataFrame.
            Training data.
        test : DataFrame
            Test data.
        n_estimators : Int
            Number of estimators passed to sklearn's RandomForestClassifier.
            For more information, please see sklearn documentation.
        max_depth : int
            Max depth of the tree, passed to sklearn's RandomForestClassifier.
            For more information, please see sklearn documentation.
        Returns
        -------
        None.

        '''
        self.data = data
        self.test = test
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def model(self, topvalues = 0):
        '''
        This function runs Random Forest Classifier model .
        
        In default setting (model()), the function will print a table with
        number of features and accuracy (AUC) metric. 

        ----------
        topvalues : Int
            The default is 0. That will make many combinations of number
            of features and display AUC of each.
            If this is set to a value greater than 0, model will only run once,
            returning the chosen features (number of list will correspond 
                                           to the passed number). 

        Returns
        -------
        If topvalues == 0: returned is a table with number of features and AUC score.
        If topvalues > 0: returned is a list of names of x top variables (str)
        '''
        data = self.data
        test = self.test
        
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
        if topvalues == 0:
            x = int(data.shape[1] / 10)
            val = [int(x), int(x*2), int(x*5), int(x*7), int(x*10)]
        else:
            val = [topvalues]
        AUC = []
        features_no = []
        features_return = []
        print("Running feature engineering....")
        # creating model object
        rfmodel = RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,
                                         random_state=66, 
                                              n_jobs=int(os.cpu_count() / 3))
        rfmodel.fit(X_train,y_train)
        important = rfmodel.feature_importances_
        
        # creating df for displaying the picked scenario
        df_picked = pd.DataFrame(sorted(zip(important,initcolnames),reverse=True))
        df_picked.columns = ['RF_importance','Feature_name']
        df_picked.sort_values(by='RF_importance',ascending=False,inplace=True)
        print("Performing accuracy tests")
        for i in val:
            df_picked2 = df_picked.copy()
            df_picked2 = df_picked2[:i]

            selected_feat = list(df_picked2['Feature_name'].values)
            if topvalues != 0:
                features_return = selected_feat

            X_testtemp = new_data2.copy()
            X_testtemp = X_testtemp[selected_feat]
            X_test = transform.fit_transform(X_testtemp)
            rfmodel.fit(X_test,y_test)
            aucscore = np.mean(cross_val_score(rfmodel , X_test, y_test,
                                    cv=7, scoring='roc_auc'))
            AUC.append(aucscore)
            
        datamerge = []

        for t in zip(val,AUC):
            datamerge.append(t)

        returndata = pd.DataFrame(datamerge,columns=['Number of features','AUC'])
        print("Feature engineering completed.")
        if topvalues == 0:
            return returndata
        else:
            return features_return





