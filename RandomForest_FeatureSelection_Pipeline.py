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
        self.savedfeatures_names = savedfeatures_names
        self.current_features = current_features
        
    def fit (self, X, y=None):
        self.all_savedfeatures_names = self.savedfeatures_names
        return self
    
    def transform (self, X):
        X = pd.DataFrame(data=X, columns = self.current_features)
        output = X.copy()
        output = output[self.savedfeatures_names]
        return output
    
    def get_support(self):
        return self.savedfeatures_names
        
        

class RandomForestFeatureSelectionPipeline:
    def __init__(self, data, test,n_estimators = 200, max_depth = None):
        self.data = data
        self.test = test
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def model(self, topvalues = 0):
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
            
        rfmodel = RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,
                                         random_state=66, 
                                              n_jobs=int(os.cpu_count() / 3))
        rfmodel.fit(X_train,y_train)
        important = rfmodel.feature_importances_
        
        
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





