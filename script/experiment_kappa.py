'''
Experiment to measure Kappa
@author: gprana
'''

import configparser
from READMEClassifier.logger import logger
import pandas
from pandas import DataFrame
import numpy as np
import sqlite3
from sqlite3 import Error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from READMEClassifier.script.helper.heuristic2 import *
from READMEClassifier.script.helper.balancer import *
import time
import operator

from sklearn.metrics import confusion_matrix
import math

class kappa_scorer:    
    def __call__(self, estimator, X, y):
        num_labels = len(y)
        kappa = 0
        y_out = estimator.predict(X)
        for i in range(num_labels):  
            y_true = y[i]
            y_pred = y_out[i]
            kappa += float(y_true.sum())/num_labels*self._kappa_score(y_true, y_pred)
        return kappa
    
    def _kappa_score(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return ((tn+fp)*(tn+fn)+(fn+tp)*(fp+tp))/math.pow(len(y_pred),2)
        

def experiment_kappa():
    start = time.time()
    config = configparser.ConfigParser()
    config.read('READMEClassifier/config/config.cfg')
    db_filename = config['DEFAULT']['db_filename']
    rng_seed = int(config['DEFAULT']['rng_seed'])

    conn = sqlite3.connect(db_filename)
    try:
        sql_text = """
        SELECT t1.file_id, t1.section_id, t1.url, t1.heading_text, t2.content_text_w_o_tags, 
        t1.abstracted_heading_text || ' ' || t2.content_text_w_o_tags AS abstracted_heading_plus_content, 
        t1.section_code
        FROM section_overview_75pct t1 
        JOIN section_content_75pct t2 ON t1.file_id=t2.file_id AND t1.section_id=t2.section_id
        """
        df = pandas.read_sql_query(con=conn, sql=sql_text)
        df_randomized_order = df.sample(frac=1, random_state=rng_seed)
        heading_plus_content_corpus = df_randomized_order['abstracted_heading_plus_content']
        content_corpus = df_randomized_order['content_text_w_o_tags']
        heading_text_corpus = df_randomized_order['heading_text']
        url_corpus = df_randomized_order['url']
        
        # Class '2' has been merged into class '1'
        label_set = ['-','1','3','4','5','6','7','8']
        labels = [str(x).split(',') for x in df_randomized_order['section_code']]
        mlb = MultiLabelBinarizer(classes=label_set)
        labels_matrix = mlb.fit_transform(labels)
        
        tfidf = TfidfVectorizer(ngram_range=(1,1), analyzer='word', stop_words='english')
        tfidfX = tfidf.fit_transform(heading_plus_content_corpus)
        
        logger.info('tfidf matrix shape: ')  
        logger.info(tfidfX.shape)
        
        # Derive features from heading text and content
        logger.info('Deriving features')
        derived_features = derive_features_using_heuristics(url_corpus, heading_text_corpus, content_corpus)
                
        logger.info('Derived features shape:')
        logger.info(derived_features.shape)
                
        features_tfidf = pandas.DataFrame(tfidfX.todense())
        # Assign column names to make it easier to print most useful features later
        features_tfidf.columns = tfidf.get_feature_names()
        features_combined = pandas.concat([features_tfidf, derived_features], axis=1)
        
        logger.info('Combined features shape:')
        logger.info(features_combined.shape)
        
        svm_object = LinearSVC() 
        classifier = OneVsRestClassifierBalance(svm_object)
        
        logger.info('Computing overall results')        
        scores_kappa = cross_val_score(classifier, features_combined.values, labels_matrix, cv=10, scoring=kappa_scorer()).mean()
        
        logger.info('Kappa : {0}'.format(scores_kappa))
                
        end = time.time()
        runtime_in_seconds = end - start
        logger.info('Processing completed in {0}'.format(runtime_in_seconds))
    except Error as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)
    finally:
        conn.close()


if __name__ == '__main__':
    experiment_kappa()
