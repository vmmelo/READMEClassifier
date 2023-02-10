import configparser
from READMEClassifier.logger import logger
import pandas
import sqlite3
from sqlite3 import Error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from READMEClassifier.script.helper.heuristic2 import *
from READMEClassifier.script.helper.balancer import *
import time
import operator
import joblib
from sklearn.calibration import CalibratedClassifierCV

def classifier_train_model_predict_proba():
    start = time.time()
    
    config = configparser.ConfigParser()
    config.read('READMEClassifier/config/config.cfg')
    db_filename = config['DEFAULT']['db_filename']
    rng_seed = int(config['DEFAULT']['rng_seed'])
    vectorizer_filename = config['DEFAULT']['vectorizer_filename'] 
    binarizer_filename = config['DEFAULT']['binarizer_filename'] 
    model_filename = config['DEFAULT']['model_filename'] 
    
    conn = sqlite3.connect(db_filename)
    try:
        sql_text1 = """
        SELECT t1.file_id, t1.section_id, t1.url, t1.heading_text, t2.content_text_w_o_tags, 
        t1.abstracted_heading_text || ' ' || t2.content_text_w_o_tags AS abstracted_heading_plus_content, 
        t1.section_code
        FROM section_overview_combined t1 
        JOIN section_content_combined t2 
        ON t1.file_id=t2.file_id AND t1.section_id=t2.section_id
        """
        df = pandas.read_sql_query(con=conn, sql=sql_text1)
        
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
        clf = CalibratedClassifierCV(svm_object) 
        classifier = OneVsRestClassifierBalance(clf)
        
        logger.info('Training classifier')
        classifier.fit(features_combined.values, labels_matrix) 
        logger.info('Saving TFIDF vectorizer')
        joblib.dump(tfidf, vectorizer_filename)
        logger.info('Saving binarizer')
        joblib.dump(mlb, binarizer_filename)
        logger.info('Saving model')
        joblib.dump(classifier, model_filename)
                
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
    classifier_train_model_predict_proba()
