import configparser
from READMEClassifier.logger import logger
import pandas
import sqlite3
from sqlite3 import Error
from READMEClassifier.script.helper.heuristic2 import *
from READMEClassifier.script.helper.balancer import *
from READMEClassifier.script.helper.extractor import *
import time
import operator
import joblib

def find_unique(csv_input_line):
    l = list(set(csv_input_line.split(',')))
    l.sort()
    return l

def load_and_classify_target():
    start = time.time()
    
    config = configparser.ConfigParser()
    config.read('READMEClassifier/config/config.cfg')
    db_filename = config['DEFAULT']['db_filename']
    rng_seed = int(config['DEFAULT']['rng_seed'])
    vectorizer = joblib.load(config['DEFAULT']['vectorizer_filename']) 
    binarizer = joblib.load(config['DEFAULT']['binarizer_filename']) 
    classifier = joblib.load(config['DEFAULT']['model_filename'])
    readme_file_dir = config['DEFAULT']['target_readme_file_dir']
    temp_abstracted_markdown_file_dir = config['DEFAULT']['temp_target_abstracted_markdown_file_dir']
    output_section_code_filename = config['DEFAULT']['output_section_code_filename']
    output_file_codes_filename = config['DEFAULT']['output_file_codes_filename']
    
    # Extract heading
    overview = extract_headings_from_files_in_directory(readme_file_dir, db_filename, 'target_section_overview')
    filenames = retrieve_readme_filenames_from_db(db_filename, 'target_section_overview')
    delete_existing_section_content_data(temp_abstracted_markdown_file_dir, db_filename, 'target_section_content')
    abstract_out_markdown(filenames, readme_file_dir, temp_abstracted_markdown_file_dir)
    extract_section_from_abstracted_files_v2(temp_abstracted_markdown_file_dir, db_filename, 'target_section_overview','target_section_content')
    
    '''
    Classifier part
    '''
    conn = sqlite3.connect(db_filename)
    try:
        sql_text = """
        SELECT t1.file_id, t1.section_id, t1.url, t1.local_readme_file, t1.heading_markdown, t1.abstracted_heading_markdown,
        t1.heading_text, t1.abstracted_heading_text, t1.heading_level, t2.content_text_w_o_tags, 
        t1.abstracted_heading_text || ' ' || t2.content_text_w_o_tags AS abstracted_heading_plus_content
        FROM target_section_overview t1 
        JOIN target_section_content t2 
        ON t1.file_id=t2.file_id AND t1.section_id=t2.section_id
        ORDER BY t1.file_id, t1.section_id
        """
        df = pandas.read_sql_query(con=conn, sql=sql_text)
        
        heading_plus_content_corpus = df['abstracted_heading_plus_content']
        content_corpus = df['content_text_w_o_tags']
        heading_text_corpus = df['heading_text']
        url_corpus = df['url']
        
        tfidfX = vectorizer.transform(heading_plus_content_corpus)
        
        logger.info('tfidf matrix shape: ')  
        logger.info(tfidfX.shape)
        
        # Derive features from heading text and content
        logger.info('Deriving features')
        derived_features = derive_features_using_heuristics(url_corpus, heading_text_corpus, content_corpus)
                
        logger.debug('Derived features shape:')
        logger.debug(derived_features.shape)
                
        features_tfidf = pandas.DataFrame(tfidfX.todense())
        features_tfidf.columns = vectorizer.get_feature_names()
        features_combined = pandas.concat([features_tfidf, derived_features], axis=1)
        
        logger.debug('Combined features shape:')
        logger.debug(features_combined.shape)

        labels_matrix = classifier.predict(features_combined.values)
        df['section_code'] = [','.join(x) for x in binarizer.inverse_transform(labels_matrix)]
        # Update DB table
        df = df.loc[:,['file_id', 'section_id', 'url', 'local_readme_file', 'heading_markdown', 'abstracted_heading_markdown','heading_text', 'abstracted_heading_text',
                       'heading_level','section_code']]
        df.to_sql(name='target_section_overview', con=conn, if_exists='replace', index=False)
        # Export result to file
        output = df.loc[:,['file_id','section_id','local_readme_file','heading_markdown','section_code']]
        output.to_csv(output_section_code_filename, sep=',', index=False)
        # Find missing section codes
        output_file_completeness = df.loc[:,['local_readme_file','section_code']]
        output_file_completeness = output_file_completeness.groupby('local_readme_file', sort=False)['section_code'].apply(lambda x: ','.join(x)).reset_index(name='section_codes_in_file')
        # Remove duplicates (i.e. change '1,1,2,2,3,3' into '1,2,3')
        output_file_completeness['section_codes_in_file'] = output_file_completeness['section_codes_in_file'].apply(lambda x: ','.join(find_unique(x)))
        # Find codes that aren't yet in the README. Omit '-' because we don't need 'Exclusion' sections.
        output_file_completeness['codes_not_in_file'] = output_file_completeness['section_codes_in_file'].apply(lambda x: ','.join(sorted(list(set(['1','3','4','5','6','7','8']) - set(x.split(','))))))
        output_file_completeness.to_csv(output_file_codes_filename, sep=',', index=False)
    except Error as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)
    finally:
        conn.close()
    end = time.time()
    runtime_in_seconds = end - start
    logger.info('Processing time: {0}'.format(runtime_in_seconds))


if __name__ == '__main__':
    load_and_classify_target()
