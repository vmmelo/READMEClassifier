'''
@author: gprana
'''
import configparser
from READMEClassifier.script.helper.extractor import *
import logging
import sqlite3

def load_dev_and_eval_datasets():
    log_filename = 'READMEClassifier/log/load_dev_and_eval_datasets.log'
    logging.basicConfig(handlers=[logging.FileHandler(log_filename, 'w+', 'utf-8')], level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    
    config = configparser.ConfigParser()
    config.read('READMEClassifier/config/config.cfg')
    db_filename = config['DEFAULT']['db_filename']
    readme_file_dir = config['DEFAULT']['readme_file_dir']
    temp_abstracted_markdown_file_dir = config['DEFAULT']['temp_abstracted_markdown_file_dir']
    
    for n in ['25','75']:
        logging.info(f'Emptying tables for {n} pct data')
        conn = sqlite3.connect(db_filename)
        # Delete existing data
        c = conn.cursor()
        c.execute(f'DELETE FROM section_overview_{n}pct')
        c.execute(f'DELETE FROM section_content_{n}pct')
        conn.commit()
        logging.info('Cleanup completed')
        
        input_filename_csv = config['DEFAULT'][f'section_overview_{n}pct_filename']
        logging.info(f'Processing files from {input_filename_csv}')
        load_section_overview_from_csv(input_filename_csv, db_filename, f'section_overview_{n}pct')
        filenames = retrieve_readme_filenames_from_db(db_filename, f'section_overview_{n}pct')
        logging.info(f'Processing README file. Using {temp_abstracted_markdown_file_dir} for temp storage')
        delete_existing_section_content_data(temp_abstracted_markdown_file_dir, db_filename, f'section_content_{n}pct')
        abstract_out_markdown(filenames, readme_file_dir, temp_abstracted_markdown_file_dir)
        extract_section_from_abstracted_files(temp_abstracted_markdown_file_dir, db_filename,
                                              f'section_overview_{n}pct',
                                              f'section_content_{n}pct')
        logging.info(f'Finished loading data for {n} pct set')
    logging.info('Operation completed')

if __name__ == '__main__':
    load_dev_and_eval_datasets()
