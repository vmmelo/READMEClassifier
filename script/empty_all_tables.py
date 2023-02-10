import configparser
import sys
import sqlite3
from READMEClassifier.logger import logger

def empty_all_tables():
    config = configparser.ConfigParser()
    config.read('READMEClassifier/config/config.cfg')
    db_filename = config['DEFAULT']['db_filename']

    try:
        logger.info('Emptying all tables')
        conn = sqlite3.connect(db_filename)
        # Delete existing data
        c = conn.cursor()
        c.execute('DELETE FROM section_overview_25pct')
        c.execute('DELETE FROM section_content_25pct')
        c.execute('DELETE FROM section_overview_75pct')
        c.execute('DELETE FROM section_content_75pct')
        c.execute('DELETE FROM section_content_combined')
        c.execute('DELETE FROM section_overview_combined')
        c.execute('DELETE FROM target_section_overview')
        c.execute('DELETE FROM target_section_content')
        conn.commit()
        logger.info('Operation completed')
    except Exception as e:
        logger.exception(e)
    finally:
        conn.close()


if __name__ == '__main__':
    empty_all_tables()
