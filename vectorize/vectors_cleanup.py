import logging
import os
import sqlite3

"""
Script to prune db entries whose audio files no longer exist on disk
Updated: 2022-10-04
"""

logger = logging.getLogger(__name__)

if __name__ == '__main__':
  os.chdir(os.path.dirname(__file__))

  # Read from SQLite DB
  con = sqlite3.connect("../models/mp3tovecs.db")
  cur = con.cursor()
  # query_like_dirs = ['%7!!%'] # Add/Edit here for custom query
  query_like_dirs = [r'%D:\%'] # Add/Edit here for custom query

  for query in query_like_dirs:
    try:
      cur.execute("""SELECT fullpath FROM vectors WHERE fullpath LIKE (?)""", [query])
      db_p = ""
      pending_delete_filepath = []
      while(db_p is not None):
        db_p = cur.fetchone()
        if (db_p is not None):
          fullpath = db_p[0]
          if not os.path.exists(fullpath):
            print(f'File not found: {fullpath}')
            pending_delete_filepath.append(fullpath)
      
      for file in pending_delete_filepath:
        cur.execute("""DELETE FROM vectors WHERE fullpath=(?)""", [file])
    except:
      print(f'Exception, skipping query: {query}')
      continue

  con.commit()

  # Compact database (Will rewrite entire file)
  con.execute("VACUUM")

  con.close()
