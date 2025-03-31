import os
import sqlite3
import json
from utils import *

def extract_tables_from_sqlite(sqlite_path):
    tables = []

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Obtener los nombres de las tablas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]  # Extraemos solo los nombres

    conn.close()
    return tables

def create_tables_dataset(base_folder):
    
    tables_dataset = []

    for db_folder in os.listdir(base_folder):
        db_folder_path = os.path.join(base_folder, db_folder)
        if not os.path.isdir(db_folder_path):
            continue

        for sql_file in os.listdir(db_folder_path):
            if sql_file.endswith(".sqlite"):
                sqlite_path = os.path.join(db_folder_path, sql_file)
                print(f"Procesing: {sqlite_path}")

                tables = extract_tables_from_sqlite(sqlite_path)

                tables_entry = {
                    "db_id": db_folder,  #Database id
                    "tables": tables     #List of the tables
                }
                tables_dataset.append(tables_entry)
                

    return tables_dataset