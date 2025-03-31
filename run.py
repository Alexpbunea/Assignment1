import sys
import os
import sqlite3
import json
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from scripts.training_file import *
from scripts.m_schema import *
from scripts.utils import *
from scripts.generating import *
from scripts.metrics import *
from scripts.refining import *


folder = "./dev/dev_databases/dev_databases/" #Databases inital folder
output_directory = "./jsons/" #for the jsons

train_dataset_path = "./jsons/train_dataset.jsonl"

questions = "./dev/dev.json" #natural lenguaje questions
json_path = "./jsons/dev_dataset.jsonl" #jsonl file with the input used for generation and the ouput used for calculating metrics
model_saved_weights = "./fine_tuned_t5_small"
raw_generated_outputs_path = "./generated_jsons/raw_generated_outputs.jsonl" #raw outputs, without refining
results_path = "./generated_jsons/final_output.jsonl" #final outputs with the hybrid approach

groundtruth_path = "./jsons/dev_dataset.jsonl" #the same as the json_path, for the reason I explained



def install_requirements(): #checks if the requiermentes are installed or not. If not, it installs them using pip
    try:
        import pkg_resources
        import subprocess
        requirements = open("requirements.txt").read().splitlines()
        missing_packages = []

        for package in requirements:
            try:
                pkg_resources.require(package)
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package)

        if missing_packages:
            print(f"Installing missing packages: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        else:
            print("All packages are already installed!")

        print()

    except Exception as e:
        print(f"[ERROR]: When checking packages: {e}")
        sys.exit()


def first_question():
    print("Hi, I am the database assistant. I can help you with your databases.")
    print("Which of the following options would you like to choose?")
    print(" - 1. Would you like to train the model based on your databases?")
    print(" - 2. Would you like to use your model to generate new outputs?")
    print(" - 3. Would you like to evaluate your model's outputs?")
    print(" - 4. Exit\n")

    while True:
        answer = input()
        print()
        
        match answer:
            case "1":
                print("You have chosen to train the model based on your databases.")
                train()
                sys.exit()
            case "2":
                print("You have chosen to use your model to generate new outputs.")
                generate_output()
                sys.exit()
            case "3":
                print("You have chosen to evaluate your model's outputs.")
                evaluate_model()
                sys.exit()
            case "4":
                print("Goodbye!")
                sys.exit()
            case _:
                print("Invalid option. Please choose again.\n")

def train():
    #For training the model, you can select between 2 jsonl files. I tried to train the model with two different m-schemas. A more complex one containing not only
    #the tables and the id, but also the columns and a few examples, called "dataset_complicated.jsonl". But, by the model being so small, the results are not good at all. That's why, I came up with the idea
    #of making the schema the simple and direct as posible. It can be found in the jsons directory, the "train_dataset.jsonl" file.

    """
    #CODE I USED TO GENERATE THE TRAINING DATASET. IT HAS MORE OR LESS THE SAME PHYLOSOPHY AS THE ONE USED FOR CREATING dev_dataset.jsonl, EXPLAINED BELOW

    questions_sql = load_json("./jsons/train.json")
    m_schema_dataset = load_json("./jsons/tables_dataset(1).json")

    #m_schema_dataset = create_tables_dataset(folder)
    merged_dataset = merge_datasets(m_schema_dataset, questions_sql)
    #write_json(output_directory, "train_dataset.json", merged_dataset) #writing the json file using the function from utils.py
    jsonl_file = create_structure_for_jsonl_file(merged_dataset)
    write_json(output_directory, "train_dataset.jsonl", jsonl_file) #writing the jsonl file using the function from utils.py
    #"""

    global train_dataset_path
    print("[DISCLAIMER]: This will download the Google t5 small model. If you want a different one, please check the training_file.py file. Also, please be sure to have the correct path for the jsonl file.")
    print(f"Default path for the jsonl file of the dataset: {train_dataset_path} ; Do you want to change it? (y/n)" )
    answer = yes_or_no(input())
    if answer is True:
        print("Please enter the path: ")
        train_dataset_path = input()
    print("Training the model...")
    training_model(train_dataset_path)


def generate_output(): #generates the output and refines it automatically. However, for comparison puporses, two files are stored, a raw one and a refined one.
    global json_path
    global results_path
    global model_saved_weights
    print("[DISCLAIMER]: Please be sure to have the model downloaded and fine-tuned before running this. If not, several errors may occur. Also, please be sure to have the correct paths for the jsonl files.")
    
    """
    #UNMARK THIS SMALL AMOUNT OF LINES OF CODE IF YOU NEED TO CREATE THE JSONL FILE FOR THE MODEL TO GENERATE OUTPUTS
    
    #Basically, does this:
    # - First line -> Goes throughout all the mysql files of the databases in search of the id and the table names
    # - Second line -> Writes this information in a new json file called "m_schema_dataset.json". (Just in case there is an error, to facilite its correction)
    # - Third line -> Loads the "dev.json" file found when you download the dev set from the bird benchmark web
    # - Fourth line ->  Looking at the table id, this function, in combination of others, takes the m_schema just created and the questions json loaded, and creates as an example:
    
        # {
        # "db_id": "california_schools",
        # "tables": [
        # "frpm",
        # "satscores",
        # "schools"
        # ],
        # "qa": [
        # {
        #     "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        #     "output": [
        #     "frpm"
        #     ]
        # }
    
    # - Fith line -> Writes the whole dicctionary in a new json file.
    # - Sixth and last line -> Takes the new merge dataset and writes it on a jsonl file with this structure:
            
    #        prompt = f"Database ID: {db_id}\nQuestion: {question}\nSelect table(s) from: {schema_str}" Note: Schema_str are the tables
    #        json_line = {"input": prompt, "output": output_text}
    
    # For the example os above:
        # {
        # "input": "Database ID: california_schools\nQuestion: What is the highest eligible free rate for K-12 students in the schools in Alameda County?\nSelect table(s) from: ['frpm', 'satscores', 'schools']",
        # "output": [
        # "frpm"
        # ]
        # },
    #
    #-----------------------------------------------------------------------------------------------------------
    
    m_schema_dataset = create_tables_dataset(folder)
    
    write_json(output_directory, "m_schema_dataset.json", m_schema_dataset) #writing the json file using the function from utils.py
    
    questions_sql = load_json(train_json_path)
    merged_dataset = merge_datasets(m_schema_dataset, questions_sql)
    write_json(output_directory, "dev_dataset.json", merged_dataset) #writing the json file using the function from utils.py

    #creating the jsonl file
    jsonl_file = create_structure_for_jsonl_file(merged_dataset)
    write_json(output_directory, "dev_dataset.jsonl", jsonl_file) #writing the jsonl file using the function from utils.py
    #"""
    
    
    print(f"Default path for the jsonl file for the inputs: {json_path} ; Do you want to change it? (y/n) ------- [Note]: If you don't have the JONSL, please check the 'generated_output' function on the run.py script." )
    answer = yes_or_no(input())
    if answer is True:
        print("Please enter the path: ")
        json_path = input()
    
    print(f"Default path for the final jsonl file with the inputs with the outputs: {results_path} ; Do you want to change it? (y/n)")
    answer = yes_or_no(input())
    if answer is True:
        print("Please enter the path:")
        results_path = input()
    
    print(f"Default path for the model's saved weights:  {model_saved_weights} ; Do you want to change it? (y/n)")
    answer = yes_or_no(input())
    if answer is True:
        print("Please enter the path:")
        model_saved_weights = input()
    
    input_data = load_json(json_path)
    print("\nGenerating outputs...")
    generate(input_data, model_saved_weights, raw_generated_outputs_path) #using the function from generating.py
    print("Outputs generated successfully.\nRefining the results...\n")
    
    time.sleep(1) #Just in case
    refinining(raw_generated_outputs_path, results_path)
    print("Finished!")



def evaluate_model(): #evaluates the metrics
    global results_path
    global groundtruth_path
    print(f"Default path for the jsonl file: {results_path} ; Do you want to change it? (y/n)")
    answer = yes_or_no(input())
    if answer is True:
        print("Please enter the new path:")
        results_path = input()

    print(f"Default path for the groundtruth jsonl file: {groundtruth_path} ; Do you want to change it? (y/n)")
    answer = yes_or_no(input())
    if answer is True:
        print("Please enter the new path:")
        groundtruth_path = input()
    
    print("Calculating the metrics...")
    try:
        evaluate(results_path, groundtruth_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the paths or the jsonl files format and try again.")
        




def main():
    #checking every needed packeage is installed. If not, is installed.
    install_requirements()

    #just in case, direct and quickly initialization is needed.
    parser = argparse.ArgumentParser(description="Evaluate the model's outputs.")
    parser.add_argument("--action", type=str, choices=["train", "generate", "evaluate", "other"], default="other", help="Action to perform (evaluate or other).")
    args = parser.parse_args()
    
    if args.action == "train":
        train()
    elif args.action == "generate":
        generate_output()
    elif args.action == "evaluate":
        print("Performing evaluation...")
        evaluate_model() 
    else:
        first_question()
    
    """
    questions_sql = load_json("./jsons/train.json")
    m_schema_dataset = load_json("./jsons/tables_dataset(1).json")

    #m_schema_dataset = create_tables_dataset(folder)
    merged_dataset = merge_datasets(m_schema_dataset, questions_sql)
    #write_json(output_directory, "train_dataset.json", merged_dataset) #writing the json file using the function from utils.py
    jsonl_file = create_structure_for_jsonl_file(merged_dataset)
    write_json(output_directory, "train_dataset.jsonl", jsonl_file) #writing the jsonl file using the function from utils.py
    #"""


if __name__ == "__main__":
    main()