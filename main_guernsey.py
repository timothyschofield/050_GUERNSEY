"""
    File : main_guernsey.py

    Author: Tim Schofield
    Date: 02 July 2024

    $0.93
    index: 99 710 seconds
    num_not_200_errors=1
    num_invalid_JSON_errors=2
    num_invalid_keys_errors=0

"""
import openai
from openai import OpenAI
from dotenv import load_dotenv
from helper_functions_guernsey import get_file_timestamp, is_json, make_payload, clean_up_ocr_output_json_content, are_keys_valid, get_headers, save_dataframe_to_csv

import requests
import os
from pathlib import Path 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import json
import sys
print(f"Python version {sys.version}")

MODEL = "gpt-4o" # Context window of 128k max_tokens 4096

load_dotenv()
try:
    my_api_key = os.environ['OPENAI_API_KEY']          
    client = OpenAI(api_key=my_api_key)
except Exception as ex:
    print("Exception:", ex)
    exit()


input_folder = "guernsey_input"
input_file = "Guernsey url - Sheet1.csv"
input_path = Path(f"{input_folder}/{input_file}")

input_jpg_folder = "jpg_folder_input"

output_folder = "guernsey_output"

project_name = "guernsey_test_"


time_stamp = get_file_timestamp()

# This is just blank exept for the columns already filled in by the client
df_input_csv = pd.read_csv(input_path)

# This would all be fine except that a url column might contain multiple image urls in one line seperated a pipes ("|")
# So its easiest just to get it out of the way and make a df copy with each url having its own line - the new df will have more lines obviously
# In Guernsey's case this isn't so - but I left it in anyway
to_transcribe_list = []
for index, row in df_input_csv.iterrows():

    dar_image_url = row["url"]
    if "|" in dar_image_url:
        
        url_list = dar_image_url.split("|")
        for url in url_list:
            url = url.strip()
            this_row = df_input_csv.loc[index].copy()
            this_row["url"] = url
            to_transcribe_list.append(this_row)
    else:
        this_row  = df_input_csv.loc[index].copy() 
        to_transcribe_list.append(this_row)

df_to_transcribe = pd.DataFrame(to_transcribe_list).fillna("")

# Columns that were not in the input CSV
df_to_transcribe["ERROR"] = "OK"
df_to_transcribe["ocr_text"] = "No OCR text"

# Necessary because by copying rows to give each url a seperate row, we have also copied indexes
# We want each row to have its own index - so reset_index
df_to_transcribe.reset_index(drop=True, inplace=True)

# These are the columns that ChatGPT will try to fill from the OCR
ocr_column_names = [ 
        ("genus","genus"), 
        ("species","species"),
        ("collector","collector"),
        ("year_YYYY","year_YYYY"),	
        ("month_MM","month_MM"),	
        ("day_DD","day_DD"),
        ("locality","locality"), 
        ("lat_DMS","lat_DMS"), 
        ("lng_DMS","lng_DMS"), 
        ("lat_decimal","lat_decimal"), 
        ("lng_decimal","lng_decimal"), 
        ("location_accuracy_meters","location_accuracy_meters"), 
        ("altitude","altitude"), 
        ("altitude_units","altitude_units"), 
        ("habitat","habitat"), 
        ("cat_number","cat_number"), 
        ("kentNumber","kentNumber"),
        ("error","error"),
        ("ocr_text","ocr_text")
 ]

df_column_names = []          # To make the DataFrame with
prompt_key_names = []         # To use in the prompt for ChatGPT
empty_output_dict = dict([])  # Useful when you have an error but still need to return a whole DataFrame row
for df_name, prompt_name in ocr_column_names:
    df_column_names.append(df_name)     
    prompt_key_names.append(prompt_name)   
    empty_output_dict[df_name] = ""

keys_concatenated = ", ".join(prompt_key_names) # For the prompt


# Should check that the columns in df_column_names are in df_to_transcribe
# If they are not, no error occures, but the OCR output will be not copied into the missing columns
# This is silent and bad
df_to_transcribe_keys = list(df_to_transcribe.keys())
if set(df_column_names) <= set(df_to_transcribe_keys):
    print("df_column_names is a subset of df_to_transcribe_keys - GOOD")
else:
    print("ERROR: df_column_names is NOT a subset of df_to_transcribe_keys - BAD")
    exit()


# Guernsey
prompt = (
    f"Read this herbarium sheet and extract all the text you can."
    f"Go through the text you have extracted and return data in JSON format with {keys_concatenated} as keys."
    f"Use exactly {keys_concatenated} as keys."
    
    f"Return the text you have extracted in the field ocr_text."
    
    f"If you see a number preceded by a string like 'Lon. Cat. Ed. 8, No.' return the number as the cat_number."
    
    f"Look very carefully at the botton right of the image for the kentNumber. The kentNumber is very, very faint."
    f"Spend some time looking for the faint kentNumber in the bottom right."
    f"Examples of kentNumbers are '5.', '4.5', '4.6', '4.9', '4/10', '13.', '14.', '15.1', '15.2', '15-5', '15.7', '25.1' '46.3', '46.12', '66.6b', '79.6', '86/1', '97-1', '75.35.7', etc."
    f"Include the '.', '-' or '/' between the integers in the kentNumber you return."
    
    f"locality field information often appears after printed words like 'LOCALITY' or 'Locality'."
    
    f"Unless otherwise stated, assume the locality is somewhere in the Channel Islands in the English Channel."
    f"If the island name is not mentioned try to find it from the locality infomation. Add the island name to the locality information."
    
    f"If you have found a locality description, find out the latitude and longitude of the location."
    f"Return the latitude and longitude in degrees, minutes and seconds in the lat_DMS and lng_DMS fields respectivly."
    f"Return the decimal latitude and longitude in lat_decimal and lng_decimal fields respectivly."
    f"Estimate how accurate the latitude and longitude information is in meters and return it in the location_accuracy_meters field."
    
    f"Date field information often appears after printed words like 'DATE' or 'Date'. Return the year in year_YYYY, the month in month_MM and day in day_DD."
    f"year must be 4 digits."
    f"month and day must two digits, packing single digit values with a leading zero, e.g. '3' should be returned as '03', '9' as should be returned as '09'"
    
    f"If the altitude the specimen was collected at is mentioned return it in the altitude field. Return the unit of altitude measurement in the altitude_units field, e.g. meters or feet."

    f"habitat field information often appears after printed words like 'HABITAT' or 'Habitat', examples of habitat are 'waste ground', 'high hedged bank', 'cliff top', shady bank by stream', 'field by a road'."
    
    f"collector field information often appears after printed words like 'COLLECTOR', 'Collector', 'Coll.' or 'Det.'."
    f"If the genitive case is used, e.g. Gosselin's Specimens, the collector is Gosselin."
    
    f"If you can not find a value for a key return value ''"
)

headers = get_headers(my_api_key)

source_type = "url" # "url" or "local"
batch_size = 20 # saves every
num_not_200_errors = 0
num_invalid_JSON_errors = 0
num_invalid_keys_errors = 0
start_time = int(time.time())
print("####################################### START OUTPUT ######################################")
for index, row in df_to_transcribe.iloc[0:100].iterrows():
 
    image_path = row["url"]
    
    if source_type != "url":
        # JPGs in local folder
        filename = image_path.split("/")[-1]
        image_path = Path(f"{input_jpg_folder}/{filename}")
        if image_path.is_file() == False:
            print(f"File {image_path} does not exist")
            exit()
            
    print(f"\n########################## OCR OUTPUT {image_path} ##########################")
    end_time = int(time.time())
    print(f"index: {index} {end_time - start_time} seconds")
    
    payload = make_payload(model=MODEL, prompt=prompt, source_type=source_type, image_path=image_path, num_tokens=4096)

    num_tries = 3
    for i in range(num_tries):
        ocr_output = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        response_code = ocr_output.status_code
        if response_code != 200:
            # NOT 200
            num_not_200_errors = num_not_200_errors + 1
            print(f"======= 200 not returned {response_code}. Trying request again number: {i} ===========================")
            time.sleep(0.5)
        else:
            # YES 200
            json_returned = clean_up_ocr_output_json_content(ocr_output)
            json_valid = is_json(json_returned)
            if json_valid == False:
                # INVALID JSON
                num_invalid_JSON_errors = num_invalid_JSON_errors + 1
                print(f"======= Returned JSON content not valid. Trying request again number: {i} ===========================")
                print(f"INVALID JSON content****{json_returned}****")
            else:
                # VALID JSON
                # Have to check that the returned JSON keys are correct 
                # Sometimes ChatGPT just doesn't do as its told and changes the key names!
                if are_keys_valid(json_returned, prompt_key_names) == False:
                    # INVALID KEYS
                    num_invalid_keys_errors = num_invalid_keys_errors + 1
                    print(f"======= Returned JSON contains invalid keys. Trying request again number: {i} ===========================")
                else:
                    # VALID KEYS
                    break
                
    ###### eo try requests three times

    # OK - we've tried three time to get
    # 1. 200 returned AND
    # 2. valid JSON returned AND
    # 3. valid key names
    # Now we have to create a valid Dict line for the spreadsheet
    error_message = "OK"
    dict_returned = dict()
    if response_code != 200:
        # NOT 200
        # Make a Dict line from the standard empty Dict and 
        # put the whole of the returned message in the OcrText field
        print("RAW ocr_output ****", ocr_output.json(),"****")                   
        dict_returned = eval(str(empty_output_dict))
        dict_returned['ocr_text'] = str(ocr_output.json())
        error_message = "200 NOT RETURNED FROM GPT"
        print(error_message)
    else:
        # YES 200
        print(f"content****{json_returned}****")
    
        if is_json(json_returned):
            # VALID JSON
            
            # Have to deal with the possibility of invalid keys returned in the valid JSON
            if are_keys_valid(json_returned, prompt_key_names):
                # VALID KEYS
                # Now change all the key names from the human readable used in the prompt to 
                # DataFrame output names to match the NY spreadsheet
                
                dict_returned = eval(json_returned) # JSON -> Dict
                
                # Chenge the prompt names for CSV keys
                for df_name, prompt_name in ocr_column_names:
                    dict_returned[df_name] = dict_returned.pop(prompt_name)
            else:
                # INVALID KEYS
                dict_returned = eval(str(empty_output_dict))
                dict_returned['ocr_text'] = str(json_returned)                  
                error_message = "INVALID JSON KEYS RETURNED FROM GPT"
                print(error_message)
        else:
            # INVALID JSON
            # Make a Dict line from the standard empty Dict and 
            # just put the invalid JSON in the OcrText field
            dict_returned = eval(str(empty_output_dict))
            dict_returned['ocr_text'] = str(json_returned)
            error_message = "JSON NOT RETURNED FROM GPT"
            print(error_message)
        
   
    ###### EO dealing with various types of returned code ######
    
    dict_returned["ERROR"] = str(error_message)  # Insert error message into output
    
    df_to_transcribe.loc[index, dict_returned.keys()] = dict_returned.values() # <<<<<<<<<<<<<<<<< 
    
    if index % batch_size == 0 and index != 0:
        print(f"WRITING BATCH:{index}")
        output_path = f"{output_folder}/{project_name}_{time_stamp}-{index:04}"
        save_dataframe_to_csv(df_to_save=df_to_transcribe, output_path=output_path)

#################################### eo for loop ####################################

# For safe measure and during testing where batches are not %batch_size
print(f"WRITING BATCH:{index}")
output_path = f"{output_folder}/{project_name}_{time_stamp}-{index:04}"
save_dataframe_to_csv(df_to_save=df_to_transcribe, output_path=output_path)

print(f"index: {index} {end_time - start_time} seconds")
print(f"{num_not_200_errors=}")
print(f"{num_invalid_JSON_errors=}")
print(f"{num_invalid_keys_errors=}")

print("####################################### END OUTPUT ######################################")
  

  


