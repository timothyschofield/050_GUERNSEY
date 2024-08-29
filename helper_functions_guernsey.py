
def get_headers(api_key):
  
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }
  
  return headers


# Sometimes ChatGPT returns invalid keys in the JSON
# The json_returned must be valid JSON
def are_keys_valid(json_returned, prompt_key_names):
  
    dict_returned = eval(json_returned)
    dict_returned_keys = list(dict_returned.keys())
    if prompt_key_names == dict_returned_keys:
        print("VALID returned JSON keys")
        return True
    else:
        print("INVALID returned JSON keys")
        return False
      
# Must be called after you know 200 has been returned
def clean_up_ocr_output_json_content(ocr_output_in):
  
    # json() returns a JSON object of the result
    json_returned = ocr_output_in.json()['choices'][0]['message']['content']

    # HERE I DEAL WITH SOME FORMATS THAT CREATE INVALID JSON
    # 1) Turn to raw with "r" to avoid the escaping quotes problem
    json_returned = fr'{json_returned}'
    
    # 2) Sometimes null still gets returned, even though I asked it not to
    if "null" in json_returned: 
      json_returned = json_returned.replace("null", "'none'")
    
    # 3) Occasionaly the whole of the otherwise valid JSON is returned with surrounding square brackets like '[{"text":"tim"}]'
    # or other odd things like markup '''json and ''' etc.
    # This removes everything prior to the opening "{" and after the closeing "}"
    open_brace_index = json_returned.find("{")
    json_returned = json_returned[open_brace_index:]
    close_brace_index = json_returned.rfind("}")
    json_returned = json_returned[:close_brace_index+1]

    return json_returned


def print_all_chars_from_file(file_name):
  with open(file_name) as f:
    text = f.read()
  print_all_chars(text)

def print_all_chars(x):
  for i in range(len(x)):
    print(f"***{x[i]}*** {ord(x[i])}")


import xml.etree.ElementTree as ET
def validate_xml(xml_text):
    try:
        ET.fromstring(xml_text)
        return True, "The XML is well-formed."
    except ET.ParseError as e:
        return False, f"XML is not well-formed: {e}"

from datetime import datetime
# e.g. 2024-05-18T06-53-26
def get_file_timestamp():
  current_dateTime = datetime.now()
  year = current_dateTime.year
  month = current_dateTime.month
  day = current_dateTime.day
  hour = current_dateTime.hour
  minute = current_dateTime.minute
  second = current_dateTime.second
  return f"{year}-{month:02}-{day:02}T{hour:02}-{minute:02}-{second:02}"
    
import json
def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True
 
import torch
def get_torch_cuda_info():
  print("-----------------------")
  print("torch version", torch.__version__)
  print("CUDA avaliable", torch.cuda.is_available())
  print("torch.CUDA version", torch.version.cuda)
  print("-----------------------")

  if torch.cuda.is_available():
      print('__CUDNN VERSION:', torch.backends.cudnn.version())   # CUDA Deep Neural Networks
      print('__Number CUDA Devices:', torch.cuda.device_count())
      print('__CUDA Device Name:',torch.cuda.get_device_name(0))
      print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
      print("-----------------------")

import pandas as pd
from pathlib import Path
def create_and_save_dataframe(output_list, key_list_with_logging, output_path_name):
  output_df = pd.DataFrame(output_list)

  if key_list_with_logging != []:
    output_df = output_df[key_list_with_logging]  # Bring reorder dataframe to bring source url and error column to the front

  output_path = Path(output_path_name)
  with open(output_path, encoding="utf-8", mode="w") as f:
    output_df.to_csv(f, index=False)
    

def save_dataframe_to_csv(df_to_save, output_path):
  
  with open(f"{output_path}.csv", encoding="utf-8", mode="w") as f:
    df_to_save.to_csv(f, index=False)

      
# source_type "url" or "local"
def make_payload(model, prompt, source_type, image_path, num_tokens):
  
  if source_type == "url":
    url_request = image_path
  else:
    base64_image = encode_image(image_path)
    url_request = f"data:image/jpeg;base64,{base64_image}"
  
  payload = {
    "model": model,
    "logprobs": False,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "temperature": "0.0",
            "text": prompt
          },
          {
            "type": "image_url",
            "image_url": {"url": url_request}
          }
        ]
      }
    ],
    "max_tokens": num_tokens
  } 
  
  return payload


import base64
# Function to base64 encode an image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8') 


import requests
def download_all_jpgs():

    input_folder = "ny_herbarium_input"
    input_file = "NY_specimens_to_transcribe.csv"
    input_path = Path(f"{input_folder}/{input_file}")
    df_input_csv = pd.read_csv(input_path)

    input_url_list = df_input_csv["DarImageURL"]
    # print(input_url_list)

    output_folder = "jpg_folder_input"
    count = 1
    for url in input_url_list:

        url_list = url.split("/")
        filename = url_list[-1]
        print(f"{count}: {filename}")
        count = count + 1
        
        res = requests.get(url)

        # print(type(res)) # requests.models.Response
        # print(res.url)
        # print(res.status_code)

        output_path = Path(f"{output_folder}/{filename}")

        if res.status_code == 200:
            with open(output_path,'wb') as f:
                print(f"Writing: {output_path}")
                f.write(res.content)
        else:
            print("200 not returned")