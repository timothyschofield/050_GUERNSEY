"""
    
    File: test_structured_output.py
    Author: Tim Schofield
    Date: 13 August 2024

    https://openai.com/index/introducing-structured-outputs-in-the-api/
    https://docs.pydantic.dev/latest/
    
    
    Use gpt-4o-2024-08-06
    
    pip list

    pip install openai --upgrade
    print(openai.VERSION) # 1.40.6
    pip install --upgrade package-name
------------------------------------------------
    The types "str", "list" are provided by Pylance
    https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance
    Pylance is an extension that works alongside Python in Visual Studio Code to provide performant language support. 
    Under the hood, Pylance is powered by Pyright, Microsoft's static type checking tool. 
    
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd
import openai

load_dotenv()
try:
    my_api_key = os.environ['OPENAI_API_KEY']          
    client = OpenAI(api_key=my_api_key)
except Exception as ex:
    print("Exception:", ex)
    exit()


from pydantic import BaseModel

class MyCalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday"},
    ],
    
    # This could be any Class defention
    response_format=MyCalendarEvent,
)

event = completion.choices[0].message.parsed
print(event) # name='Science Fair' date='Friday' participants=['Alice', 'Bob']




