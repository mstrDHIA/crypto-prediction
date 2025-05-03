import requests
import pandas as pd
import os
from src.utils import date_to_timestamp, timestamp_to_date
from datetime import datetime

def fetch_data_from_api(api_url, params=None, headers=None):
    """
    Fetch data from the API.
    :param api_url: The API endpoint URL.
    :param params: Query parameters for the API request (optional).
    :param headers: Headers for the API request (optional).
    :return: JSON response from the API.
    """
    try:
        data= []
        first_year=2019
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        this_year = datetime.utcfromtimestamp(response.json()['Data']['TimeFrom'])  
        # temp_data = response.json()['Data']['Data']
        temp_data = response.json()
        data.extend(temp_data['Data']['Data'][::-1])


        # print(f"First year: {first_year}")
        # print(f"Current year: {this_year}")
        # print(f"from year: {datetime.utcfromtimestamp(response.json()['Data']['TimeFrom'])}")
        # print(f"to year: {datetime.utcfromtimestamp(response.json()['Data']['TimeTo'])}")
        while this_year.year > first_year:
        #     print(this_year)
            this_year=int(this_year.timestamp())
        #     print(this_year)
            params['toTs'] = this_year
            # print(params)
        #     print(this_year)
            response = requests.get(api_url, params=params, headers=headers)
        #     print(this_year)
            reversed_data = response.json()['Data']['Data'][::-1]
            temp_data = response.json()
            print(reversed_data[0])
            print(reversed_data[-1])
            # pri
            data.extend(temp_data['Data']['Data'][::-1])
        #     print(this_year)
            # print(f"From year: {temp_data['Data']['TimeFrom']}")
            this_year = temp_data['Data']['TimeFrom']
            # print(f"Current year: {this_year}")
            
        #     print(this_year)
        #     params['toTs'] = this_year
        #     print(this_year)
            this_year = datetime.utcfromtimestamp(this_year)
            # print(this_year)
        #     print(this_year)
        #     print(f"aaaaaa {temp_data['Data']['TimeFrom']}")
        # print(data)    
        return data
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def transform_json_to_csv(json_data, output_csv_path):
    """
    Transform JSON data into a CSV file.
    :param json_data: The JSON data to transform.
    :param output_csv_path: The path to save the CSV file.
    """
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data)
        df.drop(columns=['conversionType','conversionSymbol'], inplace=True)
        # Save DataFrame to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Data successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"Error transforming JSON to CSV: {e}")

