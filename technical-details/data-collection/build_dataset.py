import requests
import pandas as pd
import json
import fitz  
import pdfplumber
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
from difflib import SequenceMatcher


with open('technical-details/data-collection/config.json') as f:
    keys = json.load(f)
API_KEY = keys['fdaapi']

BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


pdf_path = "IPNI Estimates of Nutrient Uptake and Removal.pdf"  
with pdfplumber.open(pdf_path) as pdf:
    table_count = 1
    for page_num, page in enumerate(pdf.pages, start=1):
        # Extract tables from the page
        tables = page.extract_tables()
        
        for table in tables:
            if not table:  
                continue
            
            df = pd.DataFrame(table[1:], columns=table[0])
            
            df.columns = [str(col).strip() if col is not None else f"Column_{i}" for i, col in enumerate(df.columns)]
            df.fillna("N/A", inplace=True)  
            
            output_csv = f"table_page{page_num}_table{table_count}.csv"
            df.to_csv(output_csv, index=False)
            print(f"Saved table to {output_csv}")
            
            table_count += 1

#SOURE : https://insights-engine.refed.org/food-waste-monitor?break_by=destination&indicator=tons-surplus&view=detail&year=2018
foodwaste = pd.read_csv("data/raw-data/ReFED_US_Food_Surplus_Detail.csv")
foodwaste.columns
foodwaste["food_category"].nunique()

foodwaste['food_name'] = foodwaste['food_category'].where(foodwaste['food_category'] != "Not Applicable", foodwaste['food_type'])

def fetch_food_data(df, page_size=12):
    all_results = [] 
    if 'food_name' not in df.columns or df['food_name'].empty:
        raise ValueError("DataFrame must contain a non-empty 'food_name' column")

    for food_name in df['food_name'].unique():
        params = {
            'query': str(food_name),
            'api_key': API_KEY,
            'pageSize': page_size
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            results = response.json()
            if 'foods' in results:
                all_results.extend(results['foods'])
        else:
            print(f"Error for {food_name}: {response.status_code}")

    return all_results


def process_food_data(data):
    food_items = []

    if data:
        for food in data:
            if 'servingSizeUnit' in food and food['servingSizeUnit']:
                food_info = {
                    'food_name': food.get('description', ''),
                    'fdc_id': food.get('fdcId', ''),
                    'brand': food.get('brandOwner', ''),
                    'food_category': food.get('foodCategory', ''),
                    'market_country': food.get('marketCountry', ''),
                    'serving_size': food.get('servingSize', 0),
                    'serving_size_unit': food.get('servingSizeUnit', ''),
                }

                for nutrient in food.get('foodNutrients', []):
                    nutrient_name = nutrient.get('nutrientName', '')
                    nutrient_value = nutrient.get('value', 0)

                    if nutrient_name == 'Energy':  # Calories
                        food_info['calories'] = nutrient_value
                    elif nutrient_name == 'Protein':
                        food_info['protein'] = nutrient_value
                    elif nutrient_name == 'Total lipid (fat)':  # Fat
                        food_info['fat'] = nutrient_value
                    elif nutrient_name == 'Carbohydrate, by difference':  # Carbs
                        food_info['carbs'] = nutrient_value
                    elif nutrient_name == 'Fiber, total dietary':  # Fiber
                        food_info['fiber'] = nutrient_value
                    elif nutrient_name == 'Sugars, total':  # Sugar
                        food_info['sugar'] = nutrient_value
                    elif nutrient_name == 'Vitamin A, IU':  # Vitamin A (IU)
                        food_info['vitamin_a_iu'] = nutrient_value
                    elif nutrient_name == 'Vitamin C, total ascorbic acid':  # Vitamin C (mg)
                        food_info['vitamin_c_mg'] = nutrient_value
                    elif nutrient_name == 'Cholesterol':  # Cholesterol (mg)
                        food_info['cholesterol_mg'] = nutrient_value
                    elif nutrient_name == 'Fatty acids, total saturated':  # Saturated Fat (g)
                        food_info['saturated_fat_g'] = nutrient_value
                    elif nutrient_name == 'Calcium, Ca':  # Calcium (mg)
                        food_info['calcium'] = nutrient_value
                    elif nutrient_name == 'Iron, Fe':  # Iron (mg)
                        food_info['iron'] = nutrient_value
                    elif nutrient_name == 'Sodium, Na':  # Sodium (mg)
                        food_info['sodium'] = nutrient_value
                    elif nutrient_name == 'Potassium, K':  # Potassium (mg)
                        food_info['potassium'] = nutrient_value
                    elif nutrient_name == 'Magnesium, Mg':  # Magnesium (mg)
                        food_info['magnesium'] = nutrient_value
                    elif nutrient_name == 'Phosphorus, P':  # Phosphorus (mg)
                        food_info['phosphorus'] = nutrient_value

                    food_info['percent_daily_value'] = nutrient.get('percentDailyValue', 0)

                food_info['microbes'] = ', '.join(str(microbe) for microbe in food.get('microbes', [])) if food.get('microbes') else 'No microbes listed'
                
                food_info['allergens'] = ', '.join(str(allergen) for allergen in food.get('allergens', ['None'])) if food.get('allergens') else 'None'

                food_info['additives'] = ', '.join(str(additive) for additive in food.get('additives', [])) if food.get('additives') else 'None'

                food_info['labels'] = ', '.join(food.get('labels', ['None'])) if food.get('labels') else 'None'

                food_info['nutrient_group'] = food.get('nutrientGroup', 'Unknown')
                food_info['brand_name'] = food.get('brandOwner', 'Unknown')
                food_info['food_labels'] = ', '.join(food.get('labels', ['None'])) if food.get('labels') else 'None'
                food_info['package_size'] = food.get('packageSize', 'Not available')
                food_info['food_safety_info'] = food.get('foodSafetyInfo', 'No safety info available')
                food_info['expiration_date'] = food.get('expirationDate', 'Not available')
                food_info['country_of_origin'] = food.get('countryOfOrigin', 'Not specified')

                food_items.append(food_info)

    if food_items:
        food_df = pd.DataFrame(food_items)
    else:
        food_df = pd.DataFrame()

    return food_df
#example1 = {'list1' : ["apple"]}



if food_data:
    food_df = process_food_data(food_data)
    food_df.to_csv("foods_data3.csv", index=False)
else:
    print("No food data was fetched.")
    

food_data =  pd.read_csv('data/raw-data/foods_data.csv')
food_data3 =  pd.read_csv('data/raw-data/foods_data3.csv')
food_data = pd.concat([food_data, food_data3], axis=0, ignore_index=True)
food_waste = pd.read_csv('data/raw-data/ReFED_US_Food_Surplus_Detail.csv')

food_data.head()
food_data.shape
food_waste.head()
food_waste.shape

food_waste.drop(
    ['Unnamed: 0','sector', 'sub_sector', 'sub_sector_category',
       'food_type','surplus_upstream_100_year_mtco2e_footprint',
     'surplus_downstream_100_year_mtco2e_footprint',
     'surplus_total_100_year_mtco2e_footprint',
     'surplus_upstream_100_year_mtch4_footprint',
     'surplus_downstream_100_year_mtch4_footprint',
     'surplus_total_100_year_mtch4_footprint'], 
    axis=1, 
    inplace=True
)

food_data.drop(
    ['microbes', 'allergens', 'additives', 'labels',
       'nutrient_group', 'brand_name', 'food_labels', 'package_size',
       'food_safety_info', 'expiration_date', 'country_of_origin','fdc_id' ],
    axis=1, 
    inplace=True
)

food_waste['food_name'].head(10)
food_data['food_name'].head(10)
food_waste['food_name'] = food_waste['food_name'].str.lower().str.strip()
food_data['food_name'] = food_data['food_name'].str.lower().str.strip()

food_data.columns

food_data['serving_size_unit'].unique()



valid_units = ['g', 'grm', 'gm', 'mlt', 'ml', 'mg']


df_filtered = food_data[food_data['serving_size_unit'].str.lower().isin(valid_units)]

df_filtered.columns

def convert_to_grams(row, column):
    unit = row['serving_size_unit'].lower()
    
    # If unit is grams (g, grm, gm), return the value as is
    if unit in ['g', 'grm', 'gm']:
        return row[column]
    # If unit is milliliters (ml, mlt), assume 1 ml = 1 g (for liquids)
    elif unit in ['ml', 'mlt']:
        return row[column]  # Convert 1 ml to 1 g
    # If unit is milligrams (mg), convert to grams
    elif unit == 'mg':
        return row[column] * 0.001  # 1 mg = 0.001 g
    # If unit is unknown or unsupported, return the value as is
    else:
        return row[column]

# List of columns to convert
columns_to_convert = [
    'serving_size', 'protein', 'percent_daily_value', 'fat', 'carbs', 'calories', 
    'fiber'
]

# Apply conversion to each specified column
for column in columns_to_convert:
    df_filtered[column] = df_filtered.apply(lambda row: convert_to_grams(row, column), axis=1)

def convert_to_grams(row):
    # Convert Vitamin A (IU to grams)
    vitamin_a_grams = row['vitamin_a_iu'] * 0.0000003 if pd.notnull(row['vitamin_a_iu']) else None
    
    # Convert Vitamin C (mg to grams)
    vitamin_c_grams = row['vitamin_c_mg'] * 0.001 if pd.notnull(row['vitamin_c_mg']) else None
    
    # Convert Cholesterol (mg to grams)
    cholesterol_grams = row['cholesterol_mg'] * 0.001 if pd.notnull(row['cholesterol_mg']) else None
    
    return pd.Series({'vitamin_a_grams': vitamin_a_grams, 'vitamin_c_grams': vitamin_c_grams, 'cholesterol_grams': cholesterol_grams})

# Apply the conversion function to the DataFrame
df_filtered[['vitamin_a_grams', 'vitamin_c_grams', 'cholesterol_grams']] = df_filtered.apply(convert_to_grams, axis=1)

# Drop the original columns
df_filtered = df_filtered.drop(columns=['vitamin_a_iu', 'vitamin_c_mg', 'cholesterol_mg'])

df_filtered = df_filtered.drop(columns=['brand', 'food_category', 'market_country'])

df_filtered.columns
df_filtered.head()

def mean_excluding_nulls(series):
    return series.dropna().mean()


df_grouped = df_filtered.groupby('food_name').agg({
    'serving_size': mean_excluding_nulls,
    'protein': mean_excluding_nulls,
    'percent_daily_value': mean_excluding_nulls,
    'fat': mean_excluding_nulls,
    'carbs': mean_excluding_nulls,
    'calories': mean_excluding_nulls,
    'fiber': mean_excluding_nulls,
    'calcium': mean_excluding_nulls,
    'iron': mean_excluding_nulls,
    'potassium': mean_excluding_nulls,
    'sodium': mean_excluding_nulls,
    'phosphorus': mean_excluding_nulls
}).reset_index()


food_waste
df_grouped

def calculate_overlap(str1, str2):
    matcher = SequenceMatcher(None, str1.lower(), str2.lower())
    return matcher.ratio()  
def find_best_match(food_name, df, used_matches):
    best_match = None
    highest_overlap = 0
    best_index = -1
    
    
    for idx, name2 in df['food_name'].items():  
        if idx not in used_matches:
            overlap = calculate_overlap(food_name, name2)
            if overlap > highest_overlap:
                best_match = name2
                highest_overlap = overlap
                best_index = idx
    
    return best_match, highest_overlap, best_index
matches = []
used_matches = set()

for name1 in food_waste['food_name']:
    best_match, score, best_index = find_best_match(name1, df_grouped, used_matches)
    if best_match and score >= 0.60:  
        matches.append({
            'food_name_food_waste': name1, 
            'best_match': best_match, 
            'score': score
        })
        used_matches.add(best_index) 

matches_df = pd.DataFrame(matches)

food_waste_renamed = food_waste.rename(columns={'food_name': 'food_name_food_waste'})

result = pd.merge(food_waste_renamed, matches_df, left_on='food_name_food_waste', right_on='food_name_food_waste', how='left')


result = pd.merge(result, df_grouped, left_on='best_match', right_on='food_name', how='left', suffixes=('_food_waste', '_grouped_df'))

print(result)
columns_to_convert = ['calcium', 'iron', 'potassium', 'sodium', 'phosphorus']

#convert to grams
for col in columns_to_convert:
    if col in result.columns:
        result[col] = result[col] / 1000
        
result.to_csv('data/processed-data/food_merged.csv')


result = pd.read_csv('data/processed-data/food_merged.csv')

result.columns
result.describe()