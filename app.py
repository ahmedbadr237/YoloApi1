from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import io
import yaml
import requests


# Load class names from your YAML file
with open("data (1).yaml", "r") as f:
    class_data = yaml.safe_load(f)

# If names are a list
class_names = class_data["names"]

app = Flask(__name__)
model = YOLO("best (2).pt")  # Load your YOLOv8 classification model
class_id_to_label = class_data["names"]
class_id_to_label




@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')
    
    results = model(image)
    result = results[0]

    class_indices = result.boxes.cls

    names = model.names

    predicted_labels = [names[int(cls_idx)] for cls_idx in class_indices]

    print("Predicted food labels:", predicted_labels)



    result = results[0]

    class_indices = result.boxes.cls

    predicted_labels = [class_id_to_label[int(cls_idx)] for cls_idx in class_indices]

    print("Predicted food labels:", predicted_labels)

    from fuzzywuzzy import process
    predicted_labels = []
    for result in results:
        for box in result.boxes:
            predicted_labels.append(class_id_to_label[int(box.cls)])

    print("Predicted YOLO labels:", predicted_labels)


    def fuzzy_match_food(yolo_label, database_labels):
        best_match = process.extractOne(yolo_label, database_labels)
        if best_match and best_match[1] >= 80:  
            return best_match[0]
        return None  


    database_food_names = list(class_id_to_label.values())


    # Match YOLO labels to database names
    mapped_food_labels = []
    unmatched_labels = []  

    for label in predicted_labels:
        matched_food = fuzzy_match_food(label, database_food_names)
        if matched_food:
            mapped_food_labels.append(matched_food)
        else:
            unmatched_labels.append(label)  


    if unmatched_labels:
        print(f"Some YOLO labels couldn't be matched to the database: {unmatched_labels}")
    else:
        print("All YOLO labels were successfully mapped to database food items.")
    

    print("Mapped food labels to database:", mapped_food_labels)



    API_KEY = 'MzHoV4JvLMZu6xYx7L3v6btjqWsbc3lZ9KL6Lcu8'
    def get_nutrition_info_usda(food_name):

        """ Query USDA API for food's nutritional information """

        try:
            search_url = 'https://api.nal.usda.gov/fdc/v1/foods/search'
            params = {
                'query': food_name,
                'pageSize': 1,
                'api_key': API_KEY
            }

            search_response = requests.get(search_url, params=params)
            search_response.raise_for_status()
            search_data = search_response.json()

            foods = search_data.get('foods')
            if not foods:
                print(f"No USDA results found for '{food_name}'.")
                return None

            fdc_id = foods[0]['fdcId']
            print(f"\n Found USDA match: {foods[0]['description']} (FDC ID: {fdc_id})")

            return fdc_id

        except requests.RequestException as e:
            print(f"API request error: {e}")
            return None


    def get_detailed_nutrition_info(fdc_id):
        
        """ Get nutrition information using FDC ID """
        
        try:
            details_url = f'https://api.nal.usda.gov/fdc/v1/food/{fdc_id}'
            detail_response = requests.get(details_url, params={'api_key': API_KEY})
            detail_response.raise_for_status()
            detail_data = detail_response.json()

            print("\nRaw USDA API response:")
            print(detail_data)

            return detail_data

        except requests.RequestException as e:
            print(f"API request error: {e}")
            return None


    def extract_nutrition_info(detail_data):
        
        """ Extract nutrition data from USDA response """
        
        usda_to_key = {
            'Energy': 'calories',
            'Protein': 'proteins',
            'Total lipid (fat)': 'fat',
            'Carbohydrate, by difference': 'carbohydrates',
            'Total Sugars': 'sugars',
            'Fiber, total dietary': 'fiber',
            'Fatty acids, total saturated': 'saturated_fat',
            'Cholesterol': 'cholesterol',
            'Sodium, Na': 'sodium',
            'Potassium, K': 'potassium'
        }

        nutrition_info = {
            'food_name': detail_data.get('description', ''),
            'calories': None,
            'carbohydrates': None,
            'sugars': None,
            'fiber': None,
            'proteins': None,
            'fat': None,
            'saturated_fat': None,
            'sodium': None,
            'potassium': None,
            'cholesterol': None
        }

        # 1. Process the foodNutrients field
        for nutrient in detail_data.get('foodNutrients', []):
            name = nutrient['nutrient']['name']
            value = nutrient['amount']
            unit = nutrient['nutrient']['unitName']

            if name in usda_to_key and value is not None:
                key = usda_to_key[name]
                nutrition_info[key] = f"{value} {unit}"

        # 2. Process the labelNutrients field
        for key in nutrition_info.keys():
            if nutrition_info[key] is None:  
                label_value = detail_data.get('labelNutrients', {}).get(key, {}).get('value')
                if label_value is not None:
                    nutrition_info[key] = f"{label_value} g"  

        return nutrition_info


    def print_missing_data_warnings(nutrition_info):
        
        """Print warnings for missing data in nutrition information """
        
        for key, value in nutrition_info.items():
            if value is None:
                print(f" Missing data for {key}")

        mapped_food_labels = [matched_food]  

    all_nutrition_data = []  # Create a list to store all nutrition data
    
    for label in mapped_food_labels:
        fdc_id = get_nutrition_info_usda(label)
        if fdc_id:
            detail_data = get_detailed_nutrition_info(fdc_id)
            if detail_data:
                nutrition_info = extract_nutrition_info(detail_data)
                all_nutrition_data.append({
                    'food_label': label,
                    'nutrition_data': nutrition_info
                })
            else:
                all_nutrition_data.append({
                    'food_label': label,
                    'error': 'Could not retrieve detailed data'
                })
        else:
            all_nutrition_data.append({
                'food_label': label,
                'error': 'Could not retrieve data'
            })

    if results[0].boxes.cls.numel() == 0:
        return jsonify({'error': 'No objects detected'}), 400
    
    class_id = int(results[0].boxes.cls[0].item())
    class_name = class_names[class_id]

    # Format nutrition data as plain text
    output_text = []
    # output_text.append(f"Detected Food: {class_name}\n")
    # output_text.append("-" * 50)  # Separator line

    for item in all_nutrition_data:
        if 'error' in item:
            output_text.append(f"\nFood: {item['food_label']}")
            output_text.append(f"Status: {item['error']}\n")
        else:
            nutrition = item['nutrition_data']
            output_text.append(f"\nFood Name: {class_name}")
            output_text.append("\nNutritional Information:")
            output_text.append(f"• Calories: {nutrition['calories']}")
            
            output_text.append("\nMacronutrients:")
            output_text.append(f"• Proteins: {nutrition['proteins']}")
            output_text.append(f"• Carbohydrates: {nutrition['carbohydrates']}")
            output_text.append(f"• Fats: {nutrition['fat']}")
            
            output_text.append("\nDetailed Information:")
            output_text.append(f"• Dietary Fiber: {nutrition['fiber']}")
            output_text.append(f"• Sugars: {nutrition['sugars']}")
            output_text.append(f"• Saturated Fat: {nutrition['saturated_fat']}")
            output_text.append(f"• Cholesterol: {nutrition['cholesterol']}")
            
            output_text.append("\nMinerals:")
            output_text.append(f"• Sodium: {nutrition['sodium']}")
            output_text.append(f"• Potassium: {nutrition['potassium']}")
            output_text.append("\n" + "-" * 50)  # Separator line

    # Join all lines with newlines and return as plain text
    return "\n".join(output_text), 200, {'Content-Type': 'text/plain'}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
