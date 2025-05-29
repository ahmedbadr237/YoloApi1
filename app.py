# Ensure we have all needed modules and avoid unsupported ones in restricted environments
import sys
import os
import re
import json
import io
import yaml
import requests
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

# Load class names from YAML file
with open("data.yaml", "r") as f:
    class_data = yaml.safe_load(f)
    class_names = class_data["names"]

# Load local Egyptian nutrition database
with open("egyptian_nutrition_data.json", "r", encoding="utf-8") as f:
    egyptian_nutrition_data = json.load(f)

# Ensure all local food entries include 'food_name'
for food, data in egyptian_nutrition_data.items():
    if 'food_name' not in data:
        data['food_name'] = food

# Initialize Flask and YOLO model
app = Flask(__name__)
model = YOLO("best.pt")

# API Configuration
API_KEY = 'MzHoV4JvLMZu6xYx7L3v6btjqWsbc3lZ9KL6Lcu8' # USDA API Key

# USDA fallback methods
def get_nutrition_info_usda(food_name):
    try:
        search_url = 'https://api.nal.usda.gov/fdc/v1/foods/search'
        params = {
            'query': food_name,
            'pageSize': 1,
            'api_key': API_KEY
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        foods = response.json().get('foods')
        return foods[0]['fdcId'] if foods else None
    except requests.RequestException as e:
        print(f"USDA API Error: {e}")
        return None

def get_detailed_nutrition_info(fdc_id):
    try:
        url = f'https://api.nal.usda.gov/fdc/v1/food/{fdc_id}'
        response = requests.get(url, params={'api_key': API_KEY})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"USDA Detail Error: {e}")
        return None

def extract_nutrition_info(detail_data):
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

    nutrition_info = {key: None for key in [
        'food_name', 'calories', 'carbohydrates', 'sugars', 'fiber',
        'proteins', 'fat', 'saturated_fat', 'sodium', 'potassium', 'cholesterol'
    ]}
    nutrition_info['food_name'] = detail_data.get('description', 'unknown')

    for nutrient in detail_data.get('foodNutrients', []):
        name = nutrient.get('nutrient', {}).get('name')
        value = nutrient.get('amount')
        unit = nutrient.get('nutrient', {}).get('unitName')
        if name in usda_to_key and value is not None:
            key = usda_to_key[name]
            nutrition_info[key] = f"{value} {unit}"

    for key in nutrition_info:
        if nutrition_info[key] is None:
            label_val = detail_data.get('labelNutrients', {}).get(key, {}).get('value')
            if label_val is not None:
                nutrition_info[key] = f"{label_val} g"

    return nutrition_info

# Add dietary guidelines
dietary_guidelines = {
    "diabetes": {
        "sugar_limit_g": 20,
        "carb_limit_g": 130
    },
    "hypertension": {
        "sodium_limit_mg": 100 ################################################################################################# Normal 300 
    }
}

# Add portion size constants
PORTION_SCALES = {
    'small': 0.5,
    'medium': 1.0,
    'large': 1.5
}

# Add helper functions before the main route
def extract_numeric_value(text):
    """Extract numerical value from nutrition text"""
    try:
        # Handle cases like '100 mg' or '5 g'
        match = re.match(r"([\d.]+)", str(text))
        if match:
            return float(match.group(1))
        return 0.0
    except (ValueError, TypeError):
        return 0.0

def generate_recommendation(nutrition_info, user_profile):
    """Generate dietary recommendations based on health conditions"""
    recs = []
    conditions = user_profile.get("health_conditions", [])
    # conditions = ["diabetes"] # For testing purposes
    # If user is normal (no diagnosis), show suitable message
    if conditions == ["Normal"] or conditions == []:
        return ["✅ This food is suitable based on your health profile."]

    for cond in conditions:
        if cond == "diabetes":
            sugar = extract_numeric_value(nutrition_info.get("sugars", "0 g"))
            carbs = extract_numeric_value(nutrition_info.get("carbohydrates", "0 g"))

            if sugar >= dietary_guidelines["diabetes"]["sugar_limit_g"]:
                recs.append("⚠️ High sugar content. Consider a lower-sugar option.")
            if carbs >= dietary_guidelines["diabetes"]["carb_limit_g"]:
                recs.append("⚠️ Carbohydrate intake is high. Try portion control.")

        if cond == "hypertension":
            sodium = extract_numeric_value(nutrition_info.get("sodium", "0 mg"))

            if sodium >= dietary_guidelines["hypertension"]["sodium_limit_mg"]:
                recs.append("⚠️ High sodium content. Consider a low-sodium alternative.")

    # Only show the suitable message if no warnings were added
    return recs if recs else ["✅ This food is suitable based on your health profile."]

def scale_nutrition_info_by_portion(nutrition_info, portion_size='medium'):
    """Scale nutrition values based on portion size"""
    scale = PORTION_SCALES.get(portion_size.lower(), 1.0)
    scaled_info = {'food_name': nutrition_info.get('food_name', 'Unknown')}

    for key, value in nutrition_info.items():
        if key == 'food_name':
            continue
        if value is None:
            scaled_info[key] = None
        else:
            match = re.match(r"([\d.]+)\s*(\w+)?", str(value))
            if match:
                number = float(match.group(1))
                unit = match.group(2) or ""
                scaled_value = round(number * scale, 2)
                scaled_info[key] = f"{scaled_value} {unit}"
            else:
                scaled_info[key] = value

    return scaled_info

# Add this function after your other helper functions
def is_food_detection_confident(result, confidence_threshold=0.25):
    """Check if the model is confident about detecting food"""
    if result.probs is None:
        return False
    
    top_confidence = float(result.probs.top1conf)
    return top_confidence >= confidence_threshold

# Modify the predict route - replace the existing classification check with this:
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Please upload an image.", 400, {'Content-Type': 'text/plain'}

    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')

        results = model.predict(source=image, verbose=False, task='classify')
        result = results[0]

        # Check if the image contains food with sufficient confidence
        if not is_food_detection_confident(result):
            return "Please enter a food image. The uploaded image doesn't appear to contain food.", 400, {'Content-Type': 'text/plain'}

        top_class_index = result.probs.top1
        top_class_name = result.names[top_class_index]
        food_name = class_names[int(top_class_name)].lower()
        confidence = result.probs.top1conf

        # Try local Egyptian nutrition data
        nutrition_info = egyptian_nutrition_data.get(food_name)

        # If not found, fallback to USDA
        if not nutrition_info:
            fdc_id = get_nutrition_info_usda(food_name)
            if not fdc_id:
                return "Could not find food in any nutrition database.", 400, {'Content-Type': 'text/plain'}
            detail_data = get_detailed_nutrition_info(fdc_id)
            if not detail_data:
                return "Could not retrieve nutritional information.", 400, {'Content-Type': 'text/plain'}
            nutrition_info = extract_nutrition_info(detail_data)

        # Support JSON or form data input for health_conditions and portion_size
        portion_size = 'medium'
        if request.is_json:
            json_data = request.get_json()
            health_conditions = json_data.get("health_conditions", [])
            portion_size = json_data.get("portion_size", 'medium')
        else:
            # Handle comma-separated health conditions from form data
            raw = request.form.get('health_conditions', '')
            health_conditions = [x.strip() for x in raw.split(',') if x.strip()]
            portion_size = request.form.get('portion_size', 'medium')

        # Set default if no conditions provided
        if not health_conditions:
            health_conditions = ["Normal"]

        user_profile = {"health_conditions": health_conditions}

        # Scale nutrition info based on portion size
        scaled_info = scale_nutrition_info_by_portion(nutrition_info, portion_size)

        # Generate recommendations
        recommendations = generate_recommendation(scaled_info, user_profile)

        # Build output
        output_text = []
        output_text.append(f"Detected Food: {food_name.capitalize()}")
        output_text.append(f"Confidence: {confidence:.2%}")
        output_text.append("-" * 50)
        output_text.append(f"Portion Size: {portion_size.capitalize()}")
        output_text.append("-" * 50)

        output_text.append("\nNutritional Information:")
        output_text.append(f"• Calories: {scaled_info.get('calories', 'N/A')}")

        output_text.append("\nMacronutrients:")
        output_text.append(f"• Proteins: {scaled_info.get('proteins', 'N/A')}")
        output_text.append(f"• Carbohydrates: {scaled_info.get('carbohydrates', 'N/A')}")
        output_text.append(f"• Fats: {scaled_info.get('fat', 'N/A')}")

        output_text.append("\nDetailed Information:")
        output_text.append(f"• Dietary Fiber: {scaled_info.get('fiber', 'N/A')}")
        output_text.append(f"• Sugars: {scaled_info.get('sugars', 'N/A')}")
        output_text.append(f"• Saturated Fat: {scaled_info.get('saturated_fat', 'N/A')}")
        output_text.append(f"• Cholesterol: {scaled_info.get('cholesterol', 'N/A')}")

        output_text.append("\nMinerals:")
        output_text.append(f"• Sodium: {scaled_info.get('sodium', 'N/A')}")
        output_text.append(f"• Potassium: {scaled_info.get('potassium', 'N/A')}")
        output_text.append("\n" + "-" * 50)

        output_text.append("\nHealth Recommendations:")
        for rec in recommendations:
            output_text.append(f"• {rec}")
        output_text.append("\n" + "-" * 50)

        return "\n".join(output_text), 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        return f"Error processing image: {str(e)}", 500, {'Content-Type': 'text/plain'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)