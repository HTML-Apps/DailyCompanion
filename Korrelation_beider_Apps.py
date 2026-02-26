import pandas as pd
import json
from datetime import datetime, timedelta

# --- 1. Lade die JSON-Dateien ---
try:
    with open('daily_entries_export.json', 'r', encoding='utf-8') as f:
        symptoms_data = json.load(f)
    with open('wochenplan_data.json', 'r', encoding='utf-8') as f:
        plan_data = json.load(f)
    with open('speisekammer_data.json', 'r', encoding='utf-8') as f:
        meals_data = json.load(f)
    with open('rezepte_data.json', 'r', encoding='utf-8') as f:
        recipes_data = json.load(f)
except FileNotFoundError as e:
    print(f"Fehler: Die Datei {e.filename} wurde nicht gefunden. Bitte stellen Sie sicher, dass alle 4 JSON-Dateien hochgeladen wurden.")
else:
    # --- 2. Bereite die Daten auf (DataFrames erstellen) ---

    # 2.1. DataFrame für die Symptome
    symptoms_df = pd.DataFrame(symptoms_data)
    symptoms_df['date'] = pd.to_datetime(symptoms_df['date']).dt.date
    symptoms_df = symptoms_df[['date', 'digestionRating', 'selectedNotesTags']]
    symptoms_df.rename(columns={'digestionRating': 'digestion_rating', 'selectedNotesTags': 'symptom_tags'}, inplace=True)

    # 2.2. DataFrame für die Mahlzeiten und Zutaten
    meal_name_to_id = {meal['name']: meal['id'] for meal in meals_data}
    meal_to_ingredients = {}
    for name, meal_id in meal_name_to_id.items():
        if meal_id in recipes_data and 'zutaten' in recipes_data[meal_id]:
            ingredients = [zutat['name'].strip().lower() for zutat in recipes_data[meal_id]['zutaten'] if zutat.get('name')]
            meal_to_ingredients[name] = ingredients

    days_full = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    base_date = datetime(2026, 1, 12)

    def parse_plan_date(key):
        try:
            parts = key.split('-')
            kw = int(parts[1].replace('KW', ''))
            day_name = parts[2]
            day_index = days_full.index(day_name)
            week_offset = kw - 3
            plan_date = base_date + timedelta(weeks=week_offset, days=day_index)
            return plan_date.date()
        except (ValueError, IndexError):
            return None

    planned_meals = []
    for key, value in plan_data.items():
        date = parse_plan_date(key)
        if date:
            meal_name = value.get('meal') if isinstance(value, dict) else value
            if meal_name and meal_name != "---":
                ingredients = meal_to_ingredients.get(meal_name, [])
                planned_meals.append({'date': date, 'meal': meal_name, 'ingredients': ingredients})

    meals_df = pd.DataFrame(planned_meals)
    meals_df = meals_df.drop_duplicates(subset=['date', 'meal'])

    # --- 3. Führe die Daten mit Zeitverzögerung zusammen ---
    analysis_dfs = []
    for lag_days in [1, 2]:
        temp_meals_df = meals_df.copy()
        temp_meals_df['symptom_date'] = temp_meals_df['date'] + timedelta(days=lag_days)
        merged_df = pd.merge(temp_meals_df, symptoms_df, left_on='symptom_date', right_on='date')
        merged_df['lag_days'] = lag_days
        analysis_dfs.append(merged_df)

    final_df = pd.concat(analysis_dfs, ignore_index=True)
    ingredients_exploded_df = final_df.explode('ingredients').rename(columns={'ingredients': 'ingredient'})
    ingredients_exploded_df.dropna(subset=['ingredient'], inplace=True)
    ingredients_exploded_df = ingredients_exploded_df[ingredients_exploded_df['ingredient'] != '']

    # --- 4. Führe die Korrelationsanalyse durch --- !!!!!! ['Bauch gebläht', 'Bauch gespannt', 'Verdauung', 'Stuhl']
    digestive_symptom_tags = ['gebläht', 'bauch gespannt', 'verdauung', 'stuhl']
    def has_digestive_symptom(tags):
        if not isinstance(tags, list): return False
        return any(tag.lower() in digestive_symptom_tags for tag in tags)

    ingredients_exploded_df['has_symptom'] = ingredients_exploded_df['symptom_tags'].apply(has_digestive_symptom)

    print("Analyse der Korrelationen...\n" + "="*30)

    # A) Durchschnittlicher Verdauungs-Score (1-10) pro Zutat
    avg_digestion_score = ingredients_exploded_df.groupby(['ingredient', 'lag_days'])['digestion_rating'].mean().sort_values().reset_index()
    print("\n--- Top 15 Zutaten mit dem niedrigsten Verdauungs-Score (schlechter) ---\n")
    print(avg_digestion_score.head(15).to_string())

    # B) Wahrscheinlichkeit von Verdauungsproblemen pro Zutat
    symptom_correlation = ingredients_exploded_df.groupby(['ingredient', 'lag_days']).agg(
        times_eaten=('ingredient', 'size'),
        symptom_occurences=('has_symptom', 'sum')
    ).reset_index()
    symptom_correlation['symptom_probability_%'] = (symptom_correlation['symptom_occurences'] / symptom_correlation['times_eaten']) * 100
    symptom_correlation_filtered = symptom_correlation[symptom_correlation['times_eaten'] >= 3].sort_values(by='symptom_probability_%', ascending=False)
    
    print("\n--- Top 15 Zutaten mit der höchsten Wahrscheinlichkeit für Symptome (mind. 3x gegessen) ---\n")
    print(symptom_correlation_filtered.head(15).to_string())

