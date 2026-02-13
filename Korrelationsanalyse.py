import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Daten laden
filename = 'daily_entries_export_2026-02-13.json' 
try:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Fehler: Die Datei {filename} wurde nicht gefunden.")
    exit()

df = pd.DataFrame(data)

# --- AUFBEREITUNG (SKALA: 10 = SCHMERZFREI) ---
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date')
df['date'] = df['date'].dt.tz_localize(None)

binary_map = {'Ja': 1, 'Nein': 0, 'Gemacht': 1, 'Nicht gemacht': 0}
for col in ['glutenStatus', 'milkStatus', 'stretchingStatus']:
    if col in df.columns:
        df[col + '_num'] = df[col].map(binary_map)

def parse_sleep(s):
    try:
        if not isinstance(s, str): return None
        parts = s.split()
        return int(parts[1]) + int(parts[3])/60
    except: return None

df['sleep_hours'] = df['sleepDuration'].apply(parse_sleep)

# ---------------------------------------------------------
# ANALYSE 1.0: Wohlbefinden - TREND MIT REGRESSIONSGERADE
# ---------------------------------------------------------
# Wir wandeln das Datum in eine Zahl um (Tage seit Beginn), um die Regression zu berechnen
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Berechnung der Regressionsgeraden (y = mx + b)
m, b = np.polyfit(df['days_since_start'], df['moodRating'], 1)
df['regression_line'] = m * df['days_since_start'] + b

plt.figure(figsize=(15, 7))

# Die tatsächlichen Werte
plt.scatter(df['date'], df['moodRating'], alpha=0.3, label='Tageswert (Wohlbefinden)', color='gray')

# Der gleitende Durchschnitt (7 Tage)
df['rolling_avg'] = df['moodRating'].rolling(window=7).mean()
plt.plot(df['date'], df['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

# Die Regressionsgerade
plt.plot(df['date'], df['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
         color='red', linestyle='--', linewidth=3)

plt.title('Langzeit-Trend des Wohlbefindens (10 = schmerzfrei)')
plt.ylabel('Wohlbefinden (1-10)')
plt.grid(True, alpha=0.2)
plt.legend()

# Interpretation der Steigung ausgeben
if m > 0:
    status = "Verbesserung"
    trend_color = "grün"
else:
    status = "Verschlechterung"
    trend_color = "rot"

print(f"\n--- TREND-ANALYSE ---")
print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
print(f"Das bedeutet, pro Tag verändert sich dein Wohlbefinden im Schnitt um {m:.4f} Punkte.")

plt.show()

# ---------------------------------------------------------
# ANALYSE 1.1: Nackenschmerzen - TREND MIT REGRESSIONSGERADE
# ---------------------------------------------------------
# Wir wandeln das Datum in eine Zahl um (Tage seit Beginn), um die Regression zu berechnen
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Berechnung der Regressionsgeraden (y = mx + b)
m, b = np.polyfit(df['days_since_start'], df['neckPainRating'], 1)
df['regression_line'] = m * df['days_since_start'] + b

plt.figure(figsize=(15, 7))

# Die tatsächlichen Werte
plt.scatter(df['date'], df['neckPainRating'], alpha=0.3, label='Tageswert (Nackenschmerz)', color='gray')

# Der gleitende Durchschnitt (7 Tage)
df['rolling_avg'] = df['neckPainRating'].rolling(window=7).mean()
plt.plot(df['date'], df['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

# Die Regressionsgerade
plt.plot(df['date'], df['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
         color='red', linestyle='--', linewidth=3)

plt.title('Langzeit-Trend des Nackenschmerzes (10 = schmerzfrei)')
plt.ylabel('Nackenschmerz (1-10)')
plt.grid(True, alpha=0.2)
plt.legend()

# Interpretation der Steigung ausgeben
if m > 0:
    status = "Verbesserung"
    trend_color = "grün"
else:
    status = "Verschlechterung"
    trend_color = "rot"

print(f"\n--- TREND-ANALYSE ---")
print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
print(f"Das bedeutet, pro Tag verändert sich deinen Nackenschmerz im Schnitt um {m:.4f} Punkte.")

plt.show()

# ---------------------------------------------------------
# ANALYSE 1.2: Schulterschmerzen - TREND MIT REGRESSIONSGERADE
# ---------------------------------------------------------
# Wir wandeln das Datum in eine Zahl um (Tage seit Beginn), um die Regression zu berechnen
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Berechnung der Regressionsgeraden (y = mx + b)
m, b = np.polyfit(df['days_since_start'], df['shoulderPainRating'], 1)
df['regression_line'] = m * df['days_since_start'] + b

plt.figure(figsize=(15, 7))

# Die tatsächlichen Werte
plt.scatter(df['date'], df['shoulderPainRating'], alpha=0.3, label='Tageswert (Schulterschmerzen)', color='gray')

# Der gleitende Durchschnitt (7 Tage)
df['rolling_avg'] = df['shoulderPainRating'].rolling(window=7).mean()
plt.plot(df['date'], df['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

# Die Regressionsgerade
plt.plot(df['date'], df['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
         color='red', linestyle='--', linewidth=3)

plt.title('Langzeit-Trend der Schulterschmerzen (10 = schmerzfrei)')
plt.ylabel('Schulterschmerzen (1-10)')
plt.grid(True, alpha=0.2)
plt.legend()

# Interpretation der Steigung ausgeben
if m > 0:
    status = "Verbesserung"
    trend_color = "grün"
else:
    status = "Verschlechterung"
    trend_color = "rot"

print(f"\n--- TREND-ANALYSE ---")
print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
print(f"Das bedeutet, pro Tag verändert sich deine Schulterschmerzen im Schnitt um {m:.4f} Punkte.")

plt.show()

# ---------------------------------------------------------
# ANALYSE 1.3: Oberkörperschmerzen - TREND MIT REGRESSIONSGERADE
# ---------------------------------------------------------
# Wir wandeln das Datum in eine Zahl um (Tage seit Beginn), um die Regression zu berechnen
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Berechnung der Regressionsgeraden (y = mx + b)
m, b = np.polyfit(df['days_since_start'], df['upperBodyPainRating'], 1)
df['regression_line'] = m * df['days_since_start'] + b

plt.figure(figsize=(15, 7))

# Die tatsächlichen Werte
plt.scatter(df['date'], df['upperBodyPainRating'], alpha=0.3, label='Tageswert (Oberkörperschmerzen)', color='gray')

# Der gleitende Durchschnitt (7 Tage)
df['rolling_avg'] = df['upperBodyPainRating'].rolling(window=7).mean()
plt.plot(df['date'], df['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

# Die Regressionsgerade
plt.plot(df['date'], df['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
         color='red', linestyle='--', linewidth=3)

plt.title('Langzeit-Trend der Oberkörperschmerzen (10 = schmerzfrei)')
plt.ylabel('Oberkörperschmerzen (1-10)')
plt.grid(True, alpha=0.2)
plt.legend()

# Interpretation der Steigung ausgeben
if m > 0:
    status = "Verbesserung"
    trend_color = "grün"
else:
    status = "Verschlechterung"
    trend_color = "rot"

print(f"\n--- TREND-ANALYSE ---")
print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
print(f"Das bedeutet, pro Tag verändert sich deine Oberkörperschmerzen im Schnitt um {m:.4f} Punkte.")

plt.show()

# ---------------------------------------------------------
# ANALYSE 1.4: Unterkörperschmerzen - TREND MIT REGRESSIONSGERADE
# ---------------------------------------------------------
# Wir wandeln das Datum in eine Zahl um (Tage seit Beginn), um die Regression zu berechnen
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Berechnung der Regressionsgeraden (y = mx + b)
m, b = np.polyfit(df['days_since_start'], df['lowerBodyPainRating'], 1)
df['regression_line'] = m * df['days_since_start'] + b

plt.figure(figsize=(15, 7))

# Die tatsächlichen Werte
plt.scatter(df['date'], df['lowerBodyPainRating'], alpha=0.3, label='Tageswert (Unterkörperschmerzen)', color='gray')

# Der gleitende Durchschnitt (7 Tage)
df['rolling_avg'] = df['lowerBodyPainRating'].rolling(window=7).mean()
plt.plot(df['date'], df['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

# Die Regressionsgerade
plt.plot(df['date'], df['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
         color='red', linestyle='--', linewidth=3)

plt.title('Langzeit-Trend der Unterkörperschmerzen (10 = schmerzfrei)')
plt.ylabel('Unterkörperschmerzen (1-10)')
plt.grid(True, alpha=0.2)
plt.legend()

# Interpretation der Steigung ausgeben
if m > 0:
    status = "Verbesserung"
    trend_color = "grün"
else:
    status = "Verschlechterung"
    trend_color = "rot"

print(f"\n--- TREND-ANALYSE ---")
print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
print(f"Das bedeutet, pro Tag verändert sich deine Unterkörperschmerzen im Schnitt um {m:.4f} Punkte.")

plt.show()

# ---------------------------------------------------------
# ANALYSE 1.5: Gesamtbilanz - TREND MIT REGRESSIONSGERADE
# ---------------------------------------------------------
# Wir wandeln das Datum in eine Zahl um (Tage seit Beginn), um die Regression zu berechnen
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Berechnung der Regressionsgeraden (y = mx + b)
m, b = np.polyfit(df['days_since_start'], df['overallAverage'], 1)
df['regression_line'] = m * df['days_since_start'] + b

plt.figure(figsize=(15, 7))

# Die tatsächlichen Werte
plt.scatter(df['date'], df['overallAverage'], alpha=0.3, label='Tageswert (Gesamtbilanz)', color='gray')

# Der gleitende Durchschnitt (7 Tage)
df['rolling_avg'] = df['overallAverage'].rolling(window=7).mean()
plt.plot(df['date'], df['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

# Die Regressionsgerade
plt.plot(df['date'], df['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
         color='red', linestyle='--', linewidth=3)

plt.title('Langzeit-Trend der Gesamtbilanz (10 = schmerzfrei)')
plt.ylabel('Gesamtbilanz (1-10)')
plt.grid(True, alpha=0.2)
plt.legend()

# Interpretation der Steigung ausgeben
if m > 0:
    status = "Verbesserung"
    trend_color = "grün"
else:
    status = "Verschlechterung"
    trend_color = "rot"

print(f"\n--- TREND-ANALYSE ---")
print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
print(f"Das bedeutet, pro Tag verändert sich deine Gesamtbilanz im Schnitt um {m:.4f} Punkte.")

plt.show()

# ---------------------------------------------------------
# ANALYSE 2: WOCHENTAGS-VERGLEICH
# ---------------------------------------------------------
df['weekday'] = df['date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_de = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

plt.figure(figsize=(10, 5))
sns.boxplot(x='weekday', y='overallAverage', data=df, order=weekday_order, palette='viridis')
plt.xticks(range(7), weekday_de)
plt.title('Schmerzverteilung nach Wochentagen')
plt.ylabel('Gesamtschmerz')
plt.show()

# ---------------------------------------------------------
# ANALYSE 3: TAG-AUSWIRKUNG (RANKING DER "GUTEN" TAGE)
# ---------------------------------------------------------
all_tags = []
for tags in df['selectedNotesTags'].dropna():
    if isinstance(tags, list): all_tags.extend(tags)

unique_tags = list(set(all_tags))
tag_impact = []

for tag in unique_tags:
    mask = df['selectedNotesTags'].apply(lambda x: tag in x if isinstance(x, list) else False)
    avg_wellbeing = df[mask]['overallAverage'].mean()
    count = mask.sum()
    if count > 1:
        tag_impact.append({'Tag': tag, 'Schnitt_Wohlbefinden': avg_wellbeing, 'Vorkommen': count})

tag_df = pd.DataFrame(tag_impact).sort_values('Schnitt_Wohlbefinden', ascending=False)

if not tag_df.empty:
    plt.figure(figsize=(10, 8))
    # Grün oben = Tags, die mit Wohlbefinden korrelieren
    sns.barplot(x='Schnitt_Wohlbefinden', y='Tag', data=tag_df, palette='RdYlGn')
    plt.axvline(df['overallAverage'].mean(), color='blue', linestyle='--', label='Gesamtschnitt')
    plt.title('Welche Tags stehen für gute Tage? (Rechts = Besser)')
    plt.xlabel('Durchschnittliches Wohlbefinden (10 = Bestwert)')
    plt.show()

# ---------------------------------------------------------
# ANALYSE 4: SCHLAF VS. SCHMERZ
# ---------------------------------------------------------
if df['sleep_hours'].notnull().any():
    plt.figure(figsize=(8, 6))
    sns.regplot(x='sleep_hours', y='overallAverage', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Zusammenhang: Schlafdauer vs. Schmerz am selben Tag')
    plt.xlabel('Schlafstunden')
    plt.ylabel('Schmerz-Score')
    plt.show()

# ---------------------------------------------------------
# ANALYSE 5: CLUSTER-ANALYSE (TAGESTYPEN)
# ---------------------------------------------------------
cluster_cols = ['neckPainRating', 'shoulderPainRating', 'upperBodyPainRating', 'lowerBodyPainRating']
cluster_labels = ['Nacken', 'Schulter', 'Oberkörper', 'Unterkörper']
cluster_data = df[cluster_cols].dropna()

if len(cluster_data) > 10:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled_data)
    cluster_data['Typ'] = kmeans.labels_
    cluster_profiles = cluster_data.groupby('Typ').mean()

    angles = np.linspace(0, 2 * np.pi, len(cluster_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4'] # Grün, Orange, Blau
    
    for i in range(len(cluster_profiles)):
        values = cluster_profiles.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values, color=colors[i], alpha=0.2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), cluster_labels)
    ax.set_ylim(0, 10) # Feste Skala 1-10
    plt.title('Dein Wohlbefinden nach Körperregionen\n(Größere Fläche = Weniger Schmerz)', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()
    
# ---------------------------------------------------------
# ANALYSE 6: Korrelations-Heatmap
# ---------------------------------------------------------
    
# 1. Daten laden
filename = 'daily_entries_export_2026-02-13.json' 
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Datenaufbereitung (Binarisierung & Konvertierung)
df['sport_binary'] = df['selectedSports'].apply(lambda x: 1 if x and 'Kein Sport' not in x else 0)
df['meds_binary'] = df['selectedPainmeds'].apply(lambda x: 1 if x and 'Keine' not in x else 0)

binary_map = {'Ja': 1, 'Nein': 0, 'Gemacht': 1, 'Nicht gemacht': 0}
for col in ['glutenStatus', 'milkStatus', 'stretchingStatus']:
    if col in df.columns:
        df[col + '_num'] = df[col].map(binary_map)

# 3. TIME-LAG: Schmerz von morgen
df['overallAverage_tomorrow'] = df['overallAverage'].shift(-1)

# 4. Spalten auswählen
cols_to_analyze = [
    'moodRating', 
    'neckPainRating', 
    'shoulderPainRating', 
    'upperBodyPainRating', 
    'lowerBodyPainRating',
    'overallAverage',
    'overallAverage_tomorrow',
    'sport_binary',
    'stretchingStatus_num',
    'meds_binary',
    'glutenStatus_num',
    'milkStatus_num'
]

# Filtern und konvertieren
cols_to_analyze = [c for c in cols_to_analyze if c in df.columns]
corr_df = df[cols_to_analyze].apply(pd.to_numeric)
correlation_matrix = corr_df.corr()

# --- NEU: Übersetzung der Begriffe für die Grafik ---
label_dict = {
    'moodRating': 'Stimmung',
    'neckPainRating': 'Nackenschmerzen',
    'shoulderPainRating': 'Schulterschmerzen',
    'upperBodyPainRating': 'Oberkörperschmerz',
    'lowerBodyPainRating': 'Unterkörperschmerz',
    'overallAverage': 'Gesamtschmerz Heute',
    'sport_binary': 'Sport (Ja/Nein)',
    'meds_binary': 'Medikamente (Ja/Nein)',
    'stretchingStatus_num': 'Dehnen (Ja/Nein)',
    'glutenStatus_num': 'Gluten Heute',
    'milkStatus_num': 'Milch Heute',
    'overallAverage_tomorrow': 'Gesamtschmerz Morgen'
}

# Ersetze die Namen in der Korrelationsmatrix nur für die Anzeige
display_matrix = correlation_matrix.rename(index=label_dict, columns=label_dict)

# 5. Visualisierung
plt.figure(figsize=(14, 10))
sns.heatmap(display_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Morbus Bechterew Analyse: Zusammenhang von Alltag & Symptomen', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. Textausgabe (bleibt zur Sicherheit im Original für Vergleiche)
print("\n--- Korrelationen zum Gesamtschmerz am nächsten Tag ---")
print(display_matrix['Gesamtschmerz Morgen'].sort_values(ascending=False))

# Auswertung für gleitenden Durchschnitt Mood, Schmerzen SChulter, ok, uk, Nacken
# Zusätzliche Skala 1-10 Kategorie für ________
