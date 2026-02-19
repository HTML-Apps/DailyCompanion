import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# READ-ME: activate conda
# Zeile 467 beachten - zusätzliche Daten einblenden

# --- COLAB EINSTELLUNGEN einkommentieren---
# %matplotlib inline
# plt.rcParams['figure.figsize'] = [15, 7]

# 1. Daten laden 
filename = 'daily_entries_export.json' 
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

binary_map = {'Ja': 1, 'Nein': 0, 'Gemacht': 1, 'Physio': 1, 'Nicht gemacht': 0}
# Beachte: Die spezifischen Sport- und Schmerzmittelnamen sind hier nicht mehr nötig,
# da wir eine andere Logik für die Listen anwenden.
# 'Kein Sport': 0, 'Fitnessstudio': 1, 'Fußball': 1, 'Radfahren': 1,
# 'Keine': 0, 'IBU': 1,'Etoricoxib_60': 1,'Etoricoxib_120': 1,'Biologika': 1


# --- Spalten, die einfache String-Werte enthalten ---
for col in ['glutenStatus', 'milkStatus', 'stretchingStatus']:
    if col in df.columns:
        # Sicherstellen, dass die Spalte nicht leer ist, bevor map() aufgerufen wird
        # .fillna('') hilft, TypeError bei NaN zu vermeiden, falls map() auf NaN trifft und es kein Schlüssel im dict ist
        df[col + '_num'] = df[col].fillna('').map(binary_map)

# --- Spalten, die Listen enthalten (selectedSports, selectedPainmeds) ---
# Hier müssen wir eine andere Logik anwenden, um eine binäre 0/1 Spalte zu erstellen.

# Für 'selectedSports': 1, wenn Sport gemacht wurde (d.h., die Liste ist nicht leer UND enthält nicht nur 'Kein Sport')
if 'selectedSports' in df.columns:
    df['sport_binary'] = df['selectedSports'].apply(
        lambda x: 1 if isinstance(x, list) and len(x) > 0 and 'Kein Sport' not in x else 0
    )

# Für 'selectedPainmeds': 1, wenn Schmerzmittel genommen wurden (d.h., die Liste ist nicht leer UND enthält nicht nur 'Keine')
if 'selectedPainmeds' in df.columns:
    df['meds_binary'] = df['selectedPainmeds'].apply(
        lambda x: 1 if isinstance(x, list) and len(x) > 0 and 'Keine' not in x else 0
    )

def parse_sleep(s):
    try:
        if not isinstance(s, str): return None
        parts = s.split()
        return int(parts[1]) + int(parts[3])/60
    except (AttributeError, ValueError, IndexError): # Mehr Fehler abfangen
        return None

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

# ------------------------------------------------------------------
# ANALYSE 1.5: Bewegungsöffnung / Flow - TREND MIT REGRESSIONSGERADE
# ------------------------------------------------------------------

# WICHTIG: Erstelle ein temporäres DataFrame OHNE NaN-Werte für diese spezifische Regression
# Das verhindert Fehler, wenn nur wenige Datenpunkte vorhanden sind.
df_movement = df.dropna(subset=['movementFlowPainRating', 'days_since_start']).copy()

# Stelle sicher, dass genügend Datenpunkte vorhanden sind
if len(df_movement) >= 2: # Mindestens 2 Punkte für eine Linie
    # Berechnung der Regressionsgeraden (y = mx + b)
    m, b = np.polyfit(df_movement['days_since_start'], df_movement['movementFlowPainRating'], 1)
    df_movement['regression_line'] = m * df_movement['days_since_start'] + b

    plt.figure(figsize=(15, 7))

    # Die tatsächlichen Werte
    plt.scatter(df_movement['date'], df_movement['movementFlowPainRating'], alpha=0.3, label='Tageswert (Bewegungsöffnung / Flow)', color='gray')

    # Der gleitende Durchschnitt (7 Tage)
    # Hier solltest du den gleitenden Durchschnitt ebenfalls auf df_movement anwenden
    df_movement['rolling_avg'] = df_movement['movementFlowPainRating'].rolling(window=7, min_periods=1).mean() # min_periods=1 für Startwerte
    plt.plot(df_movement['date'], df_movement['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)

    # Die Regressionsgerade
    plt.plot(df_movement['date'], df_movement['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
             color='red', linestyle='--', linewidth=3)

    plt.title('Langzeit-Trend der Bewegungsöffnung / Flow (10 = schmerzfrei)')
    plt.ylabel('Bewegungsöffnung / Flow (1-10)')
    plt.grid(True, alpha=0.2)
    plt.legend()

    # Interpretation der Steigung ausgeben
    if m > 0:
        status = "Verbesserung"
        trend_color = "grün"
    else:
        status = "Verschlechterung"
        trend_color = "rot"

    print(f"\n--- TREND-ANALYSE (Bewegungsöffnung / Flow) ---") # Titel präzisiert
    print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
    print(f"Das bedeutet, pro Tag verändert sich deine/deinen Bewegungsöffnung / Flow im Schnitt um {m:.4f} Punkte.")

    plt.show()
else:
    print(f"\n--- TREND-ANALYSE (Bewegungsöffnung / Flow) ---")
    print("Nicht genügend Datenpunkte (mind. 2) für 'movementFlowPainRating' vorhanden, um Regression zu berechnen.")
    # Optional: Trotzdem einen leeren Plot oder eine Meldung anzeigen
    # plt.figure(figsize=(15, 7))
    # plt.title('Nicht genügend Daten für Bewegungsöffnung / Flow Trend')
    # plt.show()

# ------------------------------------------------------------------
# ANALYSE 1.6: Immunsystem - TREND MIT REGRESSIONSGERADE
# ------------------------------------------------------------------
df_immune = df.dropna(subset=['immunesystemRating', 'days_since_start']).copy()

if len(df_immune) >= 2:
    m, b = np.polyfit(df_immune['days_since_start'], df_immune['immunesystemRating'], 1)
    df_immune['regression_line'] = m * df_immune['days_since_start'] + b

    plt.figure(figsize=(15, 7))
    plt.scatter(df_immune['date'], df_immune['immunesystemRating'], alpha=0.3, label='Tageswert (Immunsystem)', color='gray')
    df_immune['rolling_avg'] = df_immune['immunesystemRating'].rolling(window=7, min_periods=1).mean()
    plt.plot(df_immune['date'], df_immune['rolling_avg'], label='7-Tage-Trend', color='green', linewidth=2, alpha=0.7)
    plt.plot(df_immune['date'], df_immune['regression_line'], label=f'Lineare Regression (Steigung: {m:.4f})', 
             color='red', linestyle='--', linewidth=3)
    plt.title('Langzeit-Trend des Immunsystems (10 = schmerzfrei)')
    plt.ylabel('Immunsystem (1-10)')
    plt.grid(True, alpha=0.2)
    plt.legend()

    if m > 0:
        status = "Verbesserung"
        trend_color = "grün"
    else:
        status = "Verschlechterung"
        trend_color = "rot"

    print(f"\n--- TREND-ANALYSE (Immunsystem) ---")
    print(f"Die statistische Tendenz zeigt eine {status} (Steigung: {m:.4f}).")
    print(f"Das bedeutet, pro Tag verändert sich dein Immunsystem im Schnitt um {m:.4f} Punkte.")

    plt.show()
else:
    print(f"\n--- TREND-ANALYSE (Immunsystem) ---")
    print("Nicht genügend Datenpunkte (mind. 2) für 'immunesystemRating' vorhanden, um Regression zu berechnen.")
    # plt.figure(figsize=(15, 7))
    # plt.title('Nicht genügend Daten für Immunsystem Trend')
    # plt.show()

# ---------------------------------------------------------
# ANALYSE 1.7: Gesamtbilanz - TREND MIT REGRESSIONSGERADE
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
# ANALYSE 5: Korrelations-Heatmap
# ---------------------------------------------------------
    
# 1. Daten laden
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Datenaufbereitung (Binarisierung & Konvertierung)
df['sport_binary'] = df['selectedSports'].apply(lambda x: 1 if x and 'Kein Sport' not in x else 0)
df['meds_binary'] = df['selectedPainmeds'].apply(lambda x: 1 if x and 'Keine' not in x else 0)

binary_map = {'Ja': 1, 'Nein': 0, 'Gemacht': 1, 'Physio' : 1, 'Nicht gemacht': 0}
for col in ['glutenStatus', 'milkStatus', 'stretchingStatus']:
    if col in df.columns:
        df[col + '_num'] = df[col].map(binary_map)

# 3. TIME-LAG: Schmerz von morgen
df['overallAverage_tomorrow'] = df['overallAverage'].shift(-1)

# 4. Spalten auswählen: Bei besserer Datenlage hinzufügen: 'movementFlowPainRating' und 'immunesystem'
cols_to_analyze = [
    'moodRating', 
    'neckPainRating', 
    'shoulderPainRating', 
    'upperBodyPainRating', 
    'lowerBodyPainRating',
    'overallAverage',
    'overallAverage_tomorrow',
    'movementFlowPainRating',
    'immunesystemRating',
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
    'overallAverage_tomorrow': 'Gesamtschmerz Morgen',
    'movementFlowPainRating': 'Bewegungsöffnung / Flow',
    'immunesystemRating':'Immunsystem'
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

# ---------------------------------------------------------
# ANALYSE 6: K-Means Cluster-Analyse
# ---------------------------------------------------------

# WICHTIG: --- Cluster Profile (Durchschnittswerte pro Gruppe) --- in Terminal interpretieren (Werte*100 = Prozent)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- NEU: Integration deiner Top-Tags ---
top_tags = [
    "Bosch-Kantine", "Schokolade", "Eis", "Müsli", "Stuhl", "Shake", "Proteinriegel", 
    "Bauch gespannt", "Bauch gebläht", "Nackensperre", "Atemsperre", "Beinsperre", "Nierenschmerz", "Zeh links", "Zeh rechts", "Zähne",
    "Erreger", "Allergie", "Creme Arme", "Creme Füße", "Creme Hüfte", "Gläschen", "Vitamin D3", "Omega3", "Mouthtape", "Biologika Spritze"
]

# Tags in binäre Spalten umwandeln (0 oder 1)
tag_features = []
for tag in top_tags:
    # Sauberen Spaltennamen erstellen (z.B. tag_bauch_gebläht)
    col_name = f"tag_{tag.replace(' ', '_').replace('-', '_').lower()}"
    df[col_name] = df['selectedNotesTags'].apply(
        lambda x: 1 if isinstance(x, list) and tag in x else 0
    )
    tag_features.append(col_name)

# 1. Auswahl der Features
features = [
    'moodRating', 'overallAverage', 'neckPainRating', 'shoulderPainRating', 
    'upperBodyPainRating', 'lowerBodyPainRating', 
    'glutenStatus_num', 'milkStatus_num', 'stretchingStatus_num', 
    'sport_binary', 'meds_binary'
] + tag_features

# --- KORREKTUR START ---
# Anstatt zu löschen, füllen wir fehlende Tag-Werte mit 0
# Nur bei den Kern-Schmerzwerten löschen wir (falls mal ein Tag gar nicht ausgefüllt wurde)
core_metrics = ['overallAverage', 'neckPainRating', 'lowerBodyPainRating']
df = df.dropna(subset=core_metrics) 

# Alle anderen (Tags, Sport, Meds) auf 0 setzen, falls leer
df[features] = df[features].fillna(0)

cluster_df = df.copy()

# 2. Daten skalieren
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df[features])

# 3. K-Means Modell (wir suchen nach 3 Clustern: z.B. Gut, Mittel, Schlecht)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)

# 4. Auswertung der Cluster-Profile
# .T dreht die Tabelle, damit du die vielen Features (Tags) besser untereinander lesen kannst
cluster_summary = cluster_df.groupby('Cluster')[features].mean().T
print("\n--- Cluster Profile (Durchschnittswerte pro Gruppe) ---")
print(cluster_summary)

# 5. Visualisierung
plt.figure(figsize=(12, 6))

# Wir plotten die Zeit gegen den Schmerz und färben nach Clustern
sns.scatterplot(
    data=cluster_df, 
    x='date', 
    y='overallAverage', 
    hue='Cluster', 
    palette='viridis', 
    s=100, 
    style='Cluster'
)

plt.title('Cluster-Analyse inkl. Tags: Automatische Gruppierung deiner Tage', fontsize=15)
plt.ylabel('Gesamtschmerz Score (10 = schmerzfrei)')
plt.xlabel('Datum')
plt.legend(title='Tag-Typ (Cluster)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

"""
Beispiel-Auswertung 16.02.2026

--- Cluster Profile (Durchschnittswerte pro Gruppe) ---
Cluster                       0        1         2
moodRating             7.668712  7.18750  6.753846
overallAverage         7.307975  7.26875  6.280000
neckPainRating         7.006135  7.00000  5.707692
shoulderPainRating     6.957055  7.37500  5.723077
upperBodyPainRating    7.319018  7.18750  6.169231
lowerBodyPainRating    7.588957  7.56250  7.046154
glutenStatus_num       0.012270  0.06250  0.323077
milkStatus_num         0.233129  0.18750  0.430769
stretchingStatus_num   0.288344  0.43750  0.123077
sport_binary           0.638037  0.75000  0.400000
meds_binary            0.122699  0.75000  0.307692
tag_bosch_kantine      0.085890  0.00000  0.169231
tag_schokolade         0.073620  0.00000  0.215385
tag_eis                0.067485  0.00000  0.153846
tag_müsli              0.000000  0.75000  0.000000
tag_stuhl              0.042945  0.00000  0.061538
tag_shake              0.490798  0.68750  0.092308
tag_proteinriegel      0.000000  0.56250  0.000000
tag_bauch_gespannt     0.245399  0.37500  0.200000
tag_bauch_gebläht      0.012270  0.00000  0.046154
tag_nackensperre       0.073620  0.25000  0.215385
tag_atemsperre         0.000000  0.00000  0.015385
tag_beinsperre         0.024540  0.00000  0.015385
tag_nierenschmerz      0.030675  0.00000  0.015385
tag_zeh_links          0.000000  0.00000  0.000000
tag_zeh_rechts         0.000000  0.00000  0.000000
tag_zähne              0.012270  0.00000  0.000000
tag_erreger            0.055215  0.00000  0.123077
tag_allergie           0.000000  0.00000  0.015385
tag_creme_arme         0.466258  0.25000  0.215385
tag_creme_füße         0.490798  0.50000  0.123077
tag_creme_hüfte        0.000000  0.00000  0.000000
tag_gläschen           0.705521  0.12500  0.261538
tag_vitamin_d3         0.184049  0.18750  0.153846
tag_omega3             0.000000  0.00000  0.000000
tag_mouthtape          0.895706  1.00000  0.507692
tag_biologika_spritze  0.000000  0.00000  0.000000

Cluster 0: Der "Clean-Life & Routine" Modus (Dein bester Zustand)
Wohlbefinden: Hier hast du die beste Stimmung (7.67) und die höchste Schmerzfreiheit (7.31).
Ernährung: Fast 0 % Gluten und moderate Milchwerte.
Der "Gläschen-Effekt": Das Tag gläschen ist hier mit 70 % extrem dominant. Es scheint für dich ein absolut sicheres "Safe-Food" zu sein, das mit hoher Schmerzfreiheit korreliert.
Lifestyle: Du nutzt hier fast immer das Mouthtape (89 %) und cremst dich regelmäßig ein.
Interpretation: Das ist dein stabiler Alltag. Wenig Experimente, gute Routine, kaum Schmerzen und fast keine Medikamente (nur 12 %).

Cluster 1: Das "Müsli- & Medikamenten-Hoch" (Die Leistungs-Falle)
Wohlbefinden: Schmerzwerte sind fast so gut wie in Cluster 0, aber...
Der Preis: Du nimmst an 75 % dieser Tage Medikamente.
Die Trigger: Dieses Cluster wird zu 75 % von Müsli und zu 56 % von Proteinriegeln dominiert.
Körper-Feedback: Obwohl der Schmerzwert okay ist, hast du hier den höchsten Wert für "Bauch gespannt" (37 %).
Interpretation: Das sind Tage, an denen du sehr aktiv bist (höchste Sport-Quote: 75 % und bestes Stretching: 43 %). Es scheint, als würdest du die negativen Effekte von Müsli/Proteinriegeln (gespannter Bauch) durch Sport und Medikamente "niederringen". Du bist leistungsfähig, aber dein System steht unter Stress.

Cluster 2: Die "Entzündungs-Abwärtsspirale" (Der Gefahrenbereich)
Wohlbefinden: Dein schlechtester Zustand. Schmerzfreiheit sinkt auf 6.28. Besonders Nacken (5.7) und Schulter (5.7) sind betroffen.
Die Ursachen-Kombi:
Ernährung: Höchste Werte bei Gluten (32 %), Milch (43 %), Schokolade (21 %) und Eis (15 %).
Mechanik: Die niedrigste Stretching-Rate (12 %) und die niedrigste Mouthtape-Quote (50 %).
Immunsystem: Hier sind die meisten Erreger (12 %) im Spiel.
Interpretation: Das ist der "Perfect Storm". Schlechtes Essen (Zucker/Gluten), wenig Bewegung und eventuell ein aufkeimender Infekt führen sofort zu einer Verschlechterung der Bechterew-Symptomatik im Oberkörper.
Die wichtigsten Erkenntnisse für dein Management:
Mouthtape & Nacken: Schau dir den Zusammenhang an: In Cluster 0 (Mouthtape 89 %) ist der Nacken super (7.0). In Cluster 2 (Mouthtape nur 50 %) ist der Nacken schlecht (5.7). Es gibt eine starke Korrelation zwischen nächtlicher Nasenatmung und Nackenentspannung bei dir.

Müsli vs. Gläschen: Dein Körper reagiert völlig unterschiedlich. Das "Gläschen" (Cluster 0) bringt schmerzfreie Ruhe ohne Medikamente. Das "Müsli" (Cluster 1) treibt dich zwar an, benötigt aber Medikamente und spannt den Bauch an.

Die "Bosch-Kantine" (Cluster 2): Hier ist die Quote mit 17 % am höchsten. Die Kombination aus Kantinenessen, Schokolade und wenig Dehnen scheint dein direkter Weg in den Schmerzschub zu sein.

"""
