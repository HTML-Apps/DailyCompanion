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
%matplotlib inline
plt.rcParams['figure.figsize'] = [15, 7]

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

"""
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

"""
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
# ANALYSE 2: WOCHENTAGSVERGLEICH
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

"""

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
    plt.show()"""

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
    'overallAverage': 'Gesamtschmerz heute',
    'sport_binary': 'Sport',
    'meds_binary': 'Medikamente',
    'stretchingStatus_num': 'Dehnen/Physio',
    'glutenStatus_num': 'Gluteneinnahme',
    'milkStatus_num': 'Milcheinnahme',
    'overallAverage_tomorrow': 'Gesamtschmerz morgen',
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
print(display_matrix['Gesamtschmerz morgen'].sort_values(ascending=False))

# ---------------------------------------------------------
# ANALYSE 6: K-Means Cluster-Analyse
# ---------------------------------------------------------

# WICHTIG: --- Cluster Profile (Durchschnittswerte pro Gruppe) --- in Terminal interpretieren (Werte*100 = Prozent)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- NEU: Dynamische Integration deiner Tags ---
# ... (dieser Teil bleibt gleich wie in der letzten Version) ...
# Extrahiere alle einzigartigen Tags aus der 'selectedNotesTags'-Spalte
all_tags_flat = []
for tags_list in df['selectedNotesTags'].dropna():
    if isinstance(tags_list, list):
        all_tags_flat.extend(tags_list)

top_tags = sorted(list(set(all_tags_flat)))

# print(f"Dynamisch gefundene Tags für Analyse: {top_tags}") # Zur Überprüfung

# Tags in binäre Spalten umwandeln (0 oder 1)
tag_features = []
for tag in top_tags:
    cleaned_tag = "".join([c if c.isalnum() else "_" for c in tag]).lower()
    col_name = f"tag_{cleaned_tag}"
    df[col_name] = df['selectedNotesTags'].apply(
        lambda x: 1 if isinstance(x, list) and tag in x else 0
    )
    tag_features.append(col_name)


# 1. Auswahl der Features
features = [
    'moodRating', 'overallAverage', 'neckPainRating', 'shoulderPainRating', 
    'upperBodyPainRating', 'lowerBodyPainRating', 
    'immunesystemRating', # Jetzt korrekt als Rating behandelt
    'movementFlowPainRating', # Jetzt korrekt als Rating behandelt
    'glutenStatus_num', 'milkStatus_num', 'stretchingStatus_num', 
    'sport_binary', 'meds_binary'
] + tag_features


# --- KORREKTUR START ---
# Wir müssen die Behandlung von NaN-Werten für Ratings und binäre Features trennen.

# 1. Definiere die Rating-Features und die Binär-Features
rating_features = [
    'moodRating', 'overallAverage', 'neckPainRating', 'shoulderPainRating', 
    'upperBodyPainRating', 'lowerBodyPainRating', 
    'immunesystemRating', 'movementFlowPainRating'
]

# Binäre Features sind alle Tag-Features sowie die festen binären
binary_features = [
    'glutenStatus_num', 'milkStatus_num', 'stretchingStatus_num', 
    'sport_binary', 'meds_binary'
] + tag_features


# 2. Schritt: Drop rows where CRITICAL_RATING_FEATURES are NaN
# Nur wenn KEINE der Kern-Ratings vorhanden ist, droppen wir die Zeile.
# Wenn nur immunesystemRating fehlt, aber der Rest da ist, soll die Zeile nicht komplett gelöscht werden.
# Es ist sinnvoller, nur die Zeilen zu löschen, wo wirklich Kern-Features fehlen,
# und andere fehlende Werte (wie immunesystemRating, wenn es nicht immer ausgefüllt wird) zu imputieren.
# Für die Cluster-Analyse ist es besser, so viele Tage wie möglich zu behalten.
# Daher: Dropna nur auf einen kleinen Satz von absoluten Kern-Metriken anwenden.
# Oder gar kein dropna() hier, wenn wir alle Features imputieren wollen.

# Für die Cluster-Analyse ist es am besten, ein eigenes DataFrame zu erstellen und
# fehlende Werte (NaNs) dort zu behandeln.
cluster_df = df.copy()

# Fülle fehlende Werte für RATING_FEATURES mit dem MEAN (Durchschnitt)
# Das ist besser als 0, da 0 ein Extremwert auf der Skala 1-10 ist.
# Wenn du KEINE fehlenden Ratings imputieren willst, müsstest du diese Zeilen droppen.
for col in rating_features:
    # Nur füllen, wenn die Spalte existiert und fehlende Werte hat
    if col in cluster_df.columns and cluster_df[col].isnull().any():
        cluster_df[col] = cluster_df[col].fillna(cluster_df[col].mean())

# Fülle fehlende Werte für BINÄRE_FEATURES mit 0 (Tag nicht vorhanden)
for col in binary_features:
    if col in cluster_df.columns and cluster_df[col].isnull().any():
        cluster_df[col] = cluster_df[col].fillna(0)

# Nun müssen wir sicherstellen, dass die FEATURES-Liste nur Spalten enthält, die auch wirklich im cluster_df sind
# (falls z.B. eine Spalte in der Rohdaten nie auftauchte)
actual_features_for_clustering = [f for f in features if f in cluster_df.columns]

# WICHTIG: Entferne Zeilen mit NaN, die nach der Imputation noch existieren könnten (unwahrscheinlich, aber sicher ist sicher)
# Dies ist relevant, falls z.B. der Mittelwert nicht berechnet werden konnte, weil die Spalte komplett NaN war.
# Aber nach dem fillna() sollten keine NaNs mehr in den relevanten Spalten sein.
cluster_df = cluster_df.dropna(subset=actual_features_for_clustering)

# Stelle sicher, dass genügend Datenpunkte für die Cluster-Analyse vorhanden sind
if len(cluster_df) < 3: # KMeans benötigt mindestens n_clusters Datenpunkte
    print("Nicht genügend Datenpunkte für die Cluster-Analyse nach NaN-Behandlung.")
    print(cluster_summary) # Oder eine andere Fallback-Aktion
else:
    # 2. Daten skalieren (NUR die tatsächlichen Features, die verwendet werden)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df[actual_features_for_clustering])

    # 3. K-Means Modell (wir suchen nach 3 Clustern: z.B. Gut, Mittel, Schlecht)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)

    # 4. Auswertung der Cluster-Profile
    # .T dreht die Tabelle, damit du die vielen Features (Tags) besser untereinander lesen kannst
    # Hier verwenden wir wieder die "features" Liste, aber stellen sicher, dass alle drin sind
    cluster_summary = cluster_df.groupby('Cluster')[actual_features_for_clustering].mean().T
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

# Auswertung der Cluster-Analyse mithilfe der KI
