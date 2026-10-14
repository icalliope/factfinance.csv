import pandas as pd
import numpy as np
import os
import json

base_dir = os.path.dirname(os.path.abspath(__file__))

source_file = os.path.join(base_dir, "Airline_Delay_Cause.csv")
output_dir = os.path.join(base_dir, "synthetic_output")

raw_flights_file = os.path.join(output_dir, "raw_data.csv")
raw_finance_file = os.path.join(output_dir, "finance.csv")
raw_satisfaction_file = os.path.join(output_dir, "satisfaction.csv")
state_file = os.path.join(output_dir, "state.json")

os.makedirs(output_dir, exist_ok=True)

rng = np.random.default_rng(42)

start_year, start_month = 2025, 2
end_year, end_month = 2026, 8

if os.path.exists(state_file):
    with open(state_file, "r") as f:
        state = json.load(f)
    year, month = state["year"], state["month"]
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
else:
    year, month = start_year, start_month

if year > end_year or (year == end_year and month > end_month):
    print("2026-08 sonrasına geçildi.")
    print("Üretilecek yeni ay kalmadı.")
    exit()

print("=" * 70)
print(f"ÜRETİLECEK AY: {year}-{month:02d}")
print("=" * 70)

df = pd.read_csv(source_file)
print("Kaynak satır sayısı:", len(df))

reference_year = df["year"].max()
reference_month = df[df["year"] == reference_year]["month"].max()
reference = df[(df["year"] == reference_year) & (df["month"] == reference_month)].copy().reset_index(drop=True)

print(f"Referans ay: {reference_year}-{reference_month:02d}")
print("Referans kayıt sayısı:", len(reference))

new_df = reference.copy()
new_df["year"] = year
new_df["month"] = month

# --- flights ---
flights = reference["arr_flights"].fillna(0).to_numpy()
flights = flights * rng.uniform(0.95, 1.05, len(reference))
flights = np.maximum(np.round(flights).astype(int), 0)
new_df["arr_flights"] = flights

# --- delayed flights ---
delay_rate = (reference["arr_del15"].fillna(0) / reference["arr_flights"].replace(0, np.nan)).fillna(0)
arr_del15 = flights * delay_rate.to_numpy() * rng.uniform(0.95, 1.05, len(reference))
arr_del15 = np.minimum(np.round(arr_del15).astype(int), flights)
new_df["arr_del15"] = arr_del15

# --- delay causes (counts) ---
cause_columns = ["carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct"]
cause_total = reference[cause_columns].fillna(0).sum(axis=1)

for col in cause_columns:
    ratio = (reference[col].fillna(0) / cause_total.replace(0, np.nan)).fillna(0)
    new_df[col] = np.round(arr_del15 * ratio.to_numpy(), 2)

# --- cancelled ---
cancelled = reference["arr_cancelled"].fillna(0).to_numpy()
cancel_rate = np.nan_to_num(cancelled / np.where(flights == 0, np.nan, flights))
new_cancelled = flights * cancel_rate * rng.uniform(0.95, 1.05, len(reference))
new_cancelled = np.minimum(np.round(new_cancelled).astype(int), flights)
new_df["arr_cancelled"] = new_cancelled

# --- diverted ---
diverted = reference["arr_diverted"].fillna(0).to_numpy()
divert_rate = np.nan_to_num(diverted / np.where(flights == 0, np.nan, flights))
new_diverted = flights * divert_rate * rng.uniform(0.95, 1.05, len(reference))
new_diverted = np.minimum(np.round(new_diverted).astype(int), flights)
new_df["arr_diverted"] = new_diverted

# --- delay minutes ---
delay_minutes = reference["arr_delay"].fillna(0).to_numpy()
ref_del15 = reference["arr_del15"].fillna(0).to_numpy()
delay_per_delayed_flight = np.nan_to_num(delay_minutes / np.where(ref_del15 == 0, np.nan, ref_del15), nan=0)
new_delay_minutes = arr_del15 * delay_per_delayed_flight * rng.uniform(0.95, 1.05, len(reference))
new_delay_minutes = np.round(new_delay_minutes).astype(int)
new_df["arr_delay"] = new_delay_minutes

# --- delay causes (minutes) ---
delay_columns = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
delay_total = reference[delay_columns].fillna(0).sum(axis=1)

for col in delay_columns:
    ratio = (reference[col].fillna(0) / delay_total.replace(0, np.nan)).fillna(0)
    new_df[col] = np.round(new_delay_minutes * ratio.to_numpy()).astype(int)

# --- finance ---
flights = new_df["arr_flights"].fillna(0)
delay_minutes = new_df["arr_delay"].fillna(0)
cancelled = new_df["arr_cancelled"].fillna(0)
diverted = new_df["arr_diverted"].fillna(0)

revenue_per_flight = rng.uniform(5000, 9000, len(new_df))
revenue = flights * revenue_per_flight

operation_cost_ratio = rng.uniform(0.55, 0.75, len(new_df))
operation_cost = revenue * operation_cost_ratio

delay_cost = (
    delay_minutes * rng.uniform(50, 100, len(new_df))
    + cancelled * rng.uniform(1500, 3000, len(new_df))
    + diverted * rng.uniform(1000, 2000, len(new_df))
)

# --- satisfaction ---
avg_capacity = rng.integers(120, 211, len(new_df))
load_factor = rng.uniform(0.70, 0.92, len(new_df))
passenger_count = (flights * avg_capacity * load_factor).round().astype(int)

delay_per_flight = (delay_minutes / flights.replace(0, np.nan)).fillna(0)
delay_cost_per_passenger = (delay_cost / passenger_count.replace(0, np.nan)).fillna(0)
delay_cost_effect = np.clip(delay_cost_per_passenger / 100, 0, 2)

passenger_satisfaction = 4.6 - delay_cost_effect + rng.normal(0, 0.15, len(new_df))
passenger_satisfaction = np.clip(passenger_satisfaction, 1, 5).round(2)

cancel_rate = (cancelled / flights.replace(0, np.nan)).fillna(0)
divert_rate = (diverted / flights.replace(0, np.nan)).fillna(0)
baggage_penalty = (delay_per_flight / 120) * 0.35 + cancel_rate * 0.70 + divert_rate * 0.40

baggage_satisfaction = 4.45 - baggage_penalty + rng.normal(0, 0.20, len(new_df))
baggage_satisfaction = np.clip(baggage_satisfaction, 1, 5).round(2)

delay_rate = (new_df["arr_del15"] / new_df["arr_flights"].replace(0, np.nan)).fillna(0)

cabin_crew_satisfaction = 4.20 - delay_rate * 0.20 + rng.normal(0, 0.22, len(new_df))
cabin_crew_satisfaction = np.clip(cabin_crew_satisfaction, 1, 5).round(2)

comfort_satisfaction = 4.05 - delay_rate * 0.12 + rng.normal(0, 0.25, len(new_df))
comfort_satisfaction = np.clip(comfort_satisfaction, 1, 5).round(2)

food_beverage_satisfaction = 3.90 + rng.normal(0, 0.30, len(new_df))
food_beverage_satisfaction = np.clip(food_beverage_satisfaction, 1, 5).round(2)

# --- output tables ---
raw_flights = new_df[[
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct",
    "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
    "arr_delay", "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay"
]].copy()

raw_finance = new_df[["year", "month", "carrier", "carrier_name", "airport", "airport_name"]].copy()
raw_finance["Revenue"] = revenue.round(2)
raw_finance["OperationCost"] = operation_cost.round(2)
raw_finance["DelayCost"] = delay_cost.round(2)

raw_satisfaction = new_df[["year", "month", "carrier", "carrier_name", "airport", "airport_name"]].copy()
raw_satisfaction["PassengerCount"] = passenger_count
raw_satisfaction["PassengerSatisfaction"] = passenger_satisfaction
raw_satisfaction["BaggageSatisfaction"] = baggage_satisfaction
raw_satisfaction["CabinCrewSatisfaction"] = cabin_crew_satisfaction
raw_satisfaction["ComfortSatisfaction"] = comfort_satisfaction
raw_satisfaction["FoodBeverageSatisfaction"] = food_beverage_satisfaction


def save_or_append(data, path):
    if os.path.exists(path):
        data.to_csv(path, mode="a", header=False, index=False)
    else:
        data.to_csv(path, mode="w", header=True, index=False)


save_or_append(raw_flights, raw_flights_file)
save_or_append(raw_finance, raw_finance_file)
save_or_append(raw_satisfaction, raw_satisfaction_file)

with open(state_file, "w") as f:
    json.dump({"year": year, "month": month}, f, indent=4)

print()
print("=" * 70)
print("SENTETİK AY BAŞARIYLA ÜRETİLDİ")
print("=" * 70)
print(f"Dönem       : {year}-{month:02d}")
print(f"Flights     : {len(raw_flights)}")
print(f"Finance     : {len(raw_finance)}")
print(f"Satisfaction: {len(raw_satisfaction)}")
print()
print("Oluşturulan dosyalar:")
print(raw_flights_file)
print(raw_finance_file)
print(raw_satisfaction_file)
print()
print("Bir sonraki çalıştırmada bir sonraki ay üretilecek.")