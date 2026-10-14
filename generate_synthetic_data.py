import pandas as pd
import numpy as np

df = pd.read_csv("Airline_Delay_Cause.csv")
print("Satır sayısı:", len(df))
print("Kolon sayısı:", len(df.columns))
print(df.head())

rng = np.random.default_rng(42)
n = len(df)
keys = ["year", "month", "carrier", "carrier_name", "airport", "airport_name"]

flights = df["arr_flights"].fillna(0)
delay_minutes = df["arr_delay"].fillna(0)
cancelled = df["arr_cancelled"].fillna(0)
diverted = df["arr_diverted"].fillna(0)
safe_flights = flights.replace(0, np.nan)

revenue = flights * rng.uniform(5000, 9000, n)
print("\nFinansal veriler:")
print("Revenue:", revenue.head().round(2).tolist())

operation_cost = revenue * rng.uniform(0.55, 0.75, n)
print("Operation Cost:", operation_cost.head().round(2).tolist())

delay_cost = delay_minutes * rng.uniform(50, 100, n) + cancelled * rng.uniform(1500, 3000, n) + diverted * rng.uniform(1000, 2000, n)
print("Delay Cost:", delay_cost.head().round(2).tolist())

passenger_count = (flights * rng.integers(120, 211, n) * rng.uniform(0.70, 0.92, n)).round().astype(int)
print("\nYolcu sayıları:")
print(passenger_count.head().tolist())

delay_per_flight = (delay_minutes / safe_flights).fillna(0)
delay_cost_per_passenger = (delay_cost / passenger_count.replace(0, np.nan)).fillna(0)
passenger_satisfaction = np.clip(4.6 - np.clip(delay_cost_per_passenger / 100, 0, 2) + rng.normal(0, 0.15, n), 1, 5).round(2)
print("\nYolcu memnuniyet puanları:")
print(passenger_satisfaction[:5].tolist())

cancel_rate = (cancelled / safe_flights).fillna(0)
divert_rate = (diverted / safe_flights).fillna(0)
baggage_penalty = (delay_per_flight / 120) * 0.35 + cancel_rate * 0.70 + divert_rate * 0.40
baggage_satisfaction = np.clip(4.45 - baggage_penalty + rng.normal(0, 0.20, n), 1, 5).round(2)
print("\nBagaj memnuniyet puanları:")
print(baggage_satisfaction.head().tolist())

delay_rate = (df["arr_del15"] / safe_flights).fillna(0)
cabin_crew_satisfaction = np.clip(4.20 - delay_rate * 0.20 + rng.normal(0, 0.22, n), 1, 5).round(2)
print("\nKabin ekibi memnuniyet puanları:")
print(cabin_crew_satisfaction.head().tolist())

comfort_satisfaction = np.clip(4.05 - delay_rate * 0.12 + rng.normal(0, 0.25, n), 1, 5).round(2)
print("\nKonfor memnuniyet puanları:")
print(comfort_satisfaction.head().tolist())

food_beverage_satisfaction = np.clip(3.90 + rng.normal(0, 0.30, n), 1, 5).round(2)
print("\nYiyecek-içecek memnuniyet puanları:")
print(food_beverage_satisfaction[:5].tolist())

print("\n--- SATIR SAYISI KONTROLÜ ---")
for name, s in [("Revenue", revenue), ("Operation Cost", operation_cost), ("Delay Cost", delay_cost), ("Passenger Count", passenger_count), ("Passenger Satisfaction", passenger_satisfaction), ("Baggage Satisfaction", baggage_satisfaction), ("Cabin Crew Satisfaction", cabin_crew_satisfaction), ("Comfort Satisfaction", comfort_satisfaction), ("Food & Beverage Satisfaction", food_beverage_satisfaction)]:
    print(f"{name}:", len(s))

print("\n--- SENTETİK VERİ KONTROLÜ ---")
for name, s in [("Revenue", revenue), ("Operation Cost", operation_cost), ("Delay Cost", delay_cost)]:
    print(f"\n{name}:")
    print("Satır sayısı:", len(s))
    print("Minimum:", round(s.min(), 2))
    print("Ortalama:", round(s.mean(), 2))
    print("Maximum:", round(s.max(), 2))

print("\n--- UÇUŞ SAYISI / FİNANS İLİŞKİSİ ---")
print(df[["arr_flights"]].assign(Revenue=revenue, OperationCost=operation_cost, DelayCost=delay_cost).sort_values("arr_flights", ascending=False).head(10).round(2))

print("\n--- GECİKME MALİYETİ / MEMNUNİYET İLİŞKİSİ ---")
check = pd.DataFrame({"DelayCost": delay_cost, "PassengerSatisfaction": passenger_satisfaction, "BaggageSatisfaction": baggage_satisfaction})
print(check.sort_values("DelayCost", ascending=False).head(10).round(2))

print("\n--- YOLCU BAŞINA GECİKME MALİYETİ / MEMNUNİYET ---")
check["DelayCostPerPassenger"] = (check["DelayCost"] / passenger_count.replace(0, np.nan)).round(2)
print(check.sort_values("DelayCostPerPassenger", ascending=False).head(10).round(2))

print("\nKorelasyon:", round(check[["DelayCostPerPassenger", "PassengerSatisfaction"]].corr().iloc[0, 1], 2))

fact_finance = df[keys].assign(Revenue=revenue.round(2), OperationCost=operation_cost.round(2), DelayCost=delay_cost.round(2))
fact_finance.to_csv("FactFinance.csv", index=False)
print("\nFactFinance.csv başarıyla oluşturuldu.")
print("Satır sayısı:", len(fact_finance))
print(fact_finance.head())

fact_satisfaction = df[keys].assign(PassengerCount=passenger_count, PassengerSatisfaction=passenger_satisfaction, BaggageSatisfaction=baggage_satisfaction, CabinCrewSatisfaction=cabin_crew_satisfaction, ComfortSatisfaction=comfort_satisfaction, FoodBeverageSatisfaction=food_beverage_satisfaction)
fact_satisfaction.to_csv("FactSatisfaction.csv", index=False)
print("\nFactSatisfaction.csv başarıyla oluşturuldu.")
print("Satır sayısı:", len(fact_satisfaction))
print(fact_satisfaction.head())