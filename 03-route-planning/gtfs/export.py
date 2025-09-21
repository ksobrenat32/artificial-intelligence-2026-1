import pandas as pd
import io

# --- Data Loading ---
agency = pd.read_csv("agency.txt")
routes = pd.read_csv("routes.txt")
trips = pd.read_csv("trips.txt")
stops = pd.read_csv("stops.txt")
stop_times = pd.read_csv("stop_times.txt")

# --- Data Processing ---

# 1. Merge dataframes to link all the information
print("Step 1: Merging route, agency, and trip information...")
route_agency_info = pd.merge(routes, agency, on='agency_id', how='left')
trip_info = pd.merge(trips, route_agency_info, on='route_id', how='left')
stop_times_with_names = pd.merge(stop_times, stops, on='stop_id', how='left')

# 2. Sort by trip and sequence to prepare for pairing
print("Step 2: Sorting stops by trip and sequence...")
stop_times_with_names = stop_times_with_names.sort_values(['trip_id', 'stop_sequence'])

# 3. Create a shifted dataframe to get the "next stop" for each stop
print("Step 3: Creating origin-destination stop pairs...")
next_stop = stop_times_with_names.groupby('trip_id').shift(-1)

# 4. Combine original and shifted dataframes
journey_pairs = pd.concat([stop_times_with_names, next_stop.add_suffix('_next')], axis=1)

# 5. Remove the last stop of each trip, which doesn't have a "next stop"
journey_pairs = journey_pairs.dropna(subset=['stop_id_next'])

# 6. Convert time columns to timedelta for calculation
print("Step 4: Calculating journey times...")
journey_pairs['departure_time'] = pd.to_timedelta(journey_pairs['departure_time'])
journey_pairs['arrival_time_next'] = pd.to_timedelta(journey_pairs['arrival_time_next'])

# 7. Calculate journey time
journey_pairs['time_of_journey'] = journey_pairs['arrival_time_next'] - journey_pairs['departure_time']

# 8. Merge with trip information
print("Step 5: Assembling the final dataset...")
final_df = pd.merge(journey_pairs, trip_info, on='trip_id', how='left')

# 9. Select and rename the final columns for the output
final_df = final_df[[
    'agency_name',
    'route_long_name',
    'route_short_name',
    'stop_name',
    'stop_name_next',
    'time_of_journey'
]]
final_df = final_df.rename(columns={
    'stop_name': 'origin_stop',
    'stop_name_next': 'destiny_stop'
})

# 10. Sort the final dataframe by agency and route names
final_df = final_df.sort_values(['agency_name', 'route_long_name', 'route_short_name', 'origin_stop', 'destiny_stop'])

# 11. Delete duplicate rows
final_df = final_df.drop_duplicates()

# 12. Change the time_of_journey to string format HH:MM:SS
final_df['time_of_journey'] = final_df['time_of_journey'].dt.components.apply(
    lambda x: f"{int(x['hours']):02}:{int(x['minutes']):02}:{int(x['seconds']):02}", axis=1
)

# --- Output ---

# Save the final result to a CSV file
output_filename = 'transit_graph.csv'
final_df.to_csv(output_filename, index=False)

print(f"\n✅ Processing complete. The file '{output_filename}' has been created.")
print("Preview of the generated data:")
print(final_df.head())