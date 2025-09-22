import pandas as pd

agency = pd.read_csv("agency.txt")
routes = pd.read_csv("routes.txt")
trips = pd.read_csv("trips.txt")
stops = pd.read_csv("stops.txt")
stop_times = pd.read_csv("stop_times.txt")

# merge all
route_agency_info = pd.merge(routes, agency, on='agency_id', how='left')
trip_info = pd.merge(trips, route_agency_info, on='route_id', how='left')
stop_times_with_names = pd.merge(stop_times, stops, on='stop_id', how='left')

# get times
stop_times_with_names = stop_times_with_names.sort_values(['trip_id', 'stop_sequence'])
next_stop = stop_times_with_names.groupby('trip_id').shift(-1)
journey_pairs = pd.concat([stop_times_with_names, next_stop.add_suffix('_next')], axis=1)
journey_pairs = journey_pairs.dropna(subset=['stop_id_next'])
journey_pairs['departure_time'] = pd.to_timedelta(journey_pairs['departure_time'])
journey_pairs['arrival_time_next'] = pd.to_timedelta(journey_pairs['arrival_time_next'])
journey_pairs['time_of_journey'] = journey_pairs['arrival_time_next'] - journey_pairs['departure_time']
final_df = pd.merge(journey_pairs, trip_info, on='trip_id', how='left')

# rename
final_df = final_df[[
    'agency_name',
    'route_long_name',
    'route_short_name',
    'stop_name',
    'stop_lat',
    'stop_lon',
    'stop_name_next',
    'stop_lat_next',
    'stop_lon_next',
    'time_of_journey'
]]
final_df = final_df.rename(columns={
    'stop_name': 'origin_stop',
    'stop_lat': 'origin_lat',
    'stop_lon': 'origin_lon',
    'stop_name_next': 'destiny_stop',
    'stop_lat_next': 'destiny_lat',
    'stop_lon_next': 'destiny_lon'
})

# clean data
final_df = final_df.sort_values(['agency_name', 'route_long_name', 'route_short_name', 'origin_stop', 'destiny_stop'])
final_df = final_df.drop_duplicates()
# format time
final_df['time_of_journey'] = final_df['time_of_journey'].dt.components.apply(
    lambda x: f"{int(x['hours']):02}:{int(x['minutes']):02}:{int(x['seconds']):02}", axis=1
)

output_filename = 'transit_graph.csv'
final_df.to_csv(output_filename, index=False)
print(final_df.head())