import csv
import heapq
from collections import defaultdict
from typing import List, Tuple, Optional

class TransitGraph:
    """Graph for transit network"""

    def __init__(self):
        # graph: {stop_id: [(neighbor_stop_id, time, route_info), ...]}
        self.graph = defaultdict(list)
        # all stops IDs
        self.stops = set()
        # stop entity: {stop_id: {'name': str, 'lat': float, 'lon': float, 'agencies': set()}}
        self.stop_info = {}
        # stop names to list of stop_ids for duplicate name handling
        self.stops_by_name = defaultdict(list)
        # stop names to agencies: {stop_name: {agency1, agency2, ...}}
        self.stop_agencies = defaultdict(set)
        # distance tolerance to consider stops as the same (500[m] ~ 0.0045)
        self.coordinate_tolerance = 0.0045

    def find_nearby_stop(self, stop_name: str, lat: float, lon: float) -> Optional[str]:
        """Find if there is already joined stops with the same name within tolerance distance"""
        if stop_name in self.stops_by_name:
            for existing_stop_id in self.stops_by_name[stop_name]:
                existing_info = self.stop_info[existing_stop_id]
                lat_diff = abs(existing_info['lat'] - lat)
                lon_diff = abs(existing_info['lon'] - lon)

                if lat_diff < self.coordinate_tolerance and lon_diff < self.coordinate_tolerance:
                    return existing_stop_id

        return None

    def create_stop_id(self, stop_name: str, lat: float, lon: float) -> str:
        """Create a stop ID"""
        lat_rounded = round(lat, 6)
        lon_rounded = round(lon, 6)
        return f"{stop_name}@{lat_rounded},{lon_rounded}"

    def add_stop_info(self, stop_name: str, lat: float, lon: float, agency: Optional[str] = None) -> str:
        """Add stop info"""
        existing_stop_id = self.find_nearby_stop(stop_name, lat, lon)
        if existing_stop_id:
            if agency:
                self.stop_info[existing_stop_id]['agencies'].add(agency)
                self.stop_agencies[stop_name].add(agency)
            return existing_stop_id

        stop_id = self.create_stop_id(stop_name, lat, lon)

        if stop_id not in self.stop_info:
            self.stop_info[stop_id] = {
                'name': stop_name,
                'lat': lat,
                'lon': lon,
                'agencies': set([agency]) if agency else set()
            }
            self.stops_by_name[stop_name].append(stop_id)
            self.stops.add(stop_id)
            if agency:
                self.stop_agencies[stop_name].add(agency)

        return stop_id

    def add_edge(self, origin_id: str, destination_id: str, time: float, route_info: dict):
        # Add directed edge to graph
        self.graph[origin_id].append((destination_id, time, route_info))

    def parse_time_to_minutes(self, time_str: str) -> float:
        try:
            time_parts = time_str.split(':')
            if len(time_parts) == 3:
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = int(time_parts[2])
                return hours * 60 + minutes + seconds / 60
            elif len(time_parts) == 2:
                minutes = int(time_parts[0])
                seconds = int(time_parts[1])
                return minutes + seconds / 60
            else:
                return float(time_str)
        except (ValueError, IndexError):
            return 0.0

    def load_from_csv(self, csv_file: str):
        print(f"Loading transit data from {csv_file}...")

        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            expected_columns = ['agency_name', 'route_long_name', 'route_short_name', 'origin_stop', 'origin_lat', 'origin_lon', 'destiny_stop', 'destiny_lat', 'destiny_lon', 'time_of_journey']

            fieldnames = reader.fieldnames or []
            if not all(col in fieldnames for col in expected_columns):
                missing = set(expected_columns) - set(fieldnames)
                raise ValueError(f"Missing expected columns: {missing}")

            routes_loaded = 0
            for row in reader:
                try:
                    origin_name = row['origin_stop'].strip()
                    destination_name = row['destiny_stop'].strip()
                    origin_lat = float(row['origin_lat'])
                    origin_lon = float(row['origin_lon'])
                    destiny_lat = float(row['destiny_lat'])
                    destiny_lon = float(row['destiny_lon'])
                    time_str = row['time_of_journey'].strip()
                    time = self.parse_time_to_minutes(time_str)

                    if not origin_name or not destination_name or time <= 0:
                        continue

                    agency_name = row['agency_name'].strip()

                    origin_id = self.add_stop_info(origin_name, origin_lat, origin_lon, agency_name)
                    destination_id = self.add_stop_info(destination_name, destiny_lat, destiny_lon, agency_name)

                    route_info = {
                        'agency': agency_name,
                        'route_long': row['route_long_name'].strip(),
                        'route_short': row['route_short_name'].strip()
                    }

                    self.add_edge(origin_id, destination_id, time, route_info)
                    routes_loaded += 1

                except (ValueError, KeyError) as e:
                    continue

            print(f"loaded file: {routes_loaded} routes connecting {len(self.stops)} stops")

    def find_stop_by_name(self, stop_name: str) -> List[str]:
        return self.stops_by_name.get(stop_name, [])

    def find_stop_by_name_and_agency(self, stop_name: str, agency: str) -> List[str]:
        all_stops = self.find_stop_by_name(stop_name)
        return [stop_id for stop_id in all_stops if agency in self.stop_info[stop_id].get('agencies', set())]

    def get_agencies_for_stop_name(self, stop_name: str) -> List[str]:
        return sorted(list(self.stop_agencies.get(stop_name, set())))

    def get_stop_display_name(self, stop_id: str) -> str:
        # return station name
        if stop_id in self.stop_info:
            info = self.stop_info[stop_id]
            return f"{info['name']}"
        return stop_id

    def find_fastest_route(self, start_name: str, end_name: str, start_agency: Optional[str] = None, end_agency: Optional[str] = None) -> Tuple[Optional[float], Optional[List[dict]]]:
        """Find route between two stops using dijkstra"""
        if start_agency:
            start_stops = self.find_stop_by_name_and_agency(start_name, start_agency)
        else:
            start_stops = self.find_stop_by_name(start_name)

        if end_agency:
            end_stops = self.find_stop_by_name_and_agency(end_name, end_agency)
        else:
            end_stops = self.find_stop_by_name(end_name)

        if not start_stops or not end_stops:
            raise ValueError("Stop not found")

        best_time = None
        best_route = None

        for start_id in start_stops:
            for end_id in end_stops:
                if start_id == end_id:
                    return 0.0, []

                time, route = self._dijkstra_search(start_id, end_id)
                if time is not None and (best_time is None or time < best_time):
                    best_time = time
                    best_route = route

        return best_time, best_route

    def _dijkstra_search(self, start_id: str, end_id: str) -> Tuple[Optional[float], Optional[List[dict]]]:
        """dijkstra algorithm with transfer penalties for transfer agencies and routes"""

        if start_id == end_id:
            return 0.0, []

        distances = {}
        previous = {}
        route_info = {}

        agency_change_penalty = 8.0  # 8 minute penalty for changing agencies
        route_change_penalty = 8.0  # 8 minute penalty for changing routes within same agency

        start_state = (start_id, None, None, None)
        distances[start_state] = 0

        pq = [(0, start_state)]
        visited = set()

        while pq:
            current_dist, current_state = heapq.heappop(pq)
            current_stop, last_agency, last_route_short, last_route_long = current_state

            if current_state in visited:
                continue

            visited.add(current_state)

            # check all neighbors from current stop
            for neighbor, travel_time, route_data in self.graph[current_stop]:
                new_agency = route_data['agency']
                new_route_short = route_data['route_short']
                new_route_long = route_data['route_long']
                new_state = (neighbor, new_agency, new_route_short, new_route_long)

                if new_state in visited:
                    continue

                # get transfer penalty
                transfer_penalty = 0.0
                if last_agency is not None:
                    if last_agency != new_agency:
                        transfer_penalty = agency_change_penalty
                    elif (last_route_short != new_route_short or last_route_long != new_route_long):
                        transfer_penalty = route_change_penalty

                new_distance = current_dist + travel_time + transfer_penalty

                # update if we found a better path
                if new_state not in distances or new_distance < distances[new_state]:
                    distances[new_state] = new_distance
                    previous[new_state] = current_state
                    route_info[new_state] = {
                        'route_data': route_data,
                        'travel_time': travel_time,
                        'transfer_penalty': transfer_penalty,
                        'is_transfer': transfer_penalty > 0
                    }
                    heapq.heappush(pq, (new_distance, new_state))

        # find the best final state
        best_end_state = None
        best_distance = float('inf')

        for state, distance in distances.items():
            if state[0] == end_id and distance < best_distance:
                best_distance = distance
                best_end_state = state

        if best_end_state is None:
            return None, None

        # reconstruct path
        path = []
        current_state = best_end_state
        route_steps = []

        while current_state in previous:
            path.append(current_state[0])
            prev_state = previous[current_state]
            route_data = route_info[current_state]

            route_steps.append({
                'from': prev_state[0],
                'to': current_state[0],
                'from_display': self.get_stop_display_name(prev_state[0]),
                'to_display': self.get_stop_display_name(current_state[0]),
                'time': route_data['travel_time'],
                'transfer_penalty': route_data['transfer_penalty'],
                'is_transfer': route_data['is_transfer'],
                'route_info': route_data['route_data']
            })

            current_state = prev_state

        path.append(start_id)
        path.reverse()
        route_steps.reverse()

        # return the total time including penalties (best_distance)
        return best_distance, route_steps

def display_route(total_time: float, route_steps: List[dict]):
    """Display the route details"""
    print(f"\n--- Solution Found ---")

    transfers = sum(1 for step in route_steps if step.get('is_transfer', False))

    print(f"Total time: {total_time:.1f} min")
    print(f"Number of stops: {len(route_steps)}")
    print(f"Number of transfers: {transfers}")
    print("\nRoute details:")
    print("-" * 80)

    for i, step in enumerate(route_steps, 1):
        route_info = step['route_info']
        is_transfer = step.get('is_transfer', False)

        transfer_indicator = "(---TRANSFER---)" if is_transfer else ""

        print(f"{i:2d}. {step['from_display']} → {step['to_display']} {transfer_indicator}")
        print(f"- Route: {route_info['route_short']} - {route_info['route_long']}")
        print(f"- Transport system: {route_info['agency']}")
        print(f"- Estimated travel time: {step['time']:.1f} min")

        if step.get('transfer_penalty', 0) > 0:
            penalty_type = "agency change" if step['transfer_penalty'] >= 5.0 else "route change"
            print(f"- Transfer penalty: {step['transfer_penalty']:.1f} min ({penalty_type})")

        print()

def get_user_input(transit_graph: TransitGraph) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Get terminal input"""
    stop_names = set(transit_graph.stops_by_name.keys())
    print(f"\nAvailable stops: {len(stop_names)} total")

    while True:
        print("\n" + "="*50)
        origin = input("Where are you now?: ").strip()
        if not origin:
            print("Please enter a valid origin stop :)")
            continue

        if origin not in stop_names:
            print(f"Stop '{origin}' not found in the transit network")
            continue

        origin_agencies = transit_graph.get_agencies_for_stop_name(origin)
        selected_origin_agency = None

        if len(origin_agencies) > 1:
            print(f"\nMultiple transport method serve '{origin}':")
            for i, agency in enumerate(origin_agencies, 1):
                print(f"  {i}. {agency}")

            while True:
                try:
                    choice = input(f"Select transport method for origin (1-{len(origin_agencies)}): ").strip()
                    agency_index = int(choice) - 1
                    if 0 <= agency_index < len(origin_agencies):
                        selected_origin_agency = origin_agencies[agency_index]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(origin_agencies)}")
                except ValueError:
                    print("Please enter a valid number")
        elif len(origin_agencies) == 1:
            selected_origin_agency = origin_agencies[0]

        if selected_origin_agency:
            origin_locations = transit_graph.find_stop_by_name_and_agency(origin, selected_origin_agency)
            if len(origin_locations) > 1:
                print(f"Found {len(origin_locations)} locations for '{origin}' ({selected_origin_agency}):")
                for i, stop_id in enumerate(origin_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f})")
        else:
            origin_locations = transit_graph.find_stop_by_name(origin)
            if len(origin_locations) > 1:
                print(f"Found {len(origin_locations)} locations for '{origin}':")
                for i, stop_id in enumerate(origin_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    agencies = ", ".join(sorted(info.get('agencies', set())))
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f}) - {agencies}")

        destination = input("Enter destination stop: ").strip()
        if not destination:
            print("Please enter a valid destination stop.")
            continue

        if destination not in stop_names:
            print(f"Stop '{destination}' not found in the transit network.")
            similar = [stop for stop in stop_names if destination.lower() in stop.lower()][:5]
            if similar:
                print(f"Did you mean one of these? {', '.join(similar)}")
            continue

        dest_agencies = transit_graph.get_agencies_for_stop_name(destination)
        selected_dest_agency = None

        if len(dest_agencies) > 1:
            print(f"\nMultiple transport method serve '{destination}':")
            for i, agency in enumerate(dest_agencies, 1):
                print(f"  {i}. {agency}")

            while True:
                try:
                    choice = input(f"Select agency for destination (1-{len(dest_agencies)}): ").strip()
                    agency_index = int(choice) - 1
                    if 0 <= agency_index < len(dest_agencies):
                        selected_dest_agency = dest_agencies[agency_index]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(dest_agencies)}")
                except ValueError:
                    print("Please enter a valid number")
        elif len(dest_agencies) == 1:
            selected_dest_agency = dest_agencies[0]

        if selected_dest_agency:
            dest_locations = transit_graph.find_stop_by_name_and_agency(destination, selected_dest_agency)
            if len(dest_locations) > 1:
                print(f"Found {len(dest_locations)} locations for '{destination}' ({selected_dest_agency}):")
                for i, stop_id in enumerate(dest_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f})")
        else:
            dest_locations = transit_graph.find_stop_by_name(destination)
            if len(dest_locations) > 1:
                print(f"Found {len(dest_locations)} locations for '{destination}':")
                for i, stop_id in enumerate(dest_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    agencies = ", ".join(sorted(info.get('agencies', set())))
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f}) - {agencies}")

        return origin, destination, selected_origin_agency, selected_dest_agency

def main():
    print("--- Transport Route Planner ---")

    transit_graph = TransitGraph()

    try:
        csv_file = "gtfs/transit_graph.csv"
        transit_graph.load_from_csv(csv_file)

        if len(transit_graph.stops) == 0:
            print("No valid transit data found. Please check the CSV file.")
            return

        while True:
            try:
                origin, destination, origin_agency, dest_agency = get_user_input(transit_graph)

                print(f"\nSearching for fastest route from '{origin}' to '{destination}'...")
                if origin_agency:
                    print(f"Origin agency: {origin_agency}")
                if dest_agency:
                    print(f"Destination agency: {dest_agency}")

                total_time, route_steps = transit_graph.find_fastest_route(origin, destination, origin_agency, dest_agency)

                if total_time is None:
                    print(f"\nNo route found from '{origin}' to '{destination}'")
                    print("These stops might be in different transit networks or not connected.")
                else:
                    if route_steps is not None:
                        display_route(total_time, route_steps)
                    else:
                        print(f"Route found but no route details available.")

                again = input("\nWould you like to search for any other route? (y/n): ").strip().lower()
                if again not in ['y', 'yes']:
                    break

            except ValueError as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nRoute search cancelled by user.")
                break

    except Exception as e:
        print(f"Error: {e}")
        return

    print("\nThank you for using the Transit Route Planner!")

if __name__ == "__main__":
    main()
