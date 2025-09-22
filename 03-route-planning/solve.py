import csv
import heapq
from collections import defaultdict
from typing import List, Tuple, Optional

class TransitGraph:
    """Graph for transit network."""
    
    def __init__(self):
        # Graph as adjacency list: {stop_id: [(neighbor_stop_id, time, route_info), ...]}
        self.graph = defaultdict(list)
        self.stops = set()
        # Map stop_id to stop info: {stop_id: {'name': str, 'lat': float, 'lon': float, 'agencies': set}}
        self.stop_info = {}
        # Map stop names to list of stop_ids for duplicate name handling
        self.stops_by_name = defaultdict(list)
        # Map stop names to agencies: {stop_name: {agency1, agency2, ...}}
        self.stop_agencies = defaultdict(set)
        # Coordinate tolerance for transfer stations (in degrees)
        self.coordinate_tolerance = 0.0045  # ~500 meters
    
    def find_nearby_stop(self, stop_name: str, lat: float, lon: float) -> Optional[str]:
        """Find if there's already a stop with the same name within tolerance distance."""
        if stop_name in self.stops_by_name:
            for existing_stop_id in self.stops_by_name[stop_name]:
                existing_info = self.stop_info[existing_stop_id]
                lat_diff = abs(existing_info['lat'] - lat)
                lon_diff = abs(existing_info['lon'] - lon)
                
                # If within tolerance, consider it the same stop (transfer station)
                if lat_diff < self.coordinate_tolerance and lon_diff < self.coordinate_tolerance:
                    return existing_stop_id
        
        return None
    
    def create_stop_id(self, stop_name: str, lat: float, lon: float) -> str:
        """Create a unique stop identifier using name and coordinates."""
        # Round coordinates to avoid floating point precision issues
        lat_rounded = round(lat, 6)
        lon_rounded = round(lon, 6)
        return f"{stop_name}@{lat_rounded},{lon_rounded}"
    
    def add_stop_info(self, stop_name: str, lat: float, lon: float, agency: Optional[str] = None) -> str:
        """Add stop information and return the unique stop ID."""
        # First check if there's already a nearby stop with the same name
        existing_stop_id = self.find_nearby_stop(stop_name, lat, lon)
        if existing_stop_id:
            # Update agencies for existing stop
            if agency:
                self.stop_info[existing_stop_id]['agencies'].add(agency)
                self.stop_agencies[stop_name].add(agency)
            return existing_stop_id
        
        # Create new stop ID if no nearby stop found
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
        """Add a directed edge from origin to destination with travel time and route information."""
        self.graph[origin_id].append((destination_id, time, route_info))
    
    def parse_time_to_minutes(self, time_str: str) -> float:
        """Convert time string in HH:MM:SS format to minutes."""
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
                # Try to parse as a simple number (minutes)
                return float(time_str)
        except (ValueError, IndexError):
            return 0.0

    def load_from_csv(self, csv_file: str):
        """Load transit data from CSV file."""
        print(f"Loading transit data from {csv_file}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Verify expected columns
                expected_columns = ['agency_name', 'route_long_name', 'route_short_name', 
                                  'origin_stop', 'origin_lat', 'origin_lon', 'destiny_stop', 
                                  'destiny_lat', 'destiny_lon', 'time_of_journey']
                
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
                        
                        # Skip invalid entries
                        if not origin_name or not destination_name or time <= 0:
                            continue
                        
                        agency_name = row['agency_name'].strip()
                        
                        # Create unique stop IDs
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
                        # Skip malformed rows
                        continue
                
                print(f"Successfully loaded {routes_loaded} routes connecting {len(self.stops)} stops")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{csv_file}' not found")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")
    
    def find_stop_by_name(self, stop_name: str) -> List[str]:
        """Find all stop IDs that match the given stop name."""
        return self.stops_by_name.get(stop_name, [])
    
    def find_stop_by_name_and_agency(self, stop_name: str, agency: str) -> List[str]:
        """Find all stop IDs that match the given stop name and are served by the given agency."""
        all_stops = self.find_stop_by_name(stop_name)
        return [stop_id for stop_id in all_stops 
                if agency in self.stop_info[stop_id].get('agencies', set())]
    
    def get_agencies_for_stop_name(self, stop_name: str) -> List[str]:
        """Get all agencies that serve stops with the given name."""
        return sorted(list(self.stop_agencies.get(stop_name, set())))
    
    def get_stop_display_name(self, stop_id: str) -> str:
        """Get a human-readable display name for a stop."""
        if stop_id in self.stop_info:
            info = self.stop_info[stop_id]
            return f"{info['name']} ({info['lat']:.4f}, {info['lon']:.4f})"
        return stop_id
    
    def find_fastest_route(self, start_name: str, end_name: str, start_agency: Optional[str] = None, end_agency: Optional[str] = None) -> Tuple[Optional[float], Optional[List[dict]]]:
        """
        Find the fastest route between two stops using Dijkstra's algorithm.
        Returns (total_time, route_steps) or (None, None) if no route exists.
        """
        # Find stop IDs for the given names, filtered by agency if specified
        if start_agency:
            start_stops = self.find_stop_by_name_and_agency(start_name, start_agency)
        else:
            start_stops = self.find_stop_by_name(start_name)
            
        if end_agency:
            end_stops = self.find_stop_by_name_and_agency(end_name, end_agency)
        else:
            end_stops = self.find_stop_by_name(end_name)
        
        if not start_stops:
            raise ValueError(f"Start stop '{start_name}' not found in transit network")
        if not end_stops:
            raise ValueError(f"End stop '{end_name}' not found in transit network")
        
        # If multiple stops with same name, try all combinations and find best route
        best_time = None
        best_route = None
        best_start = None
        best_end = None
        
        for start_id in start_stops:
            for end_id in end_stops:
                if start_id == end_id:
                    return 0.0, []
                
                time, route = self._dijkstra_search(start_id, end_id)
                if time is not None and (best_time is None or time < best_time):
                    best_time = time
                    best_route = route
                    best_start = start_id
                    best_end = end_id
        
        return best_time, best_route
    
    def _dijkstra_search(self, start_id: str, end_id: str) -> Tuple[Optional[float], Optional[List[dict]]]:
        """
        Find the fastest route between two stops using Dijkstra's algorithm with transfer penalties.
        Penalizes agency changes and route changes to minimize transfers.
        
        Args:
            start: Starting stop name
            end: Destination stop name
            agency_change_penalty: Time penalty (minutes) for changing agencies
            route_change_penalty: Time penalty (minutes) for changing routes within same agency
        
        Returns (total_time, route_steps) or (None, None) if no route exists.
        """
        if start_id == end_id:
            return 0.0, []
        
        # Enhanced Dijkstra's algorithm with transfer penalties
        # State: (stop, last_agency, last_route_short, last_route_long)
        # distances[(stop, agency, route_short, route_long)] = cost
        distances = {}
        previous = {}
        route_info = {}
        
        # Transfer penalties (in minutes)
        agency_change_penalty = 8.0  # 8 minute penalty for changing agencies
        route_change_penalty = 8.0  # 8 minute penalty for changing routes within same agency

        # Initialize starting state
        start_state = (start_id, None, None, None)
        distances[start_state] = 0
        
        # Priority queue: (distance, (stop, last_agency, last_route_short, last_route_long))
        pq = [(0, start_state)]
        visited = set()
        
        while pq:
            current_dist, current_state = heapq.heappop(pq)
            current_stop, last_agency, last_route_short, last_route_long = current_state
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            
            # Check all neighbors from current stop
            for neighbor, travel_time, route_data in self.graph[current_stop]:
                new_agency = route_data['agency']
                new_route_short = route_data['route_short']
                new_route_long = route_data['route_long']
                new_state = (neighbor, new_agency, new_route_short, new_route_long)
                
                if new_state in visited:
                    continue
                
                # Calculate transfer penalty
                transfer_penalty = 0.0
                if last_agency is not None:  # Not the first segment
                    if last_agency != new_agency:
                        # Different agency - higher penalty
                        transfer_penalty = agency_change_penalty
                    elif (last_route_short != new_route_short or 
                          last_route_long != new_route_long):
                        # Same agency, different route - lower penalty
                        transfer_penalty = route_change_penalty
                
                new_distance = current_dist + travel_time + transfer_penalty
                
                # Update if we found a better path to this state
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
        
        # Find the best final state at the destination
        best_end_state = None
        best_distance = float('inf')
        
        for state, distance in distances.items():
            if state[0] == end_id and distance < best_distance:
                best_distance = distance
                best_end_state = state
        
        if best_end_state is None:
            return None, None
        
        # Reconstruct path
        path = []
        current_state = best_end_state
        route_steps = []
        
        while current_state in previous:
            path.append(current_state[0])  # Stop name
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
        
        # Return the total time including penalties (best_distance)
        # This ensures that the displayed time reflects the actual cost used in optimization
        return best_distance, route_steps

def display_route(total_time: float, route_steps: List[dict]):
    """Display the route information in a user-friendly format with transfer information."""
    print(f"\n=== FASTEST ROUTE FOUND ===")
    
    # Calculate base travel time and total penalties
    base_travel_time = sum(step['time'] for step in route_steps)
    total_penalties = sum(step.get('transfer_penalty', 0) for step in route_steps)
    
    print(f"Total time (including penalties): {total_time:.1f} minutes")
    print(f"Base travel time: {base_travel_time:.1f} minutes")
    if total_penalties > 0:
        print(f"Transfer penalties: {total_penalties:.1f} minutes")
    print(f"Number of segments: {len(route_steps)}")
    
    # Count transfers
    transfers = sum(1 for step in route_steps if step.get('is_transfer', False))
    print(f"Number of transfers: {transfers}")
    
    print("\nRoute details:")
    print("-" * 80)
    
    current_agency = None
    current_route = None
    
    for i, step in enumerate(route_steps, 1):
        route_info = step['route_info']
        is_transfer = step.get('is_transfer', False)
        
        # Show transfer indicator
        transfer_indicator = "🔄 TRANSFER" if is_transfer else ""
        
        print(f"{i:2d}. {step['from_display']} → {step['to_display']} {transfer_indicator}")
        print(f"    Route: {route_info['route_short']} - {route_info['route_long']}")
        print(f"    Agency: {route_info['agency']}")
        print(f"    Travel time: {step['time']:.1f} minutes")
        
        # Show transfer penalty if applicable
        if step.get('transfer_penalty', 0) > 0:
            penalty_type = "agency change" if step['transfer_penalty'] >= 5.0 else "route change"
            print(f"    Transfer penalty: {step['transfer_penalty']:.1f} minutes ({penalty_type})")
        
        print()
        
        current_agency = route_info['agency']
        current_route = route_info['route_short']

def get_user_input(transit_graph: TransitGraph) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Get origin and destination from user with validation. Returns (origin, destination, origin_agency, dest_agency)."""
    stop_names = set(transit_graph.stops_by_name.keys())
    print(f"\nAvailable stops: {len(stop_names)} total")
    
    # Show some example stops
    example_stops = sorted(list(stop_names))[:10]
    print("Example stops:", ", ".join(example_stops))
    if len(stop_names) > 10:
        print("... and more")
    
    while True:
        print("\n" + "="*50)
        origin = input("Enter origin stop: ").strip()
        if not origin:
            print("Please enter a valid origin stop.")
            continue
        
        if origin not in stop_names:
            print(f"Stop '{origin}' not found in the transit network.")
            similar = [stop for stop in stop_names if origin.lower() in stop.lower()][:5]
            if similar:
                print(f"Did you mean one of these? {', '.join(similar)}")
            continue
        
        # Check if multiple agencies serve this stop
        origin_agencies = transit_graph.get_agencies_for_stop_name(origin)
        selected_origin_agency = None
        
        if len(origin_agencies) > 1:
            print(f"\nMultiple agencies serve '{origin}':")
            for i, agency in enumerate(origin_agencies, 1):
                print(f"  {i}. {agency}")
            
            while True:
                try:
                    choice = input(f"Select agency for origin (1-{len(origin_agencies)}): ").strip()
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
        
        # Show selected origin locations
        if selected_origin_agency:
            origin_locations = transit_graph.find_stop_by_name_and_agency(origin, selected_origin_agency)
            if len(origin_locations) > 1:
                print(f"Found {len(origin_locations)} locations for '{origin}' ({selected_origin_agency}):")
                for i, stop_id in enumerate(origin_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f})")
        else:
            # Show all locations if no agency filter
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
        
        # Check if multiple agencies serve this stop
        dest_agencies = transit_graph.get_agencies_for_stop_name(destination)
        selected_dest_agency = None
        
        if len(dest_agencies) > 1:
            print(f"\nMultiple agencies serve '{destination}':")
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
        
        # Show selected destination locations
        if selected_dest_agency:
            dest_locations = transit_graph.find_stop_by_name_and_agency(destination, selected_dest_agency)
            if len(dest_locations) > 1:
                print(f"Found {len(dest_locations)} locations for '{destination}' ({selected_dest_agency}):")
                for i, stop_id in enumerate(dest_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f})")
        else:
            # Show all locations if no agency filter
            dest_locations = transit_graph.find_stop_by_name(destination)
            if len(dest_locations) > 1:
                print(f"Found {len(dest_locations)} locations for '{destination}':")
                for i, stop_id in enumerate(dest_locations, 1):
                    info = transit_graph.stop_info[stop_id]
                    agencies = ", ".join(sorted(info.get('agencies', set())))
                    print(f"  {i}. {info['name']} at ({info['lat']:.4f}, {info['lon']:.4f}) - {agencies}")
        
        # Return the stop names and selected agencies (if any)
        return origin, destination, selected_origin_agency, selected_dest_agency

def main():
    """Main function to run the transit route planner."""
    print("=== Transit Route Planner ===")
    
    # Initialize graph and load data
    transit_graph = TransitGraph()
    
    try:
        # Load transit data
        csv_file = "gtfs/transit_graph.csv"
        transit_graph.load_from_csv(csv_file)
        
        if len(transit_graph.stops) == 0:
            print("No valid transit data found. Please check the CSV file.")
            return
        
        while True:
            try:
                # Get user input
                origin, destination, origin_agency, dest_agency = get_user_input(transit_graph)
                
                print(f"\nSearching for fastest route from '{origin}' to '{destination}'...")
                if origin_agency:
                    print(f"Origin agency: {origin_agency}")
                if dest_agency:
                    print(f"Destination agency: {dest_agency}")
                
                # Find fastest route
                total_time, route_steps = transit_graph.find_fastest_route(origin, destination, origin_agency, dest_agency)
                
                if total_time is None:
                    print(f"\nNo route found from '{origin}' to '{destination}'")
                    print("These stops might be in different transit networks or not connected.")
                else:
                    if route_steps is not None:
                        display_route(total_time, route_steps)
                    else:
                        print(f"Route found but no route details available.")
                
                # Ask if user wants to search again
                again = input("\nWould you like to search for another route? (y/n): ").strip().lower()
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
