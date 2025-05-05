class SignalLogic:
    """
    Class to handle traffic light timing and priority logic.
    
    Signal times based on traffic density:
    - 0s (RED): No traffic
    - 15s (YELLOW): Low traffic (0-5 vehicles)
    - 30s (GREEN): Medium traffic (6-10 vehicles)
    - 60s (GREEN): High traffic (>10 vehicles or ambulance detected)
    """
    
    def __init__(self, num_lanes=4):
        """
        Initialize the signal logic.
        
        Args:
            num_lanes (int): Number of traffic lanes to manage.
        """
        self.num_lanes = num_lanes
        self.lane_vehicle_counts = {i: 0 for i in range(num_lanes)}
        self.lane_vehicle_types = {i: {} for i in range(num_lanes)}
        self.lane_has_ambulance = {i: False for i in range(num_lanes)}
        
    def update_lane_data(self, lane_id, vehicle_count, vehicle_types, has_ambulance):
        """
        Update data for a specific lane.
        
        Args:
            lane_id (int): The lane ID to update.
            vehicle_count (int): Number of vehicles in the lane.
            vehicle_types (dict): Dictionary of vehicle types and their counts.
            has_ambulance (bool): Whether an ambulance is detected in this lane.
        """
        if lane_id < 0 or lane_id >= self.num_lanes:
            raise ValueError(f"Lane ID {lane_id} is out of range (0-{self.num_lanes-1})")
            
        self.lane_vehicle_counts[lane_id] = vehicle_count
        self.lane_vehicle_types[lane_id] = vehicle_types
        self.lane_has_ambulance[lane_id] = has_ambulance
        
    def get_signal_time(self, lane_id):
        """
        Get the signal time for a specific lane based on its traffic density.
        
        Args:
            lane_id (int): The lane ID to get the signal time for.
            
        Returns:
            int: Signal time in seconds (0, 15, 30, or 60).
        """
        if lane_id < 0 or lane_id >= self.num_lanes:
            raise ValueError(f"Lane ID {lane_id} is out of range (0-{self.num_lanes-1})")
            
        vehicle_count = self.lane_vehicle_counts[lane_id]
        has_ambulance = self.lane_has_ambulance[lane_id]
        
        if has_ambulance:
            return 60  # Highest priority for ambulance
        elif vehicle_count == 0:
            return 0   # No traffic
        elif vehicle_count <= 5:
            return 15  # Low traffic
        elif vehicle_count <= 10:
            return 30  # Medium traffic
        else:
            return 60  # High traffic
    
    def calculate_signal_priority(self):
        """
        Calculate signal priority for all lanes.
        
        Returns:
            dict: Dictionary with lane IDs as keys and tuples of (signal_time, signal_color) as values.
        """
        # Calculate base signal times for all lanes
        lane_signal_times = {lane_id: self.get_signal_time(lane_id) for lane_id in range(self.num_lanes)}
        
        # Sort lanes by priority (ambulance first, then signal time)
        sorted_lanes = sorted(
            lane_signal_times.items(),
            key=lambda x: (0 if self.lane_has_ambulance[x[0]] else 1, -x[1])
        )
        
        # Assign final signal times and colors according to priority
        # No two lanes should get the same light/time at the same moment
        result = {}
        
        # Pre-defined signal assignments based on priority positions
        signal_assignments = [
            (60, "GREEN"),    # Highest priority lane gets 60s GREEN
            (45, "YELLOW"),   # Second priority lane gets 45s YELLOW
            (30, "GREEN"),    # Third priority lane gets 30s GREEN
            (0, "RED")        # Lowest priority lane gets 0s RED
        ]
        
        # Assign signals based on priority
        for i, (lane_id, _) in enumerate(sorted_lanes):
            if i < len(signal_assignments):
                result[lane_id] = signal_assignments[i]
            else:
                result[lane_id] = (0, "RED")  # Any extra lanes get RED
                
        return result
