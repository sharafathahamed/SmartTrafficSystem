import numpy as np
from PIL import Image, ImageDraw
import io
import random

def process_yolo_results(results, num_lanes=4):
    """
    Process detection results for each lane.
    
    Args:
        results: Detection results from detection model.
        num_lanes (int): Number of lanes to divide the image into.
        
    Returns:
        tuple: Lane vehicle counts, types, and ambulance presence.
    """
    if results is None:
        return None, None, None
    
    # Get the width of the image
    img_width = results[0].orig_shape[1]
    lane_width = img_width / num_lanes
    
    # Initialize lane data structures
    lane_vehicle_counts = {i: 0 for i in range(num_lanes)}
    lane_vehicle_types = {i: {} for i in range(num_lanes)}
    lane_has_ambulance = {i: False for i in range(num_lanes)}
    
    # Process each detection
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Get box center x-coordinate
            x_center = (box.xyxy[0][0][0] + box.xyxy[0][0][2]) / 2
            
            # Determine which lane this detection belongs to
            lane_id = min(int(x_center // lane_width), num_lanes - 1)
            
            # Get class name and confidence
            cls_id = int(box.cls)
            class_name = r.names[cls_id]
            confidence = float(box.conf)
            
            # Update lane data
            lane_vehicle_counts[lane_id] += 1
            
            if class_name in lane_vehicle_types[lane_id]:
                lane_vehicle_types[lane_id][class_name] += 1
            else:
                lane_vehicle_types[lane_id][class_name] = 1
                
            # Check for ambulance
            if class_name.lower() == 'ambulance':
                lane_has_ambulance[lane_id] = True
    
    return lane_vehicle_counts, lane_vehicle_types, lane_has_ambulance

def draw_detection_results(image, results, num_lanes=4):
    """
    Draw detection results with bounding boxes and labels using PIL.
    
    Args:
        image: The input image (numpy array).
        results: Detection results from detection model.
        num_lanes (int): Number of lanes to divide the image into.
        
    Returns:
        numpy.ndarray: Image with drawn detections.
    """
    if results is None:
        return image
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image.astype('uint8'))
    draw = ImageDraw.Draw(pil_image)
    
    # Get the width of the image
    img_width, img_height = pil_image.size
    lane_width = img_width / num_lanes
    
    # Draw lane divisions
    for i in range(1, num_lanes):
        x = int(i * lane_width)
        draw.line([(x, 0), (x, img_height)], fill=(255, 255, 255), width=2)
        
        # Add lane numbers
        lane_text = f"Lane {i}"
        text_x = int(lane_width * (i - 0.5))
        draw.text((text_x, 30), lane_text, fill=(255, 255, 255))
    
    # Add last lane number
    lane_text = f"Lane {num_lanes}"
    text_x = int(lane_width * (num_lanes - 0.5))
    draw.text((text_x, 30), lane_text, fill=(255, 255, 255))
    
    # Draw bounding boxes for each detection
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0][0])
            
            # Get class name and confidence
            cls_id = int(box.cls)
            class_name = r.names[cls_id]
            confidence = float(box.conf)
            
            # Determine color based on class (ambulance gets special color)
            if class_name.lower() == 'ambulance':
                color = (255, 0, 0)  # Red for ambulance
            else:
                color = (0, 255, 0)  # Green for other vehicles
            
            # Draw bounding box - convert list to tuple for PIL
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            
            # Prepare label text with clear vehicle type
            label = f"{class_name} {confidence:.2f}"
            
            # Draw label background and text
            # Use a fixed label size since textsize is deprecated
            label_height = 20
            label_width = len(label) * 8  # Approximate width based on characters
            draw.rectangle((x1, y1 - label_height - 5, x1 + label_width, y1), fill=color)
            draw.text((x1, y1 - label_height - 5), label, fill=(255, 255, 255))
            
            # Add a more visible vehicle type label with larger font and contrasting color
            vehicle_label = class_name.upper()
            
            # Draw a very visible vehicle type label
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            
            # Use a background rectangle for visibility
            label_width = len(vehicle_label) * 12  # Make it wider for better visibility
            label_height = 30
            
            # Draw black background first (slightly larger for outline effect)
            draw.rectangle(
                (mid_x - label_width//2 - 2, mid_y - label_height//2 - 2, 
                 mid_x + label_width//2 + 2, mid_y + label_height//2 + 2), 
                fill=(0, 0, 0)
            )
            
            # Draw colored background based on vehicle type
            if vehicle_label == "AMBULANCE":
                bg_color = (200, 0, 0)  # Red for ambulance
            elif vehicle_label in ["BUS", "TRUCK"]:
                bg_color = (50, 50, 200)  # Blue for large vehicles
            elif vehicle_label == "BIKE":
                bg_color = (200, 100, 0)  # Orange for bikes
            else:
                bg_color = (0, 150, 0)  # Green for other vehicles
                
            draw.rectangle(
                (mid_x - label_width//2, mid_y - label_height//2, 
                 mid_x + label_width//2, mid_y + label_height//2), 
                fill=bg_color
            )
            
            # Calculate text position to center it
            text_x = mid_x - (len(vehicle_label) * 6)
            text_y = mid_y - 10
            
            # Draw white text with a shadow for contrast
            for offset in range(2):  # Draw multiple times for "bold" effect
                draw.text((text_x + 1 + offset, text_y + 1), vehicle_label, fill=(0, 0, 0))  # Shadow
                draw.text((text_x + offset, text_y), vehicle_label, fill=(255, 255, 255))  # Text
    
    # Convert back to numpy array
    return np.array(pil_image)

def draw_traffic_light(width=100, height=250, state="RED"):
    """
    Draw a traffic light with specified state using PIL.
    
    Args:
        width (int): Width of the traffic light.
        height (int): Height of the traffic light.
        state (str): State of the traffic light ('RED', 'YELLOW', or 'GREEN').
        
    Returns:
        numpy.ndarray: Traffic light image.
    """
    # Create a new PIL Image with black background
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw traffic light housing - convert to tuples for PIL
    draw.rectangle(((0, 0), (width, height)), fill=(50, 50, 50), outline=(100, 100, 100), width=3)
    
    # Calculate circle centers and radius
    radius = min(width, height) // 7
    circle_x = width // 2
    
    red_y = height // 4
    yellow_y = height // 2
    green_y = 3 * height // 4
    
    # Draw the circles (lights) - using tuples for coordinates
    # Red light
    if state == "RED":
        draw.ellipse(((circle_x - radius, red_y - radius), (circle_x + radius, red_y + radius)), fill=(255, 0, 0))
    else:
        draw.ellipse(((circle_x - radius, red_y - radius), (circle_x + radius, red_y + radius)), fill=(100, 100, 100), outline=(50, 50, 50), width=2)
    
    # Yellow light
    if state == "YELLOW":
        draw.ellipse(((circle_x - radius, yellow_y - radius), (circle_x + radius, yellow_y + radius)), fill=(255, 255, 0))
    else:
        draw.ellipse(((circle_x - radius, yellow_y - radius), (circle_x + radius, yellow_y + radius)), fill=(100, 100, 100), outline=(50, 50, 50), width=2)
    
    # Green light
    if state == "GREEN":
        draw.ellipse(((circle_x - radius, green_y - radius), (circle_x + radius, green_y + radius)), fill=(0, 255, 0))
    else:
        draw.ellipse(((circle_x - radius, green_y - radius), (circle_x + radius, green_y + radius)), fill=(100, 100, 100), outline=(50, 50, 50), width=2)
    
    # Convert to numpy array
    return np.array(img)
