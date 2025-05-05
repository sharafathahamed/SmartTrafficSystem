import streamlit as st
from PIL import Image
import numpy as np
import time
import io
import tempfile
import os
from utils import process_yolo_results, draw_detection_results, draw_traffic_light

def setup_ui():
    """Setup the Streamlit UI for the traffic light control app."""
    st.set_page_config(
        page_title="Smart Traffic Light Control System",
        page_icon="🚦",
        layout="wide"
    )
    
    # Title and header
    st.title("🚦 Smart Traffic Light Control System")
    st.markdown("""
    This system uses YOLOv8 to detect vehicles in traffic lanes and dynamically control 
    traffic lights based on vehicle density and emergency vehicle presence.
    """)
    
    # Get configuration from sidebar and store in session state
    if 'config' not in st.session_state:
        st.session_state.config = setup_sidebar()
    
    # Main page content
    setup_upload_section()
    
def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.header("🛠️ Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_version = st.sidebar.selectbox(
        "Select YOLOv8 Model",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Select the YOLOv8 model to use. Larger models are more accurate but slower.",
        key="model_version_select"
    )
    
    # Detection settings
    detection_confidence = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence threshold for vehicle detection.",
        key="detection_confidence_slider"
    )
    
    # Number of lanes
    num_lanes = st.sidebar.number_input(
        "Number of Traffic Lanes",
        min_value=2,
        max_value=4,
        value=4,
        help="Number of traffic lanes to analyze.",
        key="num_lanes_input"
    )
    
    # Example images
    st.sidebar.subheader("Example Images")
    example_images = {
        "Traffic Intersection 1": "https://images.unsplash.com/photo-1490855680410-49b201432be4",
        "Traffic Intersection 2": "https://images.unsplash.com/photo-1540621371720-5182776e9ca5",
        "Traffic Intersection 3": "https://images.unsplash.com/photo-1508149656289-40e019ca2791",
        "Traffic Intersection 4": "https://images.unsplash.com/photo-1531270144996-257a6b16bea6",
        "Traffic Light 1": "https://images.unsplash.com/photo-1579034024130-238befbd5441",
        "Traffic Light 2": "https://images.unsplash.com/photo-1524009895966-d794500c47a2",
        "Traffic Light 3": "https://images.unsplash.com/photo-1476900164809-ff19b8ae5968",
        "Emergency Vehicle 1": "https://images.unsplash.com/photo-1554734867-bf3c00a49371",
        "Emergency Vehicle 2": "https://images.unsplash.com/photo-1602595363908-5c2c56832613"
    }
    
    selected_example = st.sidebar.selectbox(
        "Select Example Image",
        options=list(example_images.keys()),
        index=0,
        key="example_image_select"
    )
    
    if st.sidebar.button("Use Example Image", key="use_example_button"):
        # Download and use the selected example image
        try:
            import requests
            from io import BytesIO
            
            response = requests.get(example_images[selected_example])
            if response.status_code == 200:
                # Store the image in session state
                image_bytes = BytesIO(response.content)
                st.session_state.example_image = Image.open(image_bytes)
                st.session_state.using_example = True
                st.rerun()
            else:
                st.sidebar.error(f"Failed to download image: HTTP {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"Error loading example image: {str(e)}")
    
    # Return all the configuration options
    return {
        "model_version": model_version,
        "detection_confidence": detection_confidence,
        "num_lanes": int(num_lanes)
    }
    
def setup_upload_section():
    """Setup the file upload section."""
    st.header("📤 Upload Traffic Images/Videos")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Upload images or videos of traffic lanes",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Check if we have an example image from the sidebar
    if 'example_image' in st.session_state and st.session_state.using_example:
        # Process the example image
        example_image = st.session_state.example_image
        process_uploaded_image(example_image, None)
        # Reset the flag
        st.session_state.using_example = False
    
    # Process uploaded files
    if uploaded_files:
        # Collect image files
        image_files = [f for f in uploaded_files if f.name.lower().endswith(('jpg', 'jpeg', 'png'))]
        video_files = [f for f in uploaded_files if not f.name.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        # If we have multiple images, process them as lanes
        if len(image_files) > 1:
            process_multiple_images_as_lanes(image_files)
        # Otherwise process each file individually
        else:
            for uploaded_file in uploaded_files:
                # Determine if it's an image or video
                if uploaded_file.name.lower().endswith(('jpg', 'jpeg', 'png')):
                    # Process single image
                    image = Image.open(uploaded_file)
                    process_uploaded_image(image, uploaded_file.name)
                else:
                    # For video files
                    process_uploaded_video(uploaded_file)
                
def process_uploaded_image(image, filename):
    """
    Process an uploaded image with vehicle detection and traffic light control.
    
    Args:
        image: PIL Image object
        filename: Original filename or None if example
    """
    from model_loader import ModelLoader
    from signal_logic import SignalLogic
    
    # Get configuration from session state
    config = st.session_state.config
    
    # Create subheader for this image
    if filename:
        st.subheader(f"Analysis for: {filename}")
    else:
        st.subheader("Example Image Analysis")
    
    # Display the original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    # Initialize model loader
    model_loader = ModelLoader()
    model_loaded = model_loader.load_model(config["model_version"])
    
    if not model_loaded:
        st.error("Failed to load YOLOv8 model. Please check your configuration.")
        return
    
    # Convert PIL image to OpenCV format for processing
    cv2_image = np.array(image)
    
    # Detect objects in the image
    results = model_loader.detect_objects(cv2_image, conf=config["detection_confidence"])
    
    if results:
        # Process YOLO results for each lane
        lane_vehicle_counts, lane_vehicle_types, lane_has_ambulance = process_yolo_results(
            results, num_lanes=config["num_lanes"]
        )
        
        # Initialize signal logic
        signal_logic = SignalLogic(num_lanes=config["num_lanes"])
        
        # Update signal logic with lane data
        for lane_id in range(config["num_lanes"]):
            signal_logic.update_lane_data(
                lane_id,
                lane_vehicle_counts[lane_id],
                lane_vehicle_types[lane_id],
                lane_has_ambulance[lane_id]
            )
        
        # Calculate signal priority
        signal_priority = signal_logic.calculate_signal_priority()
        
        # Draw detection results on the image
        result_image = draw_detection_results(cv2_image, results, num_lanes=config["num_lanes"])
        
        # Display the result image with detections
        with col2:
            st.image(result_image, caption="Detection Results", use_container_width=True)
        
        # Display lane summary and traffic lights
        st.subheader("🚥 Lane Analysis and Traffic Signal Assignment")
        
        # Create a row of columns for each lane
        lane_cols = st.columns(config["num_lanes"])
        
        # Display traffic light and summary for each lane
        for lane_id in range(config["num_lanes"]):
            with lane_cols[lane_id]:
                signal_time, signal_color = signal_priority[lane_id]
                
                # Display lane header with special styling for ambulance
                if lane_has_ambulance[lane_id]:
                    st.markdown(f"### 🚨 Lane {lane_id+1} (EMERGENCY)")
                else:
                    st.markdown(f"### Lane {lane_id+1}")
                
                # Draw and display traffic light
                traffic_light_img = draw_traffic_light(state=signal_color)
                st.image(traffic_light_img, caption=f"{signal_color}: {signal_time}s", width=100)
                
                # Vehicle count summary
                st.write(f"**Total Vehicles:** {lane_vehicle_counts[lane_id]}")
                
                # Vehicle types breakdown
                st.write("**Vehicle Types:**")
                for vehicle_type, count in lane_vehicle_types[lane_id].items():
                    st.write(f"- {vehicle_type}: {count}")
                
                # Ambulance alert
                if lane_has_ambulance[lane_id]:
                    st.error("⚠️ Emergency vehicle detected!")
    else:
        st.error("No detection results available. The model may have failed to process the image.")

def process_multiple_images_as_lanes(image_files):
    """
    Process multiple uploaded images, treating each as a separate lane.
    
    Args:
        image_files: List of uploaded image files
    """
    from model_loader import ModelLoader
    from signal_logic import SignalLogic
    
    # Get configuration from session state
    config = st.session_state.config
    
    # Limit to maximum number of lanes (default is 4)
    max_lanes = min(len(image_files), 4)
    image_files = image_files[:max_lanes]
    
    # Create subheader for the intersection
    st.subheader(f"Multi-Lane Intersection Analysis ({max_lanes} lanes)")
    
    # Create a row for displaying original images
    st.write("### Original Lane Images")
    img_cols = st.columns(max_lanes)
    
    # Open and display all images
    images = []
    for i, img_file in enumerate(image_files):
        with img_cols[i]:
            image = Image.open(img_file)
            st.image(image, caption=f"Lane {i+1}: {img_file.name}", use_container_width=True)
            images.append(image)
    
    # Initialize model loader
    model_loader = ModelLoader()
    model_loaded = model_loader.load_model(config["model_version"])
    
    if not model_loaded:
        st.error("Failed to load YOLOv8 model. Please check your configuration.")
        return
    
    # Process each image/lane
    st.write("### Detection Results")
    detection_cols = st.columns(max_lanes)
    
    # Initialize data structures for traffic analysis
    lane_vehicle_counts = {}
    lane_vehicle_types = {}
    lane_has_ambulance = {}
    
    # Process each image (lane)
    for lane_id, image in enumerate(images):
        # Convert PIL image to numpy array for processing
        cv2_image = np.array(image)
        
        # Detect objects in the image
        results = model_loader.detect_objects(cv2_image, conf=config["detection_confidence"])
        
        if results:
            # For individual lanes, we treat the entire image as a single lane
            # So we modify the process_yolo_results function behavior
            single_lane_counts, single_lane_types, single_lane_ambulance = process_yolo_results(
                results, num_lanes=1
            )
            
            # Store the results for this lane
            lane_vehicle_counts[lane_id] = single_lane_counts[0]  # Use the first (only) lane
            lane_vehicle_types[lane_id] = single_lane_types[0]
            lane_has_ambulance[lane_id] = single_lane_ambulance[0]
            
            # Draw detection results
            result_image = draw_detection_results(cv2_image, results, num_lanes=1)
            
            # Display detection results
            with detection_cols[lane_id]:
                st.image(result_image, caption=f"Lane {lane_id+1} Detections", use_container_width=True)
        else:
            # Handle case where detection failed
            lane_vehicle_counts[lane_id] = 0
            lane_vehicle_types[lane_id] = {}
            lane_has_ambulance[lane_id] = False
            
            with detection_cols[lane_id]:
                st.error(f"No detections for Lane {lane_id+1}")
    
    # Initialize signal logic with the correct number of lanes
    signal_logic = SignalLogic(num_lanes=max_lanes)
    
    # Update signal logic with lane data
    for lane_id in range(max_lanes):
        signal_logic.update_lane_data(
            lane_id,
            lane_vehicle_counts[lane_id],
            lane_vehicle_types[lane_id],
            lane_has_ambulance[lane_id]
        )
    
    # Calculate signal priority
    signal_priority = signal_logic.calculate_signal_priority()
    
    # Display lane summary and traffic lights
    st.subheader("🚥 Lane Analysis and Traffic Signal Assignment")
    
    # Create a row of columns for each lane
    lane_cols = st.columns(max_lanes)
    
    # Display traffic light and summary for each lane
    for lane_id in range(max_lanes):
        with lane_cols[lane_id]:
            signal_time, signal_color = signal_priority[lane_id]
            
            # Display lane header with special styling for ambulance
            if lane_has_ambulance[lane_id]:
                st.markdown(f"### 🚨 Lane {lane_id+1} (EMERGENCY)")
            else:
                st.markdown(f"### Lane {lane_id+1}")
            
            # Draw and display traffic light
            traffic_light_img = draw_traffic_light(state=signal_color)
            st.image(traffic_light_img, caption=f"{signal_color}: {signal_time}s", width=100)
            
            # Vehicle count summary
            st.write(f"**Total Vehicles:** {lane_vehicle_counts[lane_id]}")
            
            # Vehicle types breakdown
            st.write("**Vehicle Types:**")
            for vehicle_type, count in lane_vehicle_types[lane_id].items():
                st.write(f"- {vehicle_type}: {count}")
            
            # Ambulance alert
            if lane_has_ambulance[lane_id]:
                st.error("⚠️ Emergency vehicle detected!")

def process_uploaded_video(video_file):
    """
    Process an uploaded video with vehicle detection and traffic light control.
    
    Args:
        video_file: Uploaded video file
    """
    from model_loader import ModelLoader
    from signal_logic import SignalLogic
    import tempfile
    
    # Get configuration from session state
    config = st.session_state.config
    
    # Create subheader for this video
    st.subheader(f"Analysis for: {video_file.name}")
    
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_filename = tmp_file.name
    
    # Display the original video
    st.video(temp_filename)
    
    # Initialize model loader
    model_loader = ModelLoader()
    model_loaded = model_loader.load_model(config["model_version"])
    
    if not model_loaded:
        st.error("Failed to load YOLOv8 model. Please check your configuration.")
        # Clean up the temporary file
        os.unlink(temp_filename)
        return
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create containers for results
    frame_container = st.container()
    traffic_light_container = st.container()
    
    # Process the video
    try:
        # We'll use OpenCV to process the video frames
        import cv2
        
        # Open the video file
        cap = cv2.VideoCapture(temp_filename)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Skip frames if the video is too long
        frame_skip = max(1, int(fps / 2))  # Process 2 frames per second
        
        # Initialize signal logic
        signal_logic = SignalLogic(num_lanes=config["num_lanes"])
        
        # Process frames at intervals
        frame_count = 0
        processed_count = 0
        
        # Store the analyzed frames
        analyzed_frames = []
        
        with frame_container:
            st.subheader("🎬 Video Frame Analysis")
            frame_col1, frame_col2 = st.columns(2)
        
        with traffic_light_container:
            st.subheader("🚥 Lane Analysis and Traffic Signal Assignment")
            
            # Create a row of columns for each lane
            lane_cols = st.columns(config["num_lanes"])
        
        # Process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
                
            processed_count += 1
            
            # Update progress bar
            progress = int(frame_count / total_frames * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")
            
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects in the frame
            results = model_loader.detect_objects(rgb_frame, conf=config["detection_confidence"])
            
            if results:
                # Process YOLO results for each lane
                lane_vehicle_counts, lane_vehicle_types, lane_has_ambulance = process_yolo_results(
                    results, num_lanes=config["num_lanes"]
                )
                
                # Update signal logic with lane data
                for lane_id in range(config["num_lanes"]):
                    signal_logic.update_lane_data(
                        lane_id,
                        lane_vehicle_counts[lane_id],
                        lane_vehicle_types[lane_id],
                        lane_has_ambulance[lane_id]
                    )
                
                # Calculate signal priority
                signal_priority = signal_logic.calculate_signal_priority()
                
                # Draw detection results on the frame
                result_frame = draw_detection_results(rgb_frame, results, num_lanes=config["num_lanes"])
                
                # Store the first few analyzed frames for display
                if len(analyzed_frames) < 5:  # Store only first 5 analyzed frames
                    timestamp = frame_count / fps  # Calculate timestamp in seconds
                    analyzed_frames.append({
                        'original': rgb_frame,
                        'detected': result_frame,
                        'timestamp': timestamp,
                        'lane_vehicle_counts': lane_vehicle_counts.copy(),
                        'lane_vehicle_types': lane_vehicle_types.copy(),
                        'lane_has_ambulance': lane_has_ambulance.copy(),
                        'signal_priority': signal_priority.copy()
                    })
            
            frame_count += 1
        
        # Close the video file
        cap.release()
        
        # Display the analyzed frames
        if analyzed_frames:
            for i, frame_data in enumerate(analyzed_frames):
                with frame_container:
                    st.markdown(f"#### Frame at {frame_data['timestamp']:.2f} seconds")
                    with frame_col1:
                        st.image(frame_data['original'], caption=f"Original Frame {i+1}", use_container_width=True)
                    with frame_col2:
                        st.image(frame_data['detected'], caption=f"Detection Results {i+1}", use_container_width=True)
                    
                    # Display traffic light decisions for each frame
                    with traffic_light_container:
                        st.markdown(f"#### Traffic Light Decisions at {frame_data['timestamp']:.2f} seconds")
                        
                        # Display traffic light and summary for each lane
                        for lane_id in range(config["num_lanes"]):
                            with lane_cols[lane_id]:
                                signal_time, signal_color = frame_data['signal_priority'][lane_id]
                                
                                # Display lane header with special styling for ambulance
                                if frame_data['lane_has_ambulance'][lane_id]:
                                    st.markdown(f"### 🚨 Lane {lane_id+1} (EMERGENCY)")
                                else:
                                    st.markdown(f"### Lane {lane_id+1}")
                                
                                # Draw and display traffic light
                                traffic_light_img = draw_traffic_light(state=signal_color)
                                st.image(traffic_light_img, caption=f"{signal_color}: {signal_time}s", width=100)
                                
                                # Vehicle count summary
                                st.write(f"**Total Vehicles:** {frame_data['lane_vehicle_counts'][lane_id]}")
                                
                                # Vehicle types breakdown
                                st.write("**Vehicle Types:**")
                                for vehicle_type, count in frame_data['lane_vehicle_types'][lane_id].items():
                                    st.write(f"- {vehicle_type}: {count}")
                                
                                # Ambulance alert
                                if frame_data['lane_has_ambulance'][lane_id]:
                                    st.error("⚠️ Emergency vehicle detected!")
            
            # Show summary of video analysis
            st.subheader("📊 Video Analysis Summary")
            st.write(f"- Total frames: {total_frames}")
            st.write(f"- Processed frames: {processed_count}")
            st.write(f"- Video duration: {total_frames/fps:.2f} seconds")
            st.write(f"- Frame rate: {fps:.2f} FPS")
            st.write(f"- Analysis interval: Every {frame_skip} frames ({fps/frame_skip:.2f} FPS)")
        else:
            st.error("No detection results available. The model may have failed to process the video.")
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        # Reset progress bar
        progress_bar.empty()
        status_text.empty()
