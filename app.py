import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64
from matplotlib.figure import Figure

st.set_page_config(layout="wide", page_title="String Art Generator")

st.title("String Art Generator")
st.write("Create beautiful string art from your images using a circle of pins.")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    num_pins = st.slider("Number of pins", min_value=20, max_value=400, value=180, step=20)
    
    num_lines = st.slider("Number of lines", min_value=100, max_value=10000, value=2000, step=100)
    
    pin_connection_method = st.selectbox(
        "Pin connection method",
        ["Greedy Algorithm", "Random Connections"]
    )
    
    image_preprocessing = st.selectbox(
        "Image preprocessing",
        ["Normal", "High Contrast", "Edge Detection"]
    )
    
    line_color = st.color_picker("Line color", "#FFFFFF")
    
    background_color = st.color_picker("Background color", "#000000")
    
    line_thickness = st.slider("Line thickness", min_value=1, max_value=5, value=1)
    
    advanced_options = st.expander("Advanced Options")
    with advanced_options:
        edge_threshold = st.slider("Edge Threshold", 0, 255, 100, help="For edge detection preprocessing")
        contrast_factor = st.slider("Contrast Enhancement", 0.5, 3.0, 1.5, help="For high contrast preprocessing")
    
    if uploaded_file is not None:
        download_button = st.button("Generate String Art")
    else:
        download_button = False

# Main functions for string art generation
def create_circle_pins(num_pins, radius):
    """Create pins arranged in a circle."""
    pins = []
    for i in range(num_pins):
        angle = 2 * np.pi * i / num_pins
        x = int(radius + radius * np.cos(angle))
        y = int(radius + radius * np.sin(angle))
        pins.append((x, y))
    return pins

def get_pixel_intensity(img, x, y):
    """Get intensity of a pixel (inverted so dark areas have higher values)."""
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        # Return inverted grayscale value (dark areas are more important)
        return 255 - img[y, x]
    return 0

def calculate_line_darkness(img, start, end, num_samples=50):
    """Calculate how much darkness a line would add."""
    x1, y1 = start
    x2, y2 = end
    
    # Sample points along the line
    points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        x = int((1 - t) * x1 + t * x2)
        y = int((1 - t) * y1 + t * y2)
        points.append((x, y))
    
    # Calculate the weighted darkness along the line
    # Prioritize points with higher darkness values for better feature representation
    darkness_values = []
    for x, y in points:
        darkness_values.append(get_pixel_intensity(img, x, y))
    
    # Sort darkness values and give higher weight to the darker points (top 30%)
    darkness_values.sort(reverse=True)
    top_values = darkness_values[:int(len(darkness_values) * 0.3)]
    
    if not top_values:  # In case the list is empty
        return 0
        
    return sum(top_values) / len(top_values)

def greedy_string_art(img, pins, num_lines, current_canvas=None):
    """Generate string art connections using a greedy algorithm."""
    radius = img.shape[0] // 2
    connections = []
    
    # If no current canvas, start with a blank one
    if current_canvas is None:
        current_canvas = np.zeros_like(img)
    
    # Keep track of the lines we've added
    added_lines = set()
    
    # Create a residual image that we'll update as we draw
    residual_img = img.copy()
    
    for _ in range(num_lines):
        best_darkness = -1
        best_connection = None
        
        # Try all possible connections (can be optimized for speed)
        for i in range(len(pins)):
            for j in range(i+1, len(pins)):
                if (i, j) in added_lines or (j, i) in added_lines:
                    continue
                    
                start = pins[i]
                end = pins[j]
                
                # Calculate how much darkness this line would add
                darkness = calculate_line_darkness(residual_img, start, end)
                
                if darkness > best_darkness:
                    best_darkness = darkness
                    best_connection = (i, j)
        
        if best_connection:
            connections.append((pins[best_connection[0]], pins[best_connection[1]]))
            added_lines.add(best_connection)
            
            # Update the canvas by drawing this line
            cv2.line(current_canvas, pins[best_connection[0]], pins[best_connection[1]], 
                     color=255, thickness=1)
            
            # Subtract this line from the residual image (reduce the darkness along this line)
            # This helps prevent overemphasizing the same areas and improves distribution
            temp = np.zeros_like(residual_img)
            cv2.line(temp, pins[best_connection[0]], pins[best_connection[1]], 
                     color=30, thickness=2)
            residual_img = cv2.subtract(residual_img, temp)
    
    return connections

def random_string_art(img, pins, num_lines):
    """Generate string art connections using weighted random selection."""
    connections = []
    residual_img = img.copy()
    added_lines = set()
    
    # Create all possible pin pairs
    all_pairs = []
    for i in range(len(pins)):
        for j in range(i+1, len(pins)):
            all_pairs.append((i, j))
    
    for _ in range(num_lines):
        pin_pairs = []
        darkness_values = []
        
        # Sample a subset of pairs to speed up processing
        sample_size = min(1000, len(all_pairs))
        sample_indices = np.random.choice(len(all_pairs), size=sample_size, replace=False)
        
        for idx in sample_indices:
            i, j = all_pairs[idx]
            if (i, j) in added_lines or (j, i) in added_lines:
                continue
                
            start = pins[i]
            end = pins[j]
            darkness = calculate_line_darkness(residual_img, start, end)
            pin_pairs.append((i, j, start, end))
            darkness_values.append(darkness)
        
        # If no valid pairs left, break
        if not darkness_values:
            break
            
        # Convert to numpy array and normalize
        darkness_values = np.array(darkness_values)
        if darkness_values.sum() > 0:  # Avoid division by zero
            # Use a power function to emphasize the differences
            darkness_values = darkness_values ** 2
            probs = darkness_values / darkness_values.sum()
            
            # Randomly select a connection based on darkness
            chosen_idx = np.random.choice(len(pin_pairs), p=probs)
            i, j, start, end = pin_pairs[chosen_idx]
            
            connections.append((start, end))
            added_lines.add((i, j))
            
            # Subtract this line from the residual image
            temp = np.zeros_like(residual_img)
            cv2.line(temp, start, end, color=30, thickness=2)
            residual_img = cv2.subtract(residual_img, temp)
    
    return connections

def preprocess_image(uploaded_file, size=500, preprocessing_method="Normal", edge_threshold=100, contrast_factor=1.5):
    """Preprocess the uploaded image."""
    # Read image
    image = Image.open(uploaded_file)
    
    # Convert to square and resize
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    image = image.crop((left, top, right, bottom))
    image = image.resize((size, size))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply different preprocessing based on selected method
    if preprocessing_method == "High Contrast":
        # Enhance contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=contrast_factor, beta=0)
        img_array = cv2.equalizeHist(img_array)
    elif preprocessing_method == "Edge Detection":
        # Apply edge detection
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        img_array = cv2.Canny(img_array, edge_threshold, edge_threshold * 2)
    
    # Apply circular mask
    mask = np.zeros_like(img_array)
    center = size // 2
    cv2.circle(mask, (center, center), center, 255, -1)
    img_array = cv2.bitwise_and(img_array, mask)
    
    return img_array

def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def download_svg(fig, filename="string_art.svg"):
    """Convert matplotlib figure to SVG for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    return buf

# Main app logic
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Original Image")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    st.header("String Art Preview")
    if uploaded_file is not None and download_button:
        with st.spinner("Generating string art..."):
            # Process image
            processed_img = preprocess_image(
                uploaded_file, 
                preprocessing_method=image_preprocessing,
                edge_threshold=edge_threshold,
                contrast_factor=contrast_factor
            )
            
            # Display processed image in a small corner
            st.image(processed_img, caption="Processed Image", width=200)
            
            # Create pins
            radius = processed_img.shape[0] // 2
            pins = create_circle_pins(num_pins, radius)
            
            # Generate connections
            if pin_connection_method == "Greedy Algorithm":
                connections = greedy_string_art(processed_img, pins, num_lines)
            else:
                connections = random_string_art(processed_img, pins, num_lines)
            
            # Create figure
            fig = Figure(figsize=(10, 10), dpi=100)
            ax = fig.add_subplot(111)
            
            # Set background color
            bg_rgb = hex_to_rgb(background_color)
            bg_rgb_norm = [x/255 for x in bg_rgb]
            fig.patch.set_facecolor(bg_rgb_norm)
            ax.set_facecolor(bg_rgb_norm)
            
            # Draw pins and connections
            circle = plt.Circle((radius, radius), radius, fill=False, color='gray', alpha=0.2)
            ax.add_patch(circle)
            
            # Draw the strings
            line_rgb = hex_to_rgb(line_color)
            line_rgb_norm = [x/255 for x in line_rgb]
            
            for start, end in connections:
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=line_rgb_norm, 
                       linewidth=line_thickness, 
                       alpha=0.8)
            
            # Set limits and remove axes
            ax.set_xlim(0, processed_img.shape[1])
            ax.set_ylim(processed_img.shape[0], 0)  # Invert y-axis to match image coordinates
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Display the result
            st.pyplot(fig)
            
            # Create download buttons
            svg_buffer = download_svg(fig)
            svg_bytes = svg_buffer.getvalue()
            b64 = base64.b64encode(svg_bytes).decode()
            href = f'data:image/svg+xml;base64,{b64}'
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download SVG",
                    data=svg_bytes,
                    file_name="string_art.svg",
                    mime="image/svg+xml"
                )
            
            # Save as PNG also
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
            png_buffer.seek(0)
            with col2:
                st.download_button(
                    label="Download PNG",
                    data=png_buffer,
                    file_name="string_art.png",
                    mime="image/png"
                )

# Show instructions if no image is uploaded
if uploaded_file is None:
    st.markdown("""
    ## How to use this String Art Generator:
    
    1. **Upload an image** using the file uploader in the sidebar
    2. **Adjust settings** in the sidebar:
       - Number of pins around the circle
       - Number of lines to create the artwork
       - Connection method (greedy algorithm finds darker areas first)
       - Line and background colors
       - Line thickness
    3. **Click 'Generate String Art'** to create your artwork
    4. **Download** your creation as an SVG file
    
    The algorithm will try to recreate your image using only straight lines between pins arranged in a circle.
    """)
    
    # Display an example image
    st.image("https://i.imgur.com/7A2vamc.png", caption="Example String Art", width=300)
