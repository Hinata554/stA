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
    
    num_pins = st.slider("Number of pins", min_value=20, max_value=200, value=100, step=10)
    
    num_lines = st.slider("Number of lines", min_value=100, max_value=5000, value=1000, step=100)
    
    pin_connection_method = st.selectbox(
        "Pin connection method",
        ["Greedy Algorithm", "Random Connections"]
    )
    
    line_color = st.color_picker("Line color", "#FFFFFF")
    
    background_color = st.color_picker("Background color", "#000000")
    
    line_thickness = st.slider("Line thickness", min_value=1, max_value=5, value=1)
    
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

def calculate_line_darkness(img, start, end, num_samples=20):
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
    
    # Calculate the average darkness along the line
    total_darkness = 0
    for x, y in points:
        total_darkness += get_pixel_intensity(img, x, y)
    
    return total_darkness / (num_samples + 1)

def greedy_string_art(img, pins, num_lines, current_canvas=None):
    """Generate string art connections using a greedy algorithm."""
    radius = img.shape[0] // 2
    connections = []
    
    # If no current canvas, start with a blank one
    if current_canvas is None:
        current_canvas = np.zeros_like(img)
    
    # Keep track of the lines we've added
    added_lines = set()
    
    for _ in range(num_lines):
        best_darkness = -1
        best_connection = None
        
        # Try a sample of possible connections to speed things up
        sample_size = min(len(pins) * 20, len(pins) * (len(pins) - 1) // 2)
        pin_indices = np.random.choice(len(pins), size=(sample_size, 2), replace=True)
        
        for i, j in pin_indices:
            if i == j or (i, j) in added_lines or (j, i) in added_lines:
                continue
                
            start = pins[i]
            end = pins[j]
            
            # Calculate how much darkness this line would add
            darkness = calculate_line_darkness(img, start, end)
            
            if darkness > best_darkness:
                best_darkness = darkness
                best_connection = (i, j)
        
        if best_connection:
            connections.append((pins[best_connection[0]], pins[best_connection[1]]))
            added_lines.add(best_connection)
            
            # Update the canvas by drawing this line
            cv2.line(current_canvas, pins[best_connection[0]], pins[best_connection[1]], 
                     color=255, thickness=1)
    
    return connections

def random_string_art(img, pins, num_lines):
    """Generate string art connections using weighted random selection."""
    connections = []
    
    # Create all possible pin pairs
    pin_pairs = []
    darkness_values = []
    
    # Sample a subset of pairs to speed up processing
    sample_size = min(10000, len(pins) * (len(pins) - 1) // 2)
    pin_indices = np.random.choice(len(pins), size=(sample_size, 2), replace=True)
    
    for i, j in pin_indices:
        if i != j:
            start = pins[i]
            end = pins[j]
            darkness = calculate_line_darkness(img, start, end)
            pin_pairs.append((start, end))
            darkness_values.append(darkness)
    
    # Convert to numpy array and normalize
    darkness_values = np.array(darkness_values)
    if darkness_values.sum() > 0:  # Avoid division by zero
        probs = darkness_values / darkness_values.sum()
        
        # Randomly select connections based on darkness
        chosen_indices = np.random.choice(
            len(pin_pairs), 
            size=min(num_lines, len(pin_pairs)), 
            replace=False, 
            p=probs
        )
        
        for idx in chosen_indices:
            connections.append(pin_pairs[idx])
    
    return connections

def preprocess_image(uploaded_file, size=500):
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
            processed_img = preprocess_image(uploaded_file)
            
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
            circle = plt.Circle((radius, radius), radius, fill=False, color='gray', alpha=0.5)
            ax.add_patch(circle)
            
            # Draw the strings
            line_rgb = hex_to_rgb(line_color)
            line_rgb_norm = [x/255 for x in line_rgb]
            
            for start, end in connections:
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=line_rgb_norm, 
                       linewidth=line_thickness, 
                       alpha=0.6)
            
            # Set limits and remove axes
            ax.set_xlim(0, processed_img.shape[1])
            ax.set_ylim(0, processed_img.shape[0])
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Display the result
            st.pyplot(fig)
            
            # Create download button for SVG
            svg_buffer = download_svg(fig)
            svg_bytes = svg_buffer.getvalue()
            b64 = base64.b64encode(svg_bytes).decode()
            href = f'data:image/svg+xml;base64,{b64}'
            st.download_button(
                label="Download SVG",
                data=svg_bytes,
                file_name="string_art.svg",
                mime="image/svg+xml"
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
