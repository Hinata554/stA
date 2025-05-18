import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io

def preprocess_image(image, size=300):
    """Convert image to grayscale, resize, and enhance contrast."""
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((size, size))  # Resize to square
    img_array = np.array(img)
    # Contrast enhancement
    img_array = (img_array - img_array.min()) * (255 / (img_array.max() - img_array.min()))
    return Image.fromarray(img_array.astype(np.uint8)), img_array

def generate_nails(num_nails, size):
    """Generate coordinates for nails around a circular frame."""
    center = size / 2
    radius = size / 2 - 10  # Inset from edge
    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)
    nails = [(center + radius * np.cos(a), center + radius * np.sin(a)) for a in angles]
    return nails

def bresenham_line(x0, y0, x1, y1):
    """Generate pixel coordinates for a line using Bresenham's algorithm."""
    pixels = []
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        pixels.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pixels

def compute_line_diff(img_array, nails, start_idx, end_idx, fade, size, used_pairs):
    """Compute the pixel difference for a line, inspired by JS get_line_diff."""
    if (start_idx, end_idx) in used_pairs or (end_idx, start_idx) in used_pairs:
        return None, float('inf')

    x1, y1 = nails[start_idx]
    x2, y2 = nails[end_idx]
    pixels = bresenham_line(x1, y1, x2, y2)

    total_diff = 0
    for x, y in pixels:
        if 0 <= x < size and 0 <= y < size:
            current_val = img_array[y, x]
            # Simulate new pixel value after adding line with fade
            new_val = 255 * fade + current_val * (1 - fade)  # White line on grayscale
            # Difference from original (simplified to grayscale)
            diff = abs(current_val - new_val) - abs(current_val - current_val)
            total_diff += diff if diff < 0 else diff / 5  # Penalize positive diffs less

    avg_diff = total_diff / len(pixels) if pixels else float('inf')
    return (start_idx, end_idx), avg_diff ** 3  # Cubic scaling as in JS

def generate_string_art(image, num_nails, max_connections, line_weight=1, fade=0.1):
    """Generate string art by drawing lines between nails."""
    size = 300
    img, img_array = preprocess_image(image, size)
    result_img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(result_img)
    
    # Generate nail positions
    nails = generate_nails(num_nails, size)
    
    # Draw nails
    for x, y in nails:
        draw.ellipse([x-3, y-3, x+3, y+3], fill='black')
    
    # Track used connections
    used_pairs = set()
    current_nail = 0
    connections = 0
    
    while connections < max_connections:
        min_diff = float('inf')
        best_pair = None
        
        # Find best next nail
        for i in range(num_nails):
            if i == current_nail:
                continue
            pair, diff = compute_line_diff(img_array, nails, current_nail, i, fade, size, used_pairs)
            if pair and diff < min_diff:
                min_diff = diff
                best_pair = pair
        
        if not best_pair or min_diff == float('inf'):
            break
        
        # Draw line
        start_idx, end_idx = best_pair
        used_pairs.add(best_pair)
        x1, y1 = nails[start_idx]
        x2, y2 = nails[end_idx]
        draw.line([(x1, y1), (x2, y2)], fill='black', width=line_weight)
        
        # Update image array (simulate line effect)
        pixels = bresenham_line(x1, y1, x2, y2)
        for x, y in pixels:
            if 0 <= x < size and 0 <= y < size:
                current_val = img_array[y, x]
                img_array[y, x] = min(255, current_val + int(255 * fade))
        
        current_nail = end_idx
        connections += 1
    
    return result_img

# Streamlit app
st.title("String Art Generator")
st.write("Upload an image to create a string art pattern. Adjust parameters to customize the output.")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Parameters
    num_nails = st.slider("Number of Nails", 50, 500, 300, step=10)
    max_connections = st.slider("Max Number of Connections", 100, 15000, 10000, step=100)
    line_weight = st.slider("Line Weight", 1, 5, 1)
    fade = st.slider("Line Fade (Opacity)", 0.01, 1.0, 0.1, step=0.01)
    
    if st.button("Generate String Art"):
        with st.spinner("Generating string art..."):
            result_img = generate_string_art(image, num_nails, max_connections, line_weight, fade)
            st.image(result_img, caption="Generated String Art", use_column_width=True)
            
            # Allow download
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download String Art",
                data=byte_im,
                file_name="string_art.png",
                mime="image/png"
            )
