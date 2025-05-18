import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from io import BytesIO
import cv2

def image_to_string_art(image, pins, lines, radius=1.0):
    """Convert image to string art using improved algorithm."""
    # Convert image to grayscale and resize
    img = np.array(image.convert('L'))
    img = cv2.resize(img, (300, 300))  # Fixed size for processing
    img = 255 - img  # Invert (darker areas will get more lines)
    
    # Create pin positions
    angles = np.linspace(0, 2*np.pi, pins, endpoint=False)
    x_pins = radius * np.cos(angles)
    y_pins = radius * np.sin(angles)
    
    # Convert pin positions to image coordinates
    h, w = img.shape
    pin_coords = []
    for x, y in zip(x_pins, y_pins):
        # Map from circle (-1 to 1) to image coordinates (0 to w-1)
        img_x = int((x + radius) * (w - 1) / (2 * radius))
        img_y = int((-y + radius) * (h - 1) / (2 * radius))
        pin_coords.append((img_x, img_y))
    
    # Generate lines based on image brightness
    segments = []
    line_colors = []
    current_pin = 0
    visited = np.zeros(img.shape, dtype=np.uint8)
    
    for _ in range(lines):
        best_next_pin = None
        best_score = -1
        best_line_coords = []
        
        # Try several possible next pins
        for offset in range(1, min(pins//2, 100)):  # Limit search range
            next_pin = (current_pin + offset) % pins
            x1, y1 = pin_coords[current_pin]
            x2, y2 = pin_coords[next_pin]
            
            # Get line coordinates (Bresenham's algorithm)
            line_coords = list(zip(*line_aa(y1, x1, y2, x2)))
            if not line_coords:
                continue
                
            # Calculate score based on unvisited dark areas
            score = 0
            for y, x, a in line_coords:
                if 0 <= y < h and 0 <= x < w:
                    score += img[int(y), int(x)] * (1 - visited[int(y), int(x)])
            
            if score > best_score:
                best_score = score
                best_next_pin = next_pin
                best_line_coords = line_coords
        
        if best_next_pin is None:
            break
            
        # Add the best line
        x1, y1 = x_pins[current_pin], y_pins[current_pin]
        x2, y2 = x_pins[best_next_pin], y_pins[best_next_pin]
        segments.append([(x1, y1), (x2, y2)])
        
        # Mark visited pixels
        for y, x, a in best_line_coords:
            if 0 <= y < h and 0 <= x < w:
                visited[int(y), int(x)] += 0.1
                visited[int(y), int(x)] = min(1.0, visited[int(y), int(x)])
        
        # Calculate line color based on average brightness
        avg_brightness = np.mean([img[int(y), int(x)] for y, x, a in best_line_coords 
                                if 0 <= y < h and 0 <= x < w]) / 255
        line_colors.append((0, 0, 0, 1 - avg_brightness))
        
        current_pin = best_next_pin
    
    return segments, x_pins, y_pins, line_colors

def line_aa(x0, y0, x1, y1):
    """Generate anti-aliased line coordinates."""
    # This is a simplified version - in production you'd use a proper AA line algorithm
    # or import from skimage.draw
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 1
    
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = 1 - (x0 + 0.5) % 1
    xpxl1 = xend
    ypxl1 = int(yend)
    points.append((ypxl1, xpxl1, 1 - (yend % 1) * xgap))
    points.append((ypxl1 + 1, xpxl1, (yend % 1) * xgap))
    
    intery = yend + gradient
    
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = (x1 + 0.5) % 1
    xpxl2 = xend
    ypxl2 = int(yend)
    points.append((ypxl2, xpxl2, 1 - (yend % 1) * xgap))
    points.append((ypxl2 + 1, xpxl2, (yend % 1) * xgap))
    
    for x in range(xpxl1 + 1, xpxl2):
        points.append((int(intery), x, 1 - (intery % 1)))
        points.append((int(intery) + 1, x, intery % 1))
        intery += gradient
    
    return points

def main():
    st.title("Improved Image to String Art Converter")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Sidebar controls
        st.sidebar.header("Parameters")
        pins = st.sidebar.slider("Number of pins", 50, 300, 150, 5)
        lines = st.sidebar.slider("Number of lines", 100, 10000, 2000, 100)
        line_width = st.sidebar.slider("Line width", 0.1, 3.0, 0.8, 0.1)
        bg_color = st.sidebar.color_picker("Background color", "#ffffff")
        pin_size = st.sidebar.slider("Pin size", 0, 20, 0, 1)
        show_pins = st.sidebar.checkbox("Show pins", False)
        contrast = st.sidebar.slider("Contrast boost", 1.0, 3.0, 1.5, 0.1)
        
        # Process image with contrast adjustment
        img_array = np.array(image.convert('L'))
        img_array = cv2.resize(img_array, (300, 300))
        img_array = np.clip((img_array - 128) * contrast + 128, 0, 255).astype(np.uint8)
        processed_image = Image.fromarray(img_array)
        
        # Generate string art
        with st.spinner('Generating string art...'):
            segments, x_pins, y_pins, line_colors = image_to_string_art(
                processed_image, pins, lines, radius=0.95
            )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_facecolor(bg_color)
        ax.axis('off')
        
        # Add string art lines with varying colors
        lc = LineCollection(segments, linewidths=line_width, colors=line_colors)
        ax.add_collection(lc)
        
        # Add pins if enabled
        if show_pins and pin_size > 0:
            ax.scatter(x_pins, y_pins, s=pin_size, color='black', zorder=3)
        
        # Display plot
        with col2:
            st.subheader("String Art Result")
            st.pyplot(fig)
        
        # Add download button
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
        st.download_button(
            label="Download String Art",
            data=buf.getvalue(),
            file_name="string_art.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
