import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from io import BytesIO
import cv2

def image_to_string_art(image, pins, lines, step, radius=1.0):
    """Convert image to string art coordinates in a circular pattern."""
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
        img_x = int((x + 1) * (w - 1) / 2)
        img_y = int((-y + 1) * (h - 1) / 2)  # Negative y because image y-axis goes down
        pin_coords.append((img_x, img_y))
    
    # Generate lines based on image brightness
    segments = []
    current_pin = 0
    line_colors = []
    
    for _ in range(lines):
        best_next_pin = None
        best_score = -1
        
        # Try several possible next pins
        for offset in range(1, pins//2):
            next_pin = (current_pin + offset) % pins
            x1, y1 = pin_coords[current_pin]
            x2, y2 = pin_coords[next_pin]
            
            # Sample points along the line
            length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            x = np.linspace(x1, x2, length)
            y = np.linspace(y1, y2, length)
            
            # Get pixel values along the line
            try:
                values = img[np.round(y).astype(int), np.round(x).astype(int)]
                score = np.mean(values)
            except:
                score = 0
            
            if score > best_score:
                best_score = score
                best_next_pin = next_pin
        
        # Add the best line
        x1, y1 = x_pins[current_pin], y_pins[current_pin]
        x2, y2 = x_pins[best_next_pin], y_pins[best_next_pin]
        segments.append([(x1, y1), (x2, y2)])
        
        # Store color based on brightness
        brightness = 1 - (best_score / 255)
        line_colors.append((brightness, brightness, brightness, 0.7))
        
        current_pin = best_next_pin
    
    return segments, x_pins, y_pins, line_colors

def main():
    st.title("Image to String Art Converter")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Sidebar controls
        st.sidebar.header("Parameters")
        pins = st.sidebar.slider("Number of pins", 10, 200, 60, 1)
        lines = st.sidebar.slider("Number of lines", 100, 5000, 1000, 50)
        step = st.sidebar.slider("Step size", 1, 50, 5, 1)
        line_width = st.sidebar.slider("Line width", 0.1, 5.0, 1.0, 0.1)
        bg_color = st.sidebar.color_picker("Background color", "#f5f5f5")
        pin_size = st.sidebar.slider("Pin size", 0, 20, 5, 1)
        show_pins = st.sidebar.checkbox("Show pins", True)
        
        # Generate string art
        segments, x_pins, y_pins, line_colors = image_to_string_art(image, pins, lines, step)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_facecolor(bg_color)
        ax.axis('off')
        
        # Add string art lines with varying colors
        lc = LineCollection(segments, linewidths=line_width, colors=line_colors)
        ax.add_collection(lc)
        
        # Add pins if enabled
        if show_pins:
            ax.scatter(x_pins, y_pins, s=pin_size, color='black', zorder=3)
        
        # Display plot in Streamlit
        st.subheader("String Art Result")
        st.pyplot(fig)
        
        # Add download button
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight', pad_inches=0)
        st.download_button(
            label="Download String Art",
            data=buf.getvalue(),
            file_name="string_art.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
