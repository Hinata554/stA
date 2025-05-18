import streamlit as st
import numpy as np
import cv2
import math
import random
from io import BytesIO

# ---- Pin placement ----
def place_pins(n_pins, radius=280, center=(300, 300)):
    """Place pins evenly around a circle."""
    angle = 2 * np.pi / n_pins
    return [
        (
            int(center[0] + radius * np.cos(i * angle)),
            int(center[1] + radius * np.sin(i * angle))
        )
        for i in range(n_pins)
    ]

# ---- Line intensity function (with bounds check) ----
def line_intensity(img, p1, p2):
    """Calculate how well a line between two pins matches the target image.
    Higher return value = better match to dark areas of the image."""
    h, w = img.shape
    # Get more points for better sampling
    num_samples = max(int(np.linalg.norm(np.array(p1) - np.array(p2))) * 2, 100)
    line_iter = np.linspace(p1, p2, num=num_samples).astype(int)
    line_iter[:, 0] = np.clip(line_iter[:, 0], 0, w - 1)
    line_iter[:, 1] = np.clip(line_iter[:, 1], 0, h - 1)
    
    # Get intensity values along the line
    intensities = img[line_iter[:, 1], line_iter[:, 0]]
    
    # For intensity, we want higher scores for darker pixels
    # (0 = black, 255 = white in grayscale images)
    # So we invert the values: darker pixels = higher score
    return np.sum(255 - intensities) / len(intensities)

def draw_line(canvas, p1, p2):
    return cv2.line(canvas, p1, p2, color=0, thickness=1)

def apply_opacity(image, opacity_percent):
    """Adjust the opacity of strings in the image"""
    # Convert opacity percentage to a value between 0-255
    opacity_value = int((opacity_percent / 100) * 255)
    
    # Create a black background
    background = np.zeros_like(image)
    
    # Blend the strings with the background based on opacity
    # For white strings on black background: reduce white intensity
    result = background.copy()
    # Find all non-black pixels (string pixels)
    string_mask = image < 255
    # Set those pixels to the opacity value
    result[string_mask] = opacity_value
    
    return result

# ---- Streamlit App UI ----
st.title("🧵 String Art Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)
with col1:
    num_pins = st.slider("Number of Pins", 50, 300, 200, step=10)
with col2:
    num_connections = st.slider("Number of String Connections", 500, 5000, 1000, step=100)

# Preview opacity setting
preview_opacity = st.slider("Preview String Opacity (%)", 10, 100, 30, step=10)

if uploaded_file:
    # Load and process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (600, 600))
    
    # Display the original image
    st.image(img, caption="Original Image", use_column_width=True)
    
    # Process the image to enhance contrast
    processed_img = cv2.equalizeHist(img)  # Enhance contrast
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)  # Slight blur to reduce noise
    
    # Create target image where darker pixels will attract more strings
    target = processed_img.copy()

    # Draw the pin positions on the canvas before starting
    pin_visualization = np.zeros((600, 600, 3), dtype=np.uint8)  # BGR format
    
    # Draw the circular boundary
    cv2.circle(pin_visualization, (300, 300), 280, (50, 50, 50), 1)
    
    # Draw pins as small circles
    for pin_idx, pin_pos in enumerate(pins):
        cv2.circle(pin_visualization, pin_pos, 2, (0, 165, 255), -1)  # Orange dots
        # Add pin numbers every 25 pins
        if pin_idx % 25 == 0:
            cv2.putText(pin_visualization, str(pin_idx), 
                       (pin_pos[0] + 5, pin_pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    st.image(pin_visualization, caption=f"Pin Configuration ({num_pins} pins)", use_column_width=True)
    
    # Keep track of recently used connections to avoid loops
    recent_connections = set()
    connection_memory = num_pins // 4  # Remember more connections for better variety
    
    # Keep track of total contributions per pixel to avoid oversaturation
    contribution_map = np.zeros_like(target, dtype=np.float32)
    max_contribution = 10.0  # Cap on how "used up" a pixel can be

    for i in range(num_connections):
        best_pin = None
        best_score = -1
        candidates = []

        for j in range(num_pins):
            if j == current_pin:
                continue
                
            # Check if this connection was recently used
            connection = (current_pin, j)
            if connection in recent_connections:
                continue
            
            # Create a mask for this line to evaluate contribution
            line_mask = np.zeros_like(target, dtype=np.uint8)
            cv2.line(line_mask, pins[current_pin], pins[j], color=1, thickness=1)
            
            # Calculate the base intensity score from the image
            # Higher score means line goes through darker areas of image
            base_score = line_intensity(target, pins[current_pin], pins[j])
            
            # Calculate contribution score - prefer lines that go through less used areas
            # Get current contribution values along the line
            line_points = np.where(line_mask > 0)
            if len(line_points[0]) > 0:
                current_contributions = contribution_map[line_points]
                # Less saturated areas get higher scores
                contribution_score = np.sum(max_contribution - np.minimum(current_contributions, max_contribution))
                
                # Combined score - balance between matching image and using fresh areas
                score = base_score * 0.7 + contribution_score * 0.3
                
                candidates.append((j, score))
            
        # Sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Choose from top candidates to avoid getting stuck
        if candidates:
            # Take one of the top 3 candidates (or fewer if less available)
            top_n = min(3, len(candidates))
            idx = random.randint(0, top_n-1)
            best_pin = candidates[idx][0]
        else:
            # If no candidates, choose a random pin as fallback
            best_pin = random.choice([p for p in range(num_pins) if p != current_pin])
        
        # Add this connection to recent memory
        connection = (current_pin, best_pin)
        recent_connections.add(connection)
        
        # Remove oldest connection if we're over the limit
        if len(recent_connections) > connection_memory:
            recent_connections.pop()

        # Draw line on canvas and update contribution map
        draw_line(canvas, pins[current_pin], pins[best_pin])
        
        # Update the contribution map - mark this line as used
        line_mask = np.zeros_like(target, dtype=np.uint8)
        cv2.line(line_mask, pins[current_pin], pins[best_pin], color=1, thickness=1)
        # Increment the contribution value along the line
        contribution_map[line_mask > 0] += 1.0
        
        # Update progress
        progress_bar.progress((i + 1) / num_connections)
        if i % 50 == 0 or i == num_connections - 1:
            status_text.text(f"Creating string art: {i+1}/{num_connections}")
            # Preview the result every 50 iterations
            if i % 500 == 0 and i > 0:
                preview = 255 - canvas.copy()
                # Apply opacity to the preview image
                preview_with_opacity = apply_opacity(preview, preview_opacity)
                st.image(preview_with_opacity, caption=f"Preview after {i+1} strings", use_column_width=True)

        sequence.append(current_pin)
        current_pin = best_pin
    
    
    # Add the final pin to complete the sequence
    sequence.append(current_pin)

    # Invert the canvas to show black background with white lines
    final_image = 255 - canvas
    
    # Apply opacity to the final result
    final_image_with_opacity = apply_opacity(final_image, preview_opacity)
    
    st.image(final_image_with_opacity, caption="Generated String Art", use_column_width=True)
    
    # Also provide the full opacity version for download
    st.image(final_image, caption="Generated String Art (100% Opacity)", use_column_width=True)

    st.markdown("### Pin Sequence")
    sequence_text = ", ".join(map(str, sequence))
    st.text_area("Sequence", sequence_text, height=200)

    buffer = BytesIO()
    buffer.write(sequence_text.encode())
    buffer.seek(0)
    st.download_button("Download Sequence", buffer, file_name="string_art_sequence.txt")
