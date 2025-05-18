import streamlit as st
import numpy as np
import cv2
import math
import random
from io import BytesIO

# ---- Pin placement ----
def place_pins(n_pins, radius=300, center=(300, 300)):
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
    h, w = img.shape
    line_iter = np.linspace(p1, p2, num=100).astype(int)
    line_iter[:, 0] = np.clip(line_iter[:, 0], 0, w - 1)
    line_iter[:, 1] = np.clip(line_iter[:, 1], 0, h - 1)
    intensities = img[line_iter[:, 1], line_iter[:, 0]]
    return np.mean(intensities)

def draw_line(canvas, p1, p2):
    return cv2.line(canvas, p1, p2, color=255, thickness=1)

# ---- Streamlit App UI ----
st.title("🧵 String Art Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
num_pins = st.slider("Number of Pins", 50, 300, 200, step=10)
num_connections = st.slider("Number of String Connections", 500, 5000, 1000, step=100)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (600, 600))

    target = 255 - cv2.GaussianBlur(img, (7, 7), 0)

    pins = place_pins(num_pins)
    canvas = np.ones_like(target) * 255
    sequence = []
    current_pin = 0
    
    # Keep track of recently used connections to avoid loops
    recent_connections = set()
    connection_memory = 10  # How many recent connections to remember

    progress = st.progress(0)

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
                
            score = line_intensity(target, pins[current_pin], pins[j])
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

        draw_line(canvas, pins[current_pin], pins[best_pin])
        cv2.line(target, pins[current_pin], pins[best_pin], color=0, thickness=1)

        sequence.append(current_pin)
        current_pin = best_pin
        progress.progress((i + 1) / num_connections)
    
    # Add the final pin to complete the sequence
    sequence.append(current_pin)

    st.image(canvas, caption="Generated String Art", use_column_width=True)

    st.markdown("### Pin Sequence")
    sequence_text = ", ".join(map(str, sequence))
    st.text_area("Sequence", sequence_text, height=200)

    buffer = BytesIO()
    buffer.write(sequence_text.encode())
    buffer.seek(0)
    st.download_button("Download Sequence", buffer, file_name="string_art_sequence.txt")
