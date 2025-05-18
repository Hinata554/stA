import streamlit as st
import numpy as np
import cv2
import math
from io import BytesIO

def place_pins(n_pins, radius=300, center=(300, 300)):
    angle = 2 * np.pi / n_pins
    return [
        (
            int(center[0] + radius * np.cos(i * angle)),
            int(center[1] + radius * np.sin(i * angle))
        )
        for i in range(n_pins)
    ]

def line_intensity(img, p1, p2):
    h, w = img.shape
    line_iter = np.linspace(p1, p2, num=100).astype(int)
    line_iter[:, 0] = np.clip(line_iter[:, 0], 0, w - 1)
    line_iter[:, 1] = np.clip(line_iter[:, 1], 0, h - 1)
    intensities = img[line_iter[:, 1], line_iter[:, 0]]
    return np.mean(intensities)

def draw_line(canvas, p1, p2):
    return cv2.line(canvas, p1, p2, color=255, thickness=1)

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
    canvas = np.zeros_like(target)  # black background
    sequence = [0]  # start with pin 0
    current_pin = 0

    progress = st.progress(0)

    for i in range(num_connections):
        best_pin = None
        best_score = -1

        for j in range(num_pins):
            if j == current_pin:
                continue
            score = line_intensity(target, pins[current_pin], pins[j])
            if score > best_score:
                best_score = score
                best_pin = j

        draw_line(canvas, pins[current_pin], pins[best_pin])
        cv2.line(target, pins[current_pin], pins[best_pin], color=0, thickness=1)

        sequence.append(best_pin)
        current_pin = best_pin
        progress.progress((i + 1) / num_connections)

    st.image(canvas, caption="Generated String Art", use_column_width=True)

    st.markdown("### Pin Sequence")
    sequence_text = ", ".join(str(x) for x in sequence)
    st.text_area("Sequence", sequence_text, height=200)

    buffer = BytesIO()
    buffer.write(sequence_text.encode())
    buffer.seek(0)
    st.download_button("Download Sequence", buffer, file_name="string_art_sequence.txt")
