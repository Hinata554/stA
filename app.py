import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def generate_string_art(pins, lines, step, radius=1.0):
    """Generate string art coordinates in a circular pattern."""
    angles = np.linspace(0, 2*np.pi, pins, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    segments = []
    for i in range(lines):
        start_idx = i % pins
        end_idx = (start_idx + step * (i + 1)) % pins
        segments.append([(x[start_idx], y[start_idx]), (x[end_idx], y[end_idx])])
    
    return segments, x, y

def main():
    st.title("Circular String Art Generator")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    pins = st.sidebar.slider("Number of pins", 10, 200, 60, 1)
    lines = st.sidebar.slider("Number of lines", 10, 2000, 300, 1)
    step = st.sidebar.slider("Step size", 1, 50, 5, 1)
    line_width = st.sidebar.slider("Line width", 0.1, 5.0, 1.0, 0.1)
    line_color = st.sidebar.color_picker("Line color", "#3498db")
    bg_color = st.sidebar.color_picker("Background color", "#f5f5f5")
    pin_size = st.sidebar.slider("Pin size", 0, 20, 5, 1)
    show_pins = st.sidebar.checkbox("Show pins", True)
    
    # Generate string art
    segments, x_pins, y_pins = generate_string_art(pins, lines, step)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    # Add string art lines
    lc = LineCollection(segments, linewidths=line_width, colors=line_color, alpha=0.8)
    ax.add_collection(lc)
    
    # Add pins if enabled
    if show_pins:
        ax.scatter(x_pins, y_pins, s=pin_size, color='black', zorder=3)
    
    # Display plot in Streamlit
    st.pyplot(fig)
    
    # Add some information
    st.markdown("""
    ### How to Use
    - Adjust the sliders to change the string art pattern
    - The pattern is created by connecting pins in a circular sequence
    - **Number of pins**: Total points around the circle
    - **Number of lines**: Total connections to draw
    - **Step size**: Determines the pattern complexity (try prime numbers for interesting effects)
    """)

if __name__ == "__main__":
    main()
