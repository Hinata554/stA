import streamlit as st
import numpy as np
import io
import base64
from PIL import Image, ImageDraw

def generate_string_art(pins_count, lines_count, sequence, image_size=500):
    """
    Generates string art based on a given sequence, within a circular arrangement of pins.

    Args:
        pins_count (int): The number of pins on the circle's circumference.
        lines_count (int): The number of lines to draw.
        sequence (list of int): The sequence of pin connections.
        image_size (int, optional): The size of the output image (square). Defaults to 500.

    Returns:
        Image.Image: A PIL Image object containing the generated string art.
    """
    # Input validation
    if not isinstance(pins_count, int) or pins_count <= 1:
        raise ValueError("pins_count must be an integer greater than 1")
    if not isinstance(lines_count, int) or lines_count <= 0:
        raise ValueError("lines_count must be a positive integer")
    if not isinstance(sequence, list) or not all(isinstance(x, int) for x in sequence):
        raise ValueError("sequence must be a list of integers")
    if not all(0 <= pin_index < pins_count for pin_index in sequence):
        raise ValueError("All elements in sequence must be valid pin indices (0 to pins_count-1)")
    if len(sequence) != lines_count + 1:
        raise ValueError("Length of sequence must be lines_count + 1")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    # Calculate pin positions on the circle
    radius = image_size / 2 - 10  # Leave some margin
    center_x = image_size / 2
    center_y = image_size / 2
    pin_positions = []
    for i in range(pins_count):
        angle = (2 * np.pi / pins_count) * i
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        pin_positions.append((x, y))

    # Create a new image
    img = Image.new('RGB', (image_size, image_size), color='black')
    draw = ImageDraw.Draw(img)

    # Draw the lines based on the sequence
    for i in range(lines_count):
        start_pin_index = sequence[i]
        end_pin_index = sequence[i + 1]
        start_pos = pin_positions[start_pin_index]
        end_pos = pin_positions[end_pin_index]
        draw.line((start_pos, end_pos), fill='white', width=1)

    return img

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Circular String Art Generator")

    # Sidebar for parameters
    pins_count = st.sidebar.number_input("Number of Pins:", min_value=2, max_value=200, value=20, step=1)
    lines_count = st.sidebar.number_input("Number of Lines:", min_value=1, max_value=1000, value=50, step=1)
    sequence_input = st.sidebar.text_area("Sequence (comma-separated pin indices):", "0, 5, 10, 15, 2, 7")
    image_size = st.sidebar.number_input("Image Size:", min_value=200, max_value=1000, value=500, step=50)

    if st.sidebar.button("Generate String Art"):
        try:
            # Parse the sequence from the text input
            sequence = [int(x.strip()) for x in sequence_input.split(',')]
            if len(sequence) != lines_count + 1:
                st.error(f"Error: The sequence must contain 'Number of Lines + 1' values. Expected {lines_count + 1}, got {len(sequence)}.")
                return

            # Generate the string art
            img = generate_string_art(pins_count, lines_count, sequence, image_size)

            # Display the image
            st.image(img, caption="Generated String Art", use_column_width=True)

            # Provide a download link for the image
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:image/png;base64,{img_str}" download="string_art.png">Download Image</a>'
            st.markdown(href, unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
