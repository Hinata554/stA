import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io

def preprocess_image(image, size=300):
    """Convert image to grayscale, resize, and enhance contrast."""
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((size, size))  # Resize to square
    img = np.array(img)
    # Simple contrast enhancement
    img = (img - img.min()) * (255 / (img.max() - img.min()))
    return Image.fromarray(img.astype(np.uint8))

def generate_nails(num_nails, size):
    """Generate coordinates for nails around a circle."""
    center = size / 2
    radius = size / 2 - 10  # Slightly inset from edge
    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)
    nails = [(center + radius * np.cos(a), center + radius * np.sin(a)) for a in angles]
    return nails

def find_best_line(img_array, nails, used_pairs, darkness_threshold):
    """Find the best line between two nails based on image darkness."""
    height, width = img_array.shape
    best_score = -1
    best_pair = None
    
    for i in range(len(nails)):
        for j in range(i + 1, len(nails)):
            if (i, j) in used_pairs or (j, i) in used_pairs:
                continue
            x1, y1 = nails[i]
            x2, y2 = nails[j]
            # Sample points along the line
            num_samples = 100
            t = np.linspace(0, 1, num_samples)
            xs = x1 + t * (x2 - x1)
            ys = y1 + t * (y2 - y1)
            score = 0
            for x, y in zip(xs, ys):
                if 0 <= x < width and 0 <= y < height:
                    score += 255 - img_array[int(y), int(x)]  # Darker pixels contribute more
            score /= num_samples
            if score > best_score and score > darkness_threshold:
                best_score = score
                best_pair = (i, j)
    
    return best_pair, best_score

def generate_string_art(image, num_nails, num_lines, line_weight=1, darkness_threshold=50):
    """Generate string art by drawing lines between nails."""
    size = 300
    img = preprocess_image(image, size)
    img_array = np.array(img)
    
    # Create white background
    result_img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(result_img)
    
    # Generate nail positions
    nails = generate_nails(num_nails, size)
    
    # Draw nails as small circles
    for x, y in nails:
        draw.ellipse([x-3, y-3, x+3, y+3], fill='black')
    
    # Generate lines
    used_pairs = set()
    for _ in range(num_lines):
        pair, score = find_best_line(img_array, nails, used_pairs, darkness_threshold)
        if pair is None:
            break
        i, j = pair
        used_pairs.add(pair)
        x1, y1 = nails[i]
        x2, y2 = nails[j]
        draw.line([(x1, y1), (x2, y2)], fill='black', width=line_weight)
        
        # Subtract line from image to avoid over-drawing
        num_samples = 100
        t = np.linspace(0, 1, num_samples)
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)
        for x, y in zip(xs, ys):
            if 0 <= x < size and 0 <= y < size:
                img_array[int(y), int(x)] = min(img_array[int(y), int(x)] + 50, 255)
    
    return result_img

# Streamlit app
st.title("String Art Generator")
st.write("Upload an image and adjust parameters to create a string art pattern.")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Parameters
    num_nails = st.slider("Number of Nails", 50, 300, 150, step=10)
    num_lines = st.slider("Number of Lines", 100, 5000, 1000, step=100)
    line_weight = st.slider("Line Weight", 1, 5, 1)
    darkness_threshold = st.slider("Darkness Threshold", 0, 100, 50)
    
    if st.button("Generate String Art"):
        with st.spinner("Generating string art..."):
            result_img = generate_string_art(image, num_nails, num_lines, line_weight, darkness_threshold)
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
