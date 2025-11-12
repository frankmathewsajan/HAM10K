import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import os

def create_sample_lesion_images():
    """Create sample lesion images for the gallery"""
    
    # Ensure static/examples directory exists
    os.makedirs("static/examples", exist_ok=True)
    
    # Sample lesion characteristics
    samples = [
        {
            "name": "melanoma_sample.jpg",
            "class": "mel",
            "description": "Irregular pigmented lesion with asymmetry",
            "colors": [(139, 69, 19), (101, 67, 33), (160, 82, 45)],
            "shape": "irregular"
        },
        {
            "name": "nevus_sample.jpg", 
            "class": "nv",
            "description": "Regular brown mole with smooth borders",
            "colors": [(101, 67, 33), (139, 69, 19), (160, 132, 112)],
            "shape": "regular"
        },
        {
            "name": "bcc_sample.jpg",
            "class": "bcc", 
            "description": "Pearly nodular lesion with visible vessels",
            "colors": [(255, 228, 196), (255, 218, 185), (205, 133, 63)],
            "shape": "nodular"
        },
        {
            "name": "akiec_sample.jpg",
            "class": "akiec",
            "description": "Rough, scaly patch with irregular surface",
            "colors": [(210, 180, 140), (188, 143, 143), (205, 133, 63)],
            "shape": "scaly"
        },
        {
            "name": "bkl_sample.jpg",
            "class": "bkl",
            "description": "Well-defined keratotic lesion",
            "colors": [(160, 82, 45), (139, 69, 19), (101, 67, 33)],
            "shape": "defined"
        },
        {
            "name": "dermatofibroma_sample.jpg",
            "class": "df",
            "description": "Firm nodular lesion with dimple sign",
            "colors": [(139, 69, 19), (160, 82, 45), (101, 67, 33)],
            "shape": "firm"
        },
        {
            "name": "vascular_sample.jpg",
            "class": "vasc",
            "description": "Red vascular lesion with clear borders",
            "colors": [(220, 20, 60), (255, 69, 0), (255, 99, 71)],
            "shape": "vascular"
        }
    ]
    
    for sample in samples:
        create_lesion_image(sample)
    
    print("✅ Sample lesion images created successfully!")

def create_lesion_image(sample):
    """Create a synthetic lesion image"""
    # Create base skin-colored background
    img_size = 224
    img = Image.new('RGB', (img_size, img_size), color=(255, 220, 177))
    draw = ImageDraw.Draw(img)
    
    # Add skin texture
    add_skin_texture(img)
    
    # Create lesion based on type
    center_x, center_y = img_size // 2, img_size // 2
    
    if sample["shape"] == "irregular":
        create_irregular_lesion(draw, center_x, center_y, sample["colors"])
    elif sample["shape"] == "regular":
        create_regular_lesion(draw, center_x, center_y, sample["colors"])
    elif sample["shape"] == "nodular":
        create_nodular_lesion(draw, center_x, center_y, sample["colors"])
    elif sample["shape"] == "scaly":
        create_scaly_lesion(draw, center_x, center_y, sample["colors"])
    elif sample["shape"] == "defined":
        create_defined_lesion(draw, center_x, center_y, sample["colors"])
    elif sample["shape"] == "firm":
        create_firm_lesion(draw, center_x, center_y, sample["colors"])
    elif sample["shape"] == "vascular":
        create_vascular_lesion(draw, center_x, center_y, sample["colors"])
    
    # Add some noise and realistic effects
    img = add_realistic_effects(img)
    
    # Save the image
    img.save(f"static/examples/{sample['name']}", quality=85)

def add_skin_texture(img):
    """Add realistic skin texture"""
    # Convert to numpy for texture manipulation
    img_array = np.array(img)
    
    # Add subtle noise for skin texture
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    textured_img = Image.fromarray(img_array)
    img.paste(textured_img)

def create_irregular_lesion(draw, center_x, center_y, colors):
    """Create an irregular melanoma-like lesion"""
    # Create irregular shape with multiple colors
    points = []
    for angle in range(0, 360, 30):
        radius = np.random.uniform(25, 45)
        x = center_x + radius * np.cos(np.radians(angle))
        y = center_y + radius * np.sin(np.radians(angle))
        points.append((x, y))
    
    # Draw main lesion
    draw.polygon(points, fill=colors[0])
    
    # Add irregular color variations
    for i in range(3):
        variation_points = []
        for angle in range(0, 360, 45):
            radius = np.random.uniform(15, 30)
            x = center_x + radius * np.cos(np.radians(angle + i * 30))
            y = center_y + radius * np.sin(np.radians(angle + i * 30))
            variation_points.append((x, y))
        draw.polygon(variation_points, fill=colors[i % len(colors)])

def create_regular_lesion(draw, center_x, center_y, colors):
    """Create a regular mole-like lesion"""
    # Main circular lesion
    radius = 30
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], fill=colors[0])
    
    # Add subtle variations
    inner_radius = 20
    draw.ellipse([center_x - inner_radius, center_y - inner_radius,
                  center_x + inner_radius, center_y + inner_radius], fill=colors[1])

def create_nodular_lesion(draw, center_x, center_y, colors):
    """Create a nodular BCC-like lesion"""
    # Main nodule
    radius = 25
    draw.ellipse([center_x - radius, center_y - radius,
                  center_x + radius, center_y + radius], fill=colors[0])
    
    # Add pearly appearance
    highlight_radius = 15
    draw.ellipse([center_x - highlight_radius, center_y - highlight_radius,
                  center_x + highlight_radius, center_y + highlight_radius], fill=colors[1])

def create_scaly_lesion(draw, center_x, center_y, colors):
    """Create a scaly keratosis-like lesion"""
    # Base lesion
    radius = 35
    draw.ellipse([center_x - radius, center_y - radius,
                  center_x + radius, center_y + radius], fill=colors[0])
    
    # Add scaly texture with small rectangles
    for i in range(20):
        x = center_x + np.random.uniform(-radius, radius)
        y = center_y + np.random.uniform(-radius, radius)
        size = np.random.uniform(2, 5)
        draw.rectangle([x, y, x + size, y + size], fill=colors[1])

def create_defined_lesion(draw, center_x, center_y, colors):
    """Create a well-defined keratotic lesion"""
    # Well-defined circular lesion
    radius = 28
    draw.ellipse([center_x - radius, center_y - radius,
                  center_x + radius, center_y + radius], fill=colors[0])
    
    # Clear border
    border_width = 2
    draw.ellipse([center_x - radius - border_width, center_y - radius - border_width,
                  center_x + radius + border_width, center_y + radius + border_width], 
                 outline=colors[1], width=border_width)

def create_firm_lesion(draw, center_x, center_y, colors):
    """Create a firm dermatofibroma-like lesion"""
    # Firm, dome-shaped lesion
    radius = 20
    draw.ellipse([center_x - radius, center_y - radius,
                  center_x + radius, center_y + radius], fill=colors[0])
    
    # Add central depression
    inner_radius = 8
    draw.ellipse([center_x - inner_radius, center_y - inner_radius,
                  center_x + inner_radius, center_y + inner_radius], fill=colors[1])

def create_vascular_lesion(draw, center_x, center_y, colors):
    """Create a vascular lesion"""
    # Main vascular area
    radius = 30
    draw.ellipse([center_x - radius, center_y - radius,
                  center_x + radius, center_y + radius], fill=colors[0])
    
    # Add vascular pattern (simplified)
    for i in range(5):
        start_x = center_x + np.random.uniform(-radius//2, radius//2)
        start_y = center_y + np.random.uniform(-radius//2, radius//2)
        end_x = start_x + np.random.uniform(-10, 10)
        end_y = start_y + np.random.uniform(-10, 10)
        draw.line([start_x, start_y, end_x, end_y], fill=colors[1], width=2)

def add_realistic_effects(img):
    """Add realistic photographic effects"""
    # Slight blur to simulate camera
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Adjust brightness and contrast randomly
    from PIL import ImageEnhance
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.uniform(0.9, 1.1))
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(0.9, 1.1))
    
    return img

if __name__ == "__main__":
    create_sample_lesion_images()