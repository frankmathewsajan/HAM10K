import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from keras.layers import TFSMLayer
from PIL import Image
import uvicorn

# ===== CONFIG =====
MODEL_PATH = "model_saved"
IMG_SIZE = 224
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses (Pre-cancerous lesions)",
    "bcc": "Basal Cell Carcinoma (Most common skin cancer)",
    "bkl": "Benign Keratosis (Non-cancerous growth)",
    "df": "Dermatofibroma (Benign fibrous tissue)",
    "mel": "Melanoma (Dangerous skin cancer)",
    "nv": "Melanocytic Nevi (Moles)",
    "vasc": "Vascular Lesions (Blood vessel related)"
}

# ===== FASTAPI SETUP =====
app = FastAPI(title="HAM10K Skin Lesion Classifier", description="AI-powered skin lesion classification")
templates = Jinja2Templates(directory="templates")

# ===== LOAD MODEL =====
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.Sequential([TFSMLayer(MODEL_PATH, call_endpoint="serving_default")])
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class SkinLesionClassifier:
    def __init__(self):
        self.model = model
        
    def preprocess_image(self, image_bytes):
        """Preprocess uploaded image for model prediction"""
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # Convert PIL to numpy array
        img_array = np.array(pil_image)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        # Normalize to 0-1 range
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
    
    def predict(self, image_bytes):
        """Make prediction on uploaded image"""
        if self.model is None:
            raise Exception("Model not loaded")
            
        # Preprocess image
        img_batch, img_display = self.preprocess_image(image_bytes)
        
        # Make prediction
        pred_probs = np.array(list(self.model.predict(img_batch).values())[0])[0]
        pred_index = np.argmax(pred_probs)
        
        return {
            'predicted_class': CLASSES[pred_index],
            'predicted_class_description': CLASS_DESCRIPTIONS[CLASSES[pred_index]],
            'confidence': float(pred_probs[pred_index]),
            'probabilities': {class_name: float(prob) for class_name, prob in zip(CLASSES, pred_probs)},
            'processed_image': img_display
        }
    
    def generate_chart(self, probabilities, predicted_class):
        """Generate prediction probabilities bar chart"""
        plt.figure(figsize=(12, 8))
        
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Define colors for medical categories
        color_map = {
            'mel': '#ef4444',    # Red for melanoma
            'bcc': '#f97316',    # Orange for basal cell carcinoma
            'akiec': '#f59e0b',  # Yellow for actinic keratoses
            'nv': '#10b981',     # Green for nevi (benign)
            'bkl': '#06b6d4',    # Cyan for benign keratosis
            'df': '#8b5cf6',     # Purple for dermatofibroma
            'vasc': '#3b82f6'    # Blue for vascular
        }
        
        colors = [color_map.get(cls.lower(), '#64748b') for cls in classes]
        
        # Create bar chart
        bars = plt.bar(classes, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Highlight predicted class
        pred_index = classes.index(predicted_class)
        bars[pred_index].set_alpha(1.0)
        bars[pred_index].set_edgecolor('#ffffff')
        bars[pred_index].set_linewidth(3)
        
        # Customize chart for medical aesthetics
        plt.style.use('dark_background')
        plt.figure(facecolor='#1e293b')
        plt.gca().set_facecolor('#1e293b')
        
        plt.title('HAM10K Classification Analysis', fontsize=18, fontweight='bold', color='white', pad=20)
        plt.ylabel('Probability', fontsize=14, color='white')
        plt.xlabel('Lesion Classifications', fontsize=14, color='white')
        plt.ylim(0, 1)
        
        # Style the axes
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().tick_params(colors='white')
        
        # Add percentage labels on bars
        for i, prob in enumerate(probs):
            plt.text(i, prob + 0.02, f'{prob*100:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', color='white', fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, color='white')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#1e293b')
        buffer.seek(0)
        chart_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_b64
    
    def generate_confidence_analysis(self, probabilities):
        """Generate detailed confidence analysis"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        primary_class, primary_prob = sorted_probs[0]
        secondary_class, secondary_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
        
        # Calculate confidence metrics
        confidence_gap = primary_prob - secondary_prob if secondary_prob else primary_prob
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities.values())
        
        # Risk assessment based on class
        malignant_classes = ['mel', 'bcc']
        premalignant_classes = ['akiec']
        
        malignant_prob = sum(probabilities.get(cls, 0) for cls in malignant_classes)
        premalignant_prob = sum(probabilities.get(cls, 0) for cls in premalignant_classes)
        benign_prob = 1 - malignant_prob - premalignant_prob
        
        return {
            'primary_confidence': primary_prob,
            'secondary_confidence': secondary_prob,
            'confidence_gap': confidence_gap,
            'entropy': entropy,
            'malignant_probability': malignant_prob,
            'premalignant_probability': premalignant_prob,
            'benign_probability': benign_prob,
            'certainty_level': self._get_certainty_level(primary_prob),
            'risk_level': self._get_risk_level(malignant_prob, premalignant_prob, primary_class)
        }
    
    def _get_certainty_level(self, confidence):
        """Determine certainty level based on confidence"""
        if confidence > 0.9:
            return 'VERY_HIGH'
        elif confidence > 0.8:
            return 'HIGH'
        elif confidence > 0.7:
            return 'MODERATE'
        elif confidence > 0.6:
            return 'FAIR'
        else:
            return 'LOW'
    
    def _get_risk_level(self, malignant_prob, premalignant_prob, predicted_class):
        """Assess risk level based on probabilities"""
        if malignant_prob > 0.7:
            return 'HIGH'
        elif malignant_prob > 0.3 or premalignant_prob > 0.7:
            return 'MODERATE'
        elif predicted_class.lower() in ['mel', 'bcc'] and malignant_prob > 0.1:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def generate_recommendations(self, predicted_class, confidence_analysis):
        """Generate clinical recommendations"""
        recommendations = []
        
        predicted_lower = predicted_class.lower()
        confidence = confidence_analysis['primary_confidence']
        risk_level = confidence_analysis['risk_level']
        
        # Class-specific recommendations
        if predicted_lower == 'mel':
            recommendations.extend([
                "Urgent dermatological referral within 2 weeks",
                "Consider dermoscopy and possible biopsy",
                "Patient education on melanoma warning signs",
                "Sun protection counseling"
            ])
        elif predicted_lower == 'bcc':
            recommendations.extend([
                "Dermatological consultation within 4-6 weeks",
                "Consider surgical excision or Mohs surgery",
                "Sun protection and regular skin checks"
            ])
        elif predicted_lower == 'akiec':
            recommendations.extend([
                "Dermatological follow-up in 2-4 weeks",
                "Monitor lesion for changes",
                "Consider cryotherapy or topical treatment",
                "Annual skin cancer screening"
            ])
        else:  # Benign lesions
            recommendations.extend([
                "Routine dermatological monitoring",
                "Annual comprehensive skin examination",
                "Self-examination education",
                "Sun protection measures"
            ])
        
        # Confidence-based recommendations
        if confidence < 0.7:
            recommendations.extend([
                "Consider additional imaging (dermoscopy)",
                "Clinical correlation strongly recommended",
                "Second opinion may be beneficial"
            ])
        
        # Risk-based recommendations
        if risk_level == 'HIGH':
            recommendations.insert(0, "HIGH PRIORITY: Expedited medical evaluation")
        elif risk_level == 'MODERATE':
            recommendations.insert(0, "MODERATE PRIORITY: Timely medical consultation")
        
        return recommendations
    
    def image_to_base64(self, img_array):
        """Convert numpy array to base64 string for web display"""
        # Convert to PIL Image
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        return img_b64

# Initialize classifier
classifier = SkinLesionClassifier()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and prediction with comprehensive analysis"""
    try:
        # Check if file is an image
        if not file.content_type.startswith('image/'):
            return {"error": "Please upload a valid image file"}
        
        # Read file
        image_bytes = await file.read()
        
        # Make prediction
        result = classifier.predict(image_bytes)
        
        # Generate comprehensive analysis
        confidence_analysis = classifier.generate_confidence_analysis(result['probabilities'])
        recommendations = classifier.generate_recommendations(result['predicted_class'], confidence_analysis)
        
        # Generate visualizations
        chart_b64 = classifier.generate_chart(result['probabilities'], result['predicted_class'])
        image_b64 = classifier.image_to_base64(result['processed_image'])
        
        # Prepare detailed response
        response = {
            "success": True,
            "prediction": result['predicted_class'],
            "description": result['predicted_class_description'],
            "confidence": f"{result['confidence']*100:.2f}%",
            "probabilities": result['probabilities'],
            "chart": chart_b64,
            "image": image_b64,
            "filename": file.filename,
            
            # Enhanced analytics
            "confidence_analysis": {
                "primary_confidence": f"{confidence_analysis['primary_confidence']*100:.2f}%",
                "secondary_confidence": f"{confidence_analysis['secondary_confidence']*100:.2f}%",
                "confidence_gap": f"{confidence_analysis['confidence_gap']*100:.2f}%",
                "certainty_level": confidence_analysis['certainty_level'],
                "entropy": f"{confidence_analysis['entropy']:.3f}"
            },
            
            "risk_assessment": {
                "level": confidence_analysis['risk_level'],
                "malignant_probability": f"{confidence_analysis['malignant_probability']*100:.2f}%",
                "premalignant_probability": f"{confidence_analysis['premalignant_probability']*100:.2f}%",
                "benign_probability": f"{confidence_analysis['benign_probability']*100:.2f}%"
            },
            
            "recommendations": recommendations,
            
            "metadata": {
                "file_size": file.size,
                "content_type": file.content_type,
                "model_version": "HAM10K-v2.1",
                "processing_timestamp": "2025-09-26T00:30:00Z"
            }
        }
        
        return response
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": CLASSES
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)