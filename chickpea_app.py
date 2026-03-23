"""
Improved Automated Chickpea Quality Assessment and Counting
- Upgraded preprocessing (CLAHE + combined edges + color)
- Ensemble counting (ML + contour) with fallback logic
- Improved contour preprocessing (adaptive threshold + morphology)
- Refined per-seed feature extraction (lighting normalization, adjusted thresholds)
- Compatible with existing image_processing and chickpea_detr modules
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests

# Local imports
from image_processing import (
	preprocess_image, 
	preprocess_pil_image, 
	count_chickpeas, 
	count_chickpeas_advanced,
	visualize_preprocessing
)

# DETR integration
try:
	from chickpea_detr import ChickpeaDETR
	DETR_AVAILABLE = True
except ImportError as e:
	st.warning(f"DETR integration not available: {e}")
	DETR_AVAILABLE = False

# Roboflow integration
try:
	from chickpea_roboflow import ChickpeaRoboflowDetector, create_roboflow_detector
	ROBOFLOW_AVAILABLE = True
except ImportError as e:
	st.warning(f"Roboflow integration not available: {e}")
	ROBOFLOW_AVAILABLE = False

# RF-DETR integration
try:
	from chickpea_rf_detr import ChickpeaRFDETRDetector, create_rf_detr_detector, AVAILABLE_MODELS
	RF_DETR_AVAILABLE = True
except ImportError as e:
	st.warning(f"RF-DETR integration not available: {e}")
	RF_DETR_AVAILABLE = False

# DETR prediction processing
def process_detr_predictions(predictions_data):
	"""Process DETR predictions and convert to our format"""
	if not predictions_data or 'predictions' not in predictions_data:
		return None
	
	detections = []
	for i, pred in enumerate(predictions_data['predictions']):
		# Convert DETR format to our detection format
		detection = {
			'id': i + 1,
			'bbox': [pred['x'], pred['y'], pred['width'], pred['height']],
			'confidence': pred['confidence'],
			'class': pred['class'],
			'class_id': pred['class_id'],
			'detection_id': pred['detection_id'],
			'area': pred['width'] * pred['height']
		}
		detections.append(detection)
	
	return {
		'count': len(detections),
		'detections': detections,
		'method': 'DETR',
		'annotated_image': None  # Will be created if needed
	}

def create_detr_annotated_image(image, detections):
	"""Create annotated image from DETR detections"""
	import cv2
	import numpy as np
	
	# Convert PIL to OpenCV
	img_array = np.array(image.convert('RGB'))
	annotated = img_array.copy()
	
	for detection in detections:
		x, y, w, h = detection['bbox']
		confidence = detection['confidence']
		
		# Draw bounding box
		cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
		
		# Draw confidence score
		label = f"{detection['class']}: {confidence:.2f}"
		cv2.putText(annotated, label, (int(x), int(y - 10)), 
				   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	
	return annotated

# New imports for ML counting
from transformers import AutoImageProcessor
from safetensors.torch import load_file as load_safetensors
import torch.nn as nn

# Configure Streamlit page
st.set_page_config(
	page_title="🌱 Automated Chickpea Quality Assessment",
	page_icon="🌱",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Visibility flags for UI sections
SHOW_MODEL_INFO = False
SHOW_QUALITY_CRITERIA = False
SHOW_MODEL_PERFORMANCE = False

@st.cache_resource
def load_model(model_path: str = "./trained_model"):
	"""Load the trained Hugging Face model"""
	try:
		if not Path(model_path).exists():
			return None, None, None
		
		from transformers import AutoImageProcessor, AutoModelForImageClassification
		
		processor = AutoImageProcessor.from_pretrained(model_path)
		model = AutoModelForImageClassification.from_pretrained(model_path)
		
		# Load class names
		class_names_path = Path(model_path) / "class_names.json"
		if class_names_path.exists():
			with open(class_names_path, "r") as f:
				class_names = json.load(f)
		else:
			class_names = ["healthy", "broken", "discolored", "small", "large"]
		
		return processor, model, class_names
	except Exception as e:
		st.error(f"Error loading model: {str(e)}")
		return None, None, None

# ===== ML Counting Model Integration =====
class CountRegressor(nn.Module):
	def __init__(self, backbone_name: str = "google/vit-base-patch16-224"):
		super().__init__()
		from transformers import ViTModel
		self.backbone = ViTModel.from_pretrained(backbone_name)
		hidden = self.backbone.config.hidden_size
		self.regressor = nn.Sequential(
			nn.LayerNorm(hidden),
			nn.Linear(hidden, 256),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(256, 1)
		)
	def forward(self, pixel_values):
		out = self.backbone(pixel_values=pixel_values)
		pooled = out.pooler_output
		return self.regressor(pooled)

# ----------------- Enhanced Preprocessing Utilities -----------------

def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the V channel of HSV to improve local contrast."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge((h, s, v2))
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def enhanced_preprocess_pil(image: Image.Image, canny_low=80, canny_high=180) -> Image.Image:
    """Combine CLAHE + edges + weighted color to create a robust input for classifiers.
    Returns a PIL image in RGB.
    """
    arr = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE to reduce lighting variance
    bgr_clahe = apply_clahe(bgr)
    
    gray = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Blend edges with color image (preserve color but emphasize edges)
    combined = cv2.addWeighted(bgr_clahe, 0.85, edges_col, 0.15, 0)
    
    # Resize to a reasonable size if too large - keep aspect
    h, w = combined.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # Convert back to RGB PIL
    rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_edges_for_processor(pil_image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    # Keep existing logic but add CLAHE to the base image
    arr = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    bgr = apply_clahe(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges3)


@st.cache_resource
def load_count_model(count_model_dir: str = "./count_model"):
	try:
		model_path = Path(count_model_dir) / "model.safetensors"
		prep_path = Path(count_model_dir)
		if not model_path.exists():
			return None, None
		processor = AutoImageProcessor.from_pretrained(str(prep_path))
		model = CountRegressor()
		state = load_safetensors(str(model_path))
		model.load_state_dict(state)
		model.eval()
		return processor, model
	except Exception as e:
		st.warning(f"Counting model not loaded: {e}")
		return None, None


def predict_count_ml(pil_image: Image.Image, count_processor, count_model) -> float:
	"""Predict chickpea count using ML model with enhanced preprocessing."""
	if count_processor is None or count_model is None:
		return None
	# Use enhanced edges variant as ML input
	img_edges = preprocess_edges_for_processor(pil_image)
	inputs = count_processor(images=img_edges, return_tensors="pt")
	with torch.no_grad():
		pred = count_model(pixel_values=inputs["pixel_values"])  # log-count
		log_count = float(pred.squeeze().item())
		# numerically stable inverse of log1p
		count = float(np.expm1(log_count))
		return max(0.0, count)


def predict_quality(image: Image.Image, processor, model, class_names, canny_low=80, canny_high=180):
	"""Predict chickpea quality using the trained model with enhanced preprocessing"""
	try:
		processed_pil = enhanced_preprocess_pil(image, canny_low, canny_high)
		inputs = processor(images=processed_pil, return_tensors="pt")
		with torch.no_grad():
			outputs = model(**inputs)
			predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
			predicted_class_id = predictions.argmax().item()
			confidence = predictions[0][predicted_class_id].item()

		all_predictions = {class_names[i]: predictions[0][i].item() for i in range(len(class_names))}
		return {
			'predicted_class': class_names[predicted_class_id],
			'confidence': confidence,
			'all_predictions': all_predictions
		}
	except Exception as e:
		st.warning(f"Error during prediction: {str(e)}")
		return None


def display_quality_metrics(predictions, count_result):
	"""Display quality assessment metrics"""
	if not predictions:
		return
	
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		st.metric(
			"Predicted Quality", 
			predictions['predicted_class'].title(),
			f"{predictions['confidence']:.1%} confidence"
		)
	
	with col2:
		st.metric("Total Chickpeas", count_result['count'])
	
	with col3:
		# Handle both DETR and contour results
		if 'analysis' in count_result and count_result['analysis'].get('avg_area', 0) > 0:
			st.metric("Avg Size", f"{count_result['analysis']['avg_area']:.0f} px²")
		elif 'detections' in count_result and count_result['detections']:
			# Calculate average area from DETR detections
			areas = [d.get('area', 0) for d in count_result['detections'] if d.get('area', 0) > 0]
			if areas:
				avg_area = sum(areas) / len(areas)
				st.metric("Avg Size", f"{avg_area:.0f} px²")
			else:
				st.metric("Avg Size", "N/A")
		else:
			st.metric("Avg Size", "N/A")
	
	with col4:
		# Handle both DETR and contour results
		if 'analysis' in count_result and count_result['analysis'].get('avg_circularity', 0) > 0:
			circularity_grade = "Good" if count_result['analysis']['avg_circularity'] > 0.7 else "Fair"
			st.metric("Shape Quality", circularity_grade)
		else:
			# For DETR results, we don't have circularity data, so show method
			method = count_result.get('method', 'Unknown')
			st.metric("Detection Method", method)


def create_quality_charts(predictions, count_result):
	"""Create visualization charts for quality assessment"""
	if not predictions or not count_result:
		return None, None
	
	# Quality prediction chart
	pred_data = predictions['all_predictions']
	fig1 = px.bar(
		x=list(pred_data.keys()),
		y=list(pred_data.values()),
		title="Quality Classification Probabilities",
		labels={'x': 'Quality Class', 'y': 'Probability'},
		color=list(pred_data.values()),
		color_continuous_scale='RdYlGn'
	)
	fig1.update_layout(
		xaxis_title="Quality Class",
		yaxis_title="Probability",
		showlegend=False,
		height=400
	)
	
	# Size distribution chart
	areas = []
	if 'analysis' in count_result and count_result['analysis'].get('areas'):
		areas = count_result['analysis']['areas']
	elif 'detections' in count_result and count_result['detections']:
		# Extract areas from DETR detections
		areas = [d.get('area', 0) for d in count_result['detections'] if d.get('area', 0) > 0]
	
	if areas:
		fig2 = px.histogram(
			x=areas,
			nbins=10,
			title="Chickpea Size Distribution",
			labels={'x': 'Area (pixels²)', 'y': 'Count'},
			color_discrete_sequence=['#2E8B57']
		)
		fig2.update_layout(
			xaxis_title="Area (pixels²)",
			yaxis_title="Number of Chickpeas",
			height=400
		)
	else:
		fig2 = None
	
	return fig1, fig2


# ========= New: Per-seed feature extraction and grading =========
def _color_name_from_hsv(mean_hsv):
	# mean_hsv from OpenCV (H:0-180, S:0-255, V:0-255)
	h, s, v = mean_hsv
	if v < 40:
		return "black"
	if s < 40:
		return "white" if v > 200 else "gray"
	if h < 10 or h >= 170:
		return "red"
	if h < 25:
		return "orange"
	if h < 35:
		return "yellow"
	if h < 85:
		return "green"
	if h < 130:
		return "blue"
	if h < 170:
		return "purple"
	return "unknown"

def _features_from_contour(cnt, crop_bgr):
	# Normalize crop lighting using CLAHE on V channel for consistent means
	if crop_bgr is None or crop_bgr.size == 0:
		return None
	crop_bgr = crop_bgr.copy()
	crop_bgr = apply_clahe(crop_bgr)
	
	area_px = float(cv2.contourArea(cnt))
	perimeter_px = float(cv2.arcLength(cnt, True))
	circularity = float((4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-6)) if perimeter_px > 0 else 0.0
	equivalent_diameter_px = float(np.sqrt(4 * area_px / np.pi)) if area_px > 0 else 0.0
	rect = cv2.minAreaRect(cnt)
	rw, rh = rect[1]
	length_px = float(max(rw, rh)) if rw > 0 and rh > 0 else 0.0
	width_px = float(min(rw, rh)) if rw > 0 and rh > 0 else 0.0
	aspect_ratio = float((length_px / (width_px + 1e-6))) if length_px > 0 and width_px > 0 else 1.0
	hull = cv2.convexHull(cnt)
	hull_area = float(cv2.contourArea(hull)) if hull is not None and cv2.contourArea(hull) > 0 else max(1e-6, area_px)
	solidity = float(area_px / (hull_area + 1e-6))

	mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
	# shift cnt expected to be in crop coordinates already
	cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
	mean_bgr = cv2.mean(crop_bgr, mask=mask)[:3]
	hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

	# Ensure mask non-empty for mean calculation
	if mask.sum() == 0:
		mean_hsv = (0.0, 0.0, 0.0)
	else:
		mean_hsv = cv2.mean(hsv_crop, mask=mask)[:3]

	color_name = _color_name_from_hsv(mean_hsv)

	# Texture roughness via Laplacian variance inside the mask
	gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
	lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
	lap_var = float(np.var(lap[mask == 255])) if mask.sum() > 0 else 0.0
	# Normalize to a reasonable 0-1 range (empirical scaling)
	texture_roughness = float(min(1.0, lap_var / 5000.0))

	# Discoloration spots: fraction of very dark pixels in V channel
	v_channel = hsv_crop[:, :, 2]
	dark_spot_fraction = float((v_channel[mask == 255] < 60).mean()) if mask.sum() > 0 else 0.0

	# Broken score heuristic: tuned weights and thresholds
	circularity_norm = min(1.0, max(0.0, circularity))
	roughness = max(0.0, (perimeter_px / (2 * np.sqrt(np.pi * max(area_px, 1e-6)))) - 1.0)
	# lowered threshold sensitivity slightly
	broken_score = float(0.45 * (1.0 - solidity) + 0.45 * (1.0 - circularity_norm) + 0.1 * min(1.0, roughness))
	# Combine with discoloration and texture for a broader defect score
	defect_score = float(0.6 * broken_score + 0.25 * dark_spot_fraction + 0.15 * texture_roughness)
	is_broken = bool(broken_score > 0.30 or (solidity < 0.87 and circularity < 0.62))
	has_defect = bool(defect_score > 0.35)
	return {
		'area_px': area_px,
		'equivalent_diameter_px': equivalent_diameter_px,
		'length_px': length_px,
		'width_px': width_px,
		'perimeter_px': perimeter_px,
		'circularity': circularity,
		'aspect_ratio': aspect_ratio,
		'solidity': solidity,
		'mean_bgr': tuple(float(x) for x in mean_bgr),
		'mean_hsv': tuple(float(x) for x in mean_hsv),
		'color_name': color_name,
		'broken_score': broken_score,
		'is_broken': is_broken,
		'texture_roughness': texture_roughness,
		'dark_spot_fraction': dark_spot_fraction,
		'defect_score': defect_score,
		'has_defect': has_defect,
	}

def extract_seed_features_from_contours(original_pil: Image.Image, contours):
	if not contours:
		return []
	img_bgr = cv2.cvtColor(np.array(original_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
	features = []
	idx = 0
	for cnt in contours:
		if cv2.contourArea(cnt) <= 1:
			continue
		x, y, w, h = cv2.boundingRect(cnt)
		if w <= 0 or h <= 0:
			continue
		crop = img_bgr[y:y+h, x:x+w]
		cnt_shifted = cnt - np.array([[x, y]])
		feat = _features_from_contour(cnt_shifted, crop)
		if feat is None:
			continue
		idx += 1
		feat['id'] = idx
		features.append(feat)
	return features

def derive_quality_grades(seed_features):
	if not seed_features:
		return None
	diams = np.array([f['equivalent_diameter_px'] for f in seed_features], dtype=np.float32)
	if diams.size >= 2:
		d25, d75 = np.percentile(diams, [25, 75])
	else:
		d25, d75 = float(np.min(diams)), float(np.max(diams))
	size_grades = []
	for d in diams:
		if d < d25:
			size_grades.append('Small')
		elif d > d75:
			size_grades.append('Large')
		else:
			size_grades.append('Medium')
	circ_grades = ['Good' if f['circularity'] >= 0.75 else ('Fair' if f['circularity'] >= 0.6 else 'Poor') for f in seed_features]
	color_grades = []
	for f in seed_features:
		cn = f['color_name']
		if cn in ("yellow", "orange"):
			color_grades.append('Good')
		elif cn in ("green", "brown", "purple", "red"):
			color_grades.append('Fair')
		else:
			color_grades.append('Poor')
	defect_grades = ['Good' if f['broken_score'] < 0.25 else ('Fair' if f['broken_score'] < 0.45 else 'Poor') for f in seed_features]
	return {
		'size_grades': size_grades,
		'circularity_grades': circ_grades,
		'color_grades': color_grades,
		'defect_grades': defect_grades,
	}


# ========= Lot-level quality metrics and visualization =========
def compute_lot_metrics(seed_features):
	if not seed_features:
		return None
	areas = np.array([f['area_px'] for f in seed_features], dtype=np.float32)
	diams = np.array([f['equivalent_diameter_px'] for f in seed_features], dtype=np.float32)
	circularities = np.array([f['circularity'] for f in seed_features], dtype=np.float32)
	aspects = np.array([f['aspect_ratio'] for f in seed_features], dtype=np.float32)
	textures = np.array([f.get('texture_roughness', 0.0) for f in seed_features], dtype=np.float32)
	dark_spots = np.array([f.get('dark_spot_fraction', 0.0) for f in seed_features], dtype=np.float32)
	defects = np.array([1.0 if f.get('has_defect', False) else 0.0 for f in seed_features], dtype=np.float32)

	def safe_cv(x: np.ndarray) -> float:
		m = float(np.mean(x))
		if m <= 1e-6:
			return 0.0
		return float(np.std(x) / (m + 1e-6))

	metrics = {
		'count': int(len(seed_features)),
		'area_mean': float(np.mean(areas)) if areas.size else 0.0,
		'diameter_mean': float(np.mean(diams)) if diams.size else 0.0,
		'size_cv': safe_cv(diams),
		'circularity_mean': float(np.mean(circularities)) if circularities.size else 0.0,
		'aspect_mean': float(np.mean(aspects)) if aspects.size else 0.0,
		'aspect_low_ratio': float(np.mean(aspects < 1.3)) if aspects.size else 0.0,
		'texture_mean': float(np.mean(textures)) if textures.size else 0.0,
		'dark_spot_mean': float(np.mean(dark_spots)) if dark_spots.size else 0.0,
		'defect_rate': float(np.mean(defects)) if defects.size else 0.0,
	}

	# Aggregate quality indices on 0-1 scale (higher is better)
	# Size uniformity: lower CV is better
	size_uniformity = float(1.0 - min(1.0, metrics['size_cv']))
	# Shape quality combines circularity and low aspect ratio fraction
	shape_quality = float(np.clip(0.7 * metrics['circularity_mean'] + 0.3 * metrics['aspect_low_ratio'], 0.0, 1.0))
	# Color/spot quality: fewer dark spots is better
	color_quality = float(1.0 - np.clip(metrics['dark_spot_mean'], 0.0, 1.0))
	# Texture smoothness: lower roughness is better
	texture_quality = float(1.0 - np.clip(metrics['texture_mean'], 0.0, 1.0))
	# Purity: inverse of defect rate
	purity = float(1.0 - np.clip(metrics['defect_rate'], 0.0, 1.0))

	metrics.update({
		'size_uniformity': size_uniformity,
		'shape_quality': shape_quality,
		'color_quality': color_quality,
		'texture_quality': texture_quality,
		'purity': purity,
	})
	return metrics


def create_quality_radar(metrics):
	if not metrics:
		return None
	labels = ["Size Uniformity", "Shape", "Color", "Texture", "Purity"]
	values = [
		metrics.get('size_uniformity', 0.0),
		metrics.get('shape_quality', 0.0),
		metrics.get('color_quality', 0.0),
		metrics.get('texture_quality', 0.0),
		metrics.get('purity', 0.0),
	]
	values.append(values[0])
	labels_closed = labels + [labels[0]]
	fig = go.Figure()
	fig.add_trace(go.Scatterpolar(r=values, theta=labels_closed, fill='toself', name='Quality'))
	fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=420)
	return fig


def annotate_seed_overlay(original_pil: Image.Image, contours, seed_features):
	if not contours or not seed_features:
		return None
	img = np.array(original_pil.convert('RGB')).copy()
	for cnt, feat in zip(contours, seed_features):
		color = (220, 20, 60) if feat.get('has_defect', False) else (34, 139, 34)
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
		label = f"D{feat.get('equivalent_diameter_px', 0):.0f} C{feat.get('circularity', 0):.2f}"
		cv2.putText(img, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
	return img


def main():
	st.title("🌱 Automated Chickpea Quality Assessment ")
	#st.markdown("""
	#This application uses **Hugging Face transformers** with **Canny edge detection** preprocessing 
	#to classify chickpea quality and count the number of chickpeas in an image with high accuracy.
	#""")
	
	# Sidebar
	with st.sidebar:
		# Check if model exists
		model_path = "./trained_model"
		model_exists = Path(model_path).exists()
		if model_exists:
			processor, model, class_names = load_model(model_path)
			if not (processor and model and class_names):
				model_exists = False
		
		# Detection method fixed to contour; advanced detectors removed from UI
		
		st.header("🔢 Counting Method")
		use_ml_count = st.toggle("Use ML Counting (ViT Regression)", value=True)
		
		if SHOW_QUALITY_CRITERIA:
			st.header("📊 Quality Assessment Criteria")
			st.markdown("""
			- **Healthy**: Intact, good color, proper size
			- **Broken**: Cracked, split, or damaged  
			- **Discolored**: Off-color, stained, or blemished
			- **Small**: Below average size
			- **Large**: Above average size
			""")
		
		st.header("🎯 Counting Parameters")
		canny_low = st.slider("Canny Low Threshold", 30, 200, 80)
		canny_high = st.slider("Canny High Threshold", 80, 300, 180)
		min_area = st.slider("Min Area", 0, 500, 100)
		max_area = st.slider("Max Area", 1000, 20000, 10000)

		# Image source: URL loader (useful for very dense scenes)
		st.header("🖼 Image Source")
		image_url_input = st.text_input(
			"Image URL (optional)",
			value="",
			help="Paste a direct image link (JPG/PNG) with many chickpeas"
		)
		if st.button("Load URL Image"):
			st.session_state["image_url"] = image_url_input.strip()
	
	# Load counting model
	count_processor, count_model = load_count_model("./count_model") if use_ml_count else (None, None)
	
	# Main content
	uploaded_file = st.file_uploader(
		"Upload a chickpea image",
		type=["jpg", "jpeg", "png"],
		help="Upload a clear image of chickpea seeds for quality assessment and counting"
	)
	
	# DETR predictions input removed; using contour-based detection only
	
	# Resolve image source: URL (if provided) takes precedence over file upload
	image = None
	if "image_url" in st.session_state and st.session_state.get("image_url"):
		try:
			resp = requests.get(st.session_state["image_url"], timeout=10)
			resp.raise_for_status()
			image = Image.open(io.BytesIO(resp.content))
			st.success("✅ Loaded image from URL")
		except Exception as e:
			st.error(f"❌ Failed to load image from URL: {e}")
			# Clear invalid URL to avoid repeated errors
			st.session_state["image_url"] = ""
	elif uploaded_file is not None:
		# Load and display original image from file upload
		image = Image.open(uploaded_file)

	if image is not None:
		
		col1, col2 = st.columns([1, 1])
		
		with col1:
			st.subheader("📸 Original Image")
			st.image(image, use_column_width=True, caption="Uploaded Image")
		
		
		# Processing section
		st.subheader("🔄 Processing Results")
		
		with st.spinner("Analyzing image..."):
			cnt_result = None
			# Quality prediction (if model is available)
			predictions = None
			if model_exists and processor and model and class_names:
				predictions = predict_quality(image, processor, model, class_names, canny_low, canny_high)
			
			# Chickpea counting - both methods
			ml_count = None
			if count_processor and count_model:
				try:
					ml_count = predict_count_ml(image, count_processor, count_model)
				except Exception as e:
					st.warning(f"ML counting failed: {e}")
			
			try:
				# Save uploaded file temporarily for contour counting
				temp_path = "temp_upload.jpg"
				image.save(temp_path)
				
				# Count chickpeas with advanced analysis
				cnt_result = count_chickpeas_advanced(
					temp_path, 
					min_area=min_area, 
					max_area=max_area
				)
				
				# Clean up temp file
				if os.path.exists(temp_path):
					os.remove(temp_path)
					
			except Exception as e:
				st.error(f"Error in contour counting: {str(e)}")
				cnt_result = None

			# Ensemble logic for final count
			final_count = None
			if ml_count is not None and cnt_result is not None and 'count' in cnt_result:
				# Weighted ensemble (favor the contour when counts are small, ML for dense scenes)
				contour_count = float(cnt_result['count'])
				# Heuristic: if contour_count > 100, rely more on ML
				weight_ml = 0.6 if contour_count > 100 else 0.35
				final_count = weight_ml * ml_count + (1 - weight_ml) * contour_count
				final_count = float(round(final_count, 2))
				cnt_result['ensemble_count'] = final_count
			elif ml_count is not None:
				final_count = float(round(ml_count, 2))
				if cnt_result is None:
					cnt_result = {'count': None, 'method': 'ML-only'}
				cnt_result['ensemble_count'] = final_count
			elif cnt_result is not None:
				final_count = float(cnt_result.get('count', 0))
				cnt_result['ensemble_count'] = final_count
		
		# Display results
		if predictions or cnt_result or ml_count is not None:
			st.subheader("📊 Assessment Results")
			
			# Show ML vs ensemble counts
			count_cols = st.columns(3)
			if ml_count is not None:
				with count_cols[0]:
					st.metric("ML Estimated Count", f"{ml_count:.1f}")
			if cnt_result and cnt_result['count'] is not None:
				with count_cols[1]:
					st.metric("Contour Count", cnt_result['count'])
			if final_count is not None:
				with count_cols[2]:
					st.metric("Ensemble Count", final_count)
			
			# Quality metrics
			if predictions and cnt_result:
				display_quality_metrics(predictions, cnt_result)
			
			# Display annotated image
			if cnt_result:
				detection_method = cnt_result.get('method', 'Contour')
				# Map internal method names to display names
				method_display = {
					'DETR': 'DETR Detection',
					'Roboflow': 'Roboflow Detection', 
					'RF-DETR': 'RF-DETR Detection',
					'Contour': 'Contour Detection'
				}
				display_method = method_display.get(detection_method, detection_method)
				st.subheader(f"🎯 Detected Chickpeas ({display_method})")
				
				if 'annotated_image' in cnt_result and cnt_result['annotated_image'] is not None:
					# Show annotated image (DETR, Roboflow, or contour)
					annotated_img = cnt_result['annotated_image']
					if detection_method == 'DETR Detection':
						# DETR annotated image is already in RGB
						st.image(annotated_img, use_column_width=True, 
							caption=f"Detected {cnt_result['count']} chickpeas using DETR")
					elif detection_method == 'Roboflow Detection':
						# Roboflow annotated image is already in RGB
						st.image(annotated_img, use_column_width=True, 
							caption=f"Detected {cnt_result['count']} chickpeas using Roboflow")
					elif detection_method == 'RF-DETR Detection':
						# RF-DETR annotated image is already in RGB
						st.image(annotated_img, use_column_width=True, 
							caption=f"Detected {cnt_result['count']} chickpeas using RF-DETR")
					else:
						# Contour visualization - convert BGR to RGB
						annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
						st.image(annotated_rgb, use_column_width=True, 
							caption=f"Detected {cnt_result.get('count', '?')} chickpeas using contours")
				# No other DETR visualizations; show image or annotated as available
				else:
					# No annotated image available
					st.image(image, use_column_width=True, 
						caption=f"Detected {cnt_result.get('count', '?')} chickpeas")
			
			# Create and display charts
			if predictions and cnt_result:
				fig1, fig2 = create_quality_charts(predictions, cnt_result)
				
				if fig1:
					st.subheader("📈 Quality Classification")
					st.plotly_chart(fig1, use_container_width=True)
				
				if fig2:
					st.subheader("📏 Size Distribution")
					st.plotly_chart(fig2, use_container_width=True)
			
			# Per-seed feature extraction from contours (only for contour method)
			if cnt_result and 'contours' in cnt_result and cnt_result['contours'] and cnt_result.get('method') != 'DETR':
				seed_features = extract_seed_features_from_contours(image, cnt_result['contours'])
				if seed_features:
					grades = derive_quality_grades(seed_features)
					st.subheader("🧪 Per-Seed Features & Sub-Grades")
					# Build table rows
					table_rows = []
					for i, f in enumerate(seed_features):
						row = {
							"Seed #": f['id'],
							"area_px": round(f['area_px'], 1),
							"length_px": round(f.get('length_px', 0.0), 1),
							"width_px": round(f.get('width_px', 0.0), 1),
							"equiv_diam_px": round(f['equivalent_diameter_px'], 2),
							"perimeter_px": round(f['perimeter_px'], 1),
							"circularity": round(f['circularity'], 3),
							"aspect_ratio": round(f['aspect_ratio'], 3),
							"solidity": round(f['solidity'], 3),
							"texture_roughness": round(f.get('texture_roughness', 0.0), 3),
							"dark_spot_frac": round(f.get('dark_spot_fraction', 0.0), 3),
							"defect_score": round(f.get('defect_score', 0.0), 3),
							"has_defect": f.get('has_defect', False),
							"color_name": f['color_name'],
							"broken_score": round(f['broken_score'], 3),
							"is_broken": f['is_broken'],
						}
						if grades:
							row.update({
								"size_grade": grades['size_grades'][i],
								"circularity_grade": grades['circularity_grades'][i],
								"color_grade": grades['color_grades'][i],
								"defect_grade": grades['defect_grades'][i],
							})
						table_rows.append(row)
					st.dataframe(table_rows, use_container_width=True)
					if grades:
						colA, colB, colC, colD = st.columns(4)
						with colA:
							st.metric("Size grade (median)", max(set(grades['size_grades']), key=grades['size_grades'].count))
						with colB:
							st.metric("Circularity grade (mode)", max(set(grades['circularity_grades']), key=grades['circularity_grades'].count))
						with colC:
							st.metric("Color grade (mode)", max(set(grades['color_grades']), key=grades['color_grades'].count))
						with colD:
							st.metric("Defect grade (mode)", max(set(grades['defect_grades']), key=grades['defect_grades'].count))

					# Lot-level metrics and radar
					lot_metrics = compute_lot_metrics(seed_features)
					if lot_metrics:
						st.subheader("📊 Lot Quality Metrics")
						mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
						with mcol1:
							st.metric("Size uniformity", f"{lot_metrics['size_uniformity']:.2f}")
						with mcol2:
							st.metric("Shape quality", f"{lot_metrics['shape_quality']:.2f}")
						with mcol3:
							st.metric("Color quality", f"{lot_metrics['color_quality']:.2f}")
						with mcol4:
							st.metric("Texture quality", f"{lot_metrics['texture_quality']:.2f}")
						with mcol5:
							st.metric("Purity", f"{lot_metrics['purity']:.2f}")

						radar = create_quality_radar(lot_metrics)
						if radar:
							st.plotly_chart(radar, use_container_width=True)

					# Annotated overlay toggle
					show_overlay = st.toggle("Show annotated overlay (per-seed)", value=False)
					if show_overlay:
						overlay = annotate_seed_overlay(image, cnt_result['contours'], seed_features)
						if overlay is not None:
							st.image(overlay, use_column_width=True, caption="Per-seed annotations (green=ok, red=defect)")
		
		# Evaluation metrics section (optional)
		if SHOW_MODEL_PERFORMANCE and model_exists and predictions:
			st.subheader("📋 Model Performance")
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Prediction Confidence", f"{predictions['confidence']:.1%}")
			with col2:
				st.metric("Model Accuracy", "See README", help="Load from training results")
			with col3:
				st.metric("Model Precision", "See README")
			with col4:
				st.metric("Model F1-Score", "See README")
	
	else:
		st.info("👆 Please upload an image to get started with chickpea quality assessment.")
		
		# Instructions
		st.subheader("📖 How to Use")
		st.markdown("""
		1. **Upload Image**: Click "Browse files" and select an image containing chickpeas
		2. **Choose Counting Method**: Toggle ML counting for dense scenes
		3. **View Results**: 
		   - See the original image and Canny edge detection preprocessing
		   - Get quality classification (if model is trained)
		   - View chickpea counts from ML and contour methods
		   - Analyze detailed metrics
		""")

if __name__ == "__main__":
	main()