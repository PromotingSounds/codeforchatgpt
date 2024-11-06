import os
import time
import traceback
import numpy as np
import boto3
import pandas as pd
import csv
import cv2
import librosa
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from scipy import ndimage, stats
from scipy.signal import correlate
import logging
import scipy
from scipy import signal
import gc
import audioflux as af
from audioflux.type import WaveletContinueType, SpectralFilterBankScaleType
from deepface import DeepFace
import moviepy.editor as mp
import openl3
from collections import Counter
import colorsys
from scipy.stats import mode
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
import json
from concurrent.futures import ThreadPoolExecutor
import pytesseract
import easyocr

# Initialize logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/video_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate required components are available"""
    try:
        required_models = {
            'CLIP': os.path.exists(f"{HF_HOME}/openai/clip-vit-base-patch32"),
            'Face Cascade': os.path.exists(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'EasyOCR': True  # Will be initialized on demand
        }
        
        # Log availability
        for model, available in required_models.items():
            if not available:
                logger.warning(f"{model} not available")
            else:
                logger.info(f"{model} available")
                
        # Initialize CLIP model if available
        if required_models['CLIP']:
            global clip_model, clip_processor
            clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
            clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
            
        # Initialize DeepFace
        try:
            DeepFace.analyze(img_path=np.zeros((100, 100, 3), dtype=np.uint8), 
                           actions=['emotion'], 
                           enforce_detection=False)
        except Exception as e:
            logger.warning(f"DeepFace initialization warning: {str(e)}")
            
        return True
        
    except Exception as e:
        logger.error(f"Environment validation error: {str(e)}")
        return False

# Configure Pytesseract with environment variable
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')

# Initialize EasyOCR as fallback
reader = None
def get_easyocr_reader():
    global reader
    if reader is None:
        try:
            reader = easyocr.Reader(['en'])
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {str(e)}")
    return reader

# Initialize AWS clients (keeping your existing setup)
s3 = boto3.client('s3')
device = torch.device('cpu')
HF_HOME = "/models/huggingface"

def validate_frame(frame):
    """Validate frame properties for vertical videos with better logging"""
    try:
        if frame is None:
            logger.debug("Frame is None")
            return False
            
        if not isinstance(frame, np.ndarray):
            logger.debug(f"Frame is {type(frame)}, not numpy array")
            return False
            
        if len(frame.shape) != 3:
            logger.debug(f"Frame has {len(frame.shape)} dimensions, expected 3")
            return False
            
        if frame.shape[2] != 3:
            logger.debug(f"Frame has {frame.shape[2]} channels, expected 3")
            return False
            
        height, width = frame.shape[:2]
        min_width = 480  # Minimum width for processing
        aspect_ratio = height / width
        
        # Log the actual dimensions and aspect ratio
        logger.info(f"Frame dimensions: {width}x{height}, Aspect ratio: {aspect_ratio:.2f}")
        
        # Check if it's a vertical video with reasonable dimensions
        # TikTok aspect ratio is typically 1080x1920 (9:16 = 0.5625 inverted = 1.77)
        # We'll allow some flexibility in the ratio
        if 1.3 <= aspect_ratio <= 2.0:  # Typical range for vertical videos
            logger.info("Detected vertical video format")
        elif aspect_ratio < 1.3:
            logger.info(f"Video appears to be horizontal or square (aspect ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > 2.0:
            logger.info(f"Video appears extremely tall (aspect ratio: {aspect_ratio:.2f})")
            
        if width < min_width:
            logger.debug(f"Frame width {width} smaller than required {min_width}")
            return False
            
        if frame.dtype != np.uint8:
            logger.debug(f"Frame has incorrect dtype: {frame.dtype}")
            return False
            
        return True
        
    except Exception as e:
        logger.debug(f"Error in frame validation: {str(e)}")
        return False

CSV_HEADERS = [
    # Basic Information
    "PostID", "TimePosted", "Username",
    
    # Audio Analysis - Rhythm
    "Genre", "Tempo", "BeatStrength", "BeatRegularity", 
    "HasTrapPattern", "HasDrillPattern", "HasBoomBapPattern", "HasDancePattern",
    "BeatCount", "AverageBPM", "BPMStability", "GrooveConsistency", "TempoStability",
    
    # Audio Analysis - Spectral
    "SpectralCentroid", "SpectralBandwidth", "SpectralRolloff",
    
    # Audio Analysis - Production
    "DynamicRange", "Compression", "Reverb", "Distortion",
    "SubBassEnergy", "BassEnergy", "LowMidsEnergy", "MidsEnergy", "HighMidsEnergy", "HighsEnergy",
    "BassIntensity", "BassVariation", "BassCharacter",
    
    # Emotional Analysis
    "DominantEmotion", "EmotionalDescription", "EmotionalRange", "EmotionalConsistency",
    "PeakEmotionalIntensity", "AverageEmotionalIntensity", "EmotionalIntensityVariation",
    
    # Scene Analysis - Location and Setting
    "Location", "SettingType", "SceneChanges", "VisualElements",
    
    # Scene Analysis - Lighting
    "LightingSummary", "BrightnessLevel", "ContrastLevel", "LightingType",
    "ColorTemperature", "LightingDirection", "LightingUniformity", "LightingEffects",
    
    # Scene Analysis - Composition
    "SymmetryScore", "RuleOfThirdsScore", "BalanceScore", "DepthScore",
    "FramingScore", "FocalPointPosition",
    
    # Performance Analysis - Basic
    "DirectAddress", "HasGestures", "EmotionalRange", "PerformanceTechniques",
    
    # Performance Analysis - Detailed
    "ConfidenceScore", "FaceSizeAverage", "PositionStability",
    "GestureFrequency", "GestureIntensity", "GestureTypes", "EnergyScore",
    
    # Engagement Analysis - Strategies
    "EngagementStrategies", "PrimaryEngagementFocus", "EffectivenessScore",
    
    # Engagement Analysis - Visual
    "TextPersistence", "EffectFrequency", "HooksFrequency", "VisualVariety",
    
    # Engagement Analysis - Performance
    "DirectAddressFrequency", "EmotionalEngagementRange", "InteractionLevel",
    "EngagementConsistency", "EngagementVariety",
    
    # Sync Analysis
    "SyncScore", "SyncQuality", "TimingQuality", "MovementAlignment",
    "StrongSyncPoints", "WeakSyncPoints", "SyncPatterns"
]

DEFAULT_FRAME_SIZE = (576, 1024)  # Target dimensions for TikTok videos
MIN_FRAME_SIZE = 480  # Minimum dimension size
DEFAULT_AUDIO_FEATURES = {
    'rhythm': {'tempo': 0, 'beat_strength': 0, 'beat_regularity': 0},
    'spectral': {'spectral_centroid': 0, 'spectral_bandwidth': 0, 'spectral_rolloff': 0},
    'production': {'dynamic_range': 0, 'compression': 0, 'reverb': 0, 'distortion': 0}
}

ANALYSIS_FRAME_LIMIT = 10

def get_video_frames_fixed(video_path, max_frames=30, sample_rate=10):
    """Extract frames with enhanced debugging"""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
            
        # Log video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Video properties - Dimensions: {width}x{height}, "
                   f"Total frames: {total_frames}, FPS: {fps}")
            
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % sample_rate != 0:
                continue
                
            if max_frames and len(frames) >= max_frames:
                break
                
            try:
                processed_frame = process_frame_safely(frame)
                if processed_frame is not None:
                    frames.append(processed_frame)
            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                continue
                
        cap.release()
        
        if not frames:
            logger.warning("No valid frames extracted from video")
        else:
            logger.info(f"Successfully extracted {len(frames)} frames")
            
        return frames
        
    except Exception as e:
        logger.error(f"Error in frame extraction: {str(e)}")
        return []

def ensure_frame_size(frame, min_width=480):
    """Ensure frame maintains minimum size while preserving aspect ratio"""
    try:
        if not isinstance(frame, np.ndarray):
            logger.debug("Invalid frame type in ensure_frame_size")
            return None
            
        if len(frame.shape) != 3:
            logger.debug(f"Invalid frame dimensions: {frame.shape}")
            return None
            
        original_shape = frame.shape
        if frame.shape[1] < min_width:  # Check width
            # Calculate new dimensions
            aspect_ratio = frame.shape[0] / frame.shape[1]
            new_width = min_width
            new_height = int(new_width * aspect_ratio)
            
            # Perform resize
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized frame from {original_shape} to {frame.shape}")
            
        return frame
        
    except Exception as e:
        logger.error(f"Error in ensure_frame_size: {str(e)}")
        return None


def download_file_from_s3(bucket, key):
    """
    Downloads a file from S3 to local storage.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        str: Path to downloaded file, or None if download fails
    """
    try:
        local_path = f"/tmp/{key.split('/')[-1]}"
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading {key}: {str(e)}")
        return None

def download_metadata_from_s3(bucket, key):
    """
    Downloads and reads CSV metadata from S3.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        pandas.DataFrame: Metadata DataFrame, or None if download fails
    """
    try:
        local_path = download_file_from_s3(bucket, key)
        if local_path:
            return pd.read_csv(local_path)
        return None
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return None

# Add this at the top level of your script with other constants
default_analysis = {
    'genre': "Unknown Genre",
    'song_structure': "Unknown Structure",
    'audio_engagement': {
        'call_response': False,
        'repetition': False,
        'hooks': [],
        'dynamics': 'moderate'
    },
    'emotions': ({}, "NA"),
    'camera': "static",
    'scene': {
        'description': "No scene data available",
        'technical_data': {
            'primary_location': "unknown",
            'lighting_summary': "unknown",
            'setting_type': "unknown",
            'scene_changes': 0
        }
    },
    'activity': {
        'activity': "No activity detected",
        'performance_type': "unknown",
        'props': [],
        'location': "unknown"
    },
    'performance': {
        'direct_address': False,
        'gestures': False,
        'emotional_range': 0.0
    },
    'engagement': {
        'strategies': [],
        'primary_focus': "unknown",
        'effectiveness_score': 0.0
    },
    'sync': {
        'sync_score': 0.0,
        'description': "No sync data available",
        'details': {'strong_sync_points': 0, 'weak_sync_points': 0}
    }
}


# Initialize models
DeepFace.analyze(img_path=np.zeros((100, 100, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False)
clip_model_path = f"{HF_HOME}/openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Enhanced genre classification labels
GENRE_CATEGORIES = {
    'Trap': {
        'audio_features': ['heavy bass', 'high hat rolls', '808s'],
        'visual_cues': ['urban setting', 'dark lighting', 'street scene'],
        'energy_range': (0.6, 1.0)
    },
    'Sad Trap': {
        'audio_features': ['melodic elements', 'minor key', 'reverb'],
        'visual_cues': ['emotional expression', 'dark atmosphere', 'solo artist'],
        'energy_range': (0.4, 0.7)
    },
    'Folk': {
        'audio_features': ['acoustic guitar', 'natural harmonics', 'vocal clarity'],
        'visual_cues': ['outdoor setting', 'acoustic instrument', 'natural lighting'],
        'energy_range': (0.2, 0.5)
    },
    'Pop': {
        'audio_features': ['clear vocals', 'structured rhythm', 'melodic hooks'],
        'visual_cues': ['bright colors', 'dancing', 'high production'],
        'energy_range': (0.5, 0.8)
    }
    # Add more genres as needed
}

# Enhanced activity prompts (using your provided list)
activity_prompts = [
    # Urban/Street
    "artist rapping while walking through an urban street",
    "rapper performing under a single streetlight",
    "artist performing in a dimly lit alleyway",
    
    # Studio/Indoor
    "artist recording vocals in a professional studio",
    "artist playing piano while singing in a bedroom",
    "artist recording in a home studio with LED lights",
    
    # Nature/Outdoor
    "artist playing acoustic guitar on a rooftop",
    "artist performing acapella by the ocean at sunset",
    "artist recording live vocals with a portable mic in a forest",
    
    # Performance
    "artist performing with a band in a small venue",
    "artist performing to their mom",
    "artist performing in front of a small crowd",
    
    # Vehicle
    "artist singing while driving",
    "artist lip-syncing with expressive gestures in a car",
    "artist performing in front of a car",
    
    # Emotional/Atmospheric
    "artist singing softly while looking out of a rainy window",
    "artist sitting on the floor with head down, surrounded by crumpled lyrics",
    "artist recording vocals with closed eyes, conveying deep emotion",
    
    # High Energy
    "artist dancing playfully while singing in a brightly lit room",
    "artist performing with high energy on a bright, neon-lit stage",
    "artist dancing and performing with a mic",
    
    # Rural/Country
    "artist playing guitar on the porch of an old farmhouse",
    "artist singing while sitting on a hay bale",
    "artist performing by a campfire"
]

# Location classification categories
LOCATION_CATEGORIES = {
    'Indoor': {
        'Studio': ['microphone stand', 'sound panels', 'studio monitors'],
        'Bedroom': ['bed', 'personal items', 'desk', 'posters'],
        'Living Room': ['couch', 'tv', 'living room furniture'],
        'Garage': ['concrete floor', 'garage door', 'tools']
    },
    'Outdoor': {
        'Urban': ['buildings', 'street', 'city lights'],
        'Nature': ['trees', 'grass', 'sky', 'water'],
        'Stage': ['performance space', 'lights', 'audience'],
        'Rural': ['farm', 'fields', 'countryside']
    }
}

def analyze_movement_patterns(motion_patterns):
    """
    Analyzes patterns in camera/subject movement.
    
    Args:
        motion_patterns (list): List of dictionaries containing motion data
        
    Returns:
        list: Description of identified movement patterns
    """
    patterns = []
    
    # Calculate average movement metrics
    avg_magnitude = np.mean([m['magnitude'] for m in motion_patterns])
    avg_complexity = np.mean([m['complexity'] for m in motion_patterns])
    
    # Identify common movement types
    if avg_magnitude > 1.0:
        if avg_complexity > 0.5:
            patterns.append("dynamic camera work")
        else:
            patterns.append("smooth panning shots")
    elif avg_magnitude > 0.5:
        if avg_complexity > 0.3:
            patterns.append("handheld movement")
        else:
            patterns.append("subtle camera motion")
    else:
        patterns.append("minimal movement")
    
    return patterns

def generate_activity_description(primary_activity, primary_location, key_props):
    """Generate descriptive activity text"""
    try:
        description_parts = []
        
        # Add activity type
        if primary_activity and primary_activity != "unknown":
            description_parts.append(primary_activity)
            
        # Add location
        if primary_location and primary_location != "unknown":
            description_parts.append(f"in {primary_location}")
            
        # Add props if present
        if key_props:
            props_str = ", ".join(key_props)
            description_parts.append(f"using {props_str}")
            
        if description_parts:
            return " ".join(description_parts)
        return "unknown activity"
        
    except Exception as e:
        logger.error(f"Error generating activity description: {str(e)}")
        return "unknown activity"

def analyze_dominant_colors(color_data):
    """
    Analyzes color data to find dominant colors across frames.
    
    Args:
        color_data (list): List of frame color analyses
        
    Returns:
        list: List of tuples containing (color, percentage)
    """
    all_colors = []
    for frame_colors in color_data:
        for color, percentage in frame_colors:
            if percentage > 0.1:  # Only consider significant colors
                all_colors.append(tuple(color))
    
    # Cluster similar colors
    color_counter = Counter(all_colors)
    dominant = color_counter.most_common(5)
    
    return [(np.array(color), count/len(all_colors)) for color, count in dominant]

def analyze_setting_details(frame):
    """
    Analyzes frame for detailed setting information.
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        dict: Dictionary of setting features
    """
    # Convert frame to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges for structure analysis
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate basic metrics
    brightness = np.mean(gray)
    texture = np.std(gray)
    edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1])
    
    return {
        'brightness': brightness,
        'texture': texture,
        'edge_density': edge_density
    }

def detect_audio_engagement_fixed(audio_path):
    """Analyzes audio for engagement patterns with fixed parameters"""
    try:
        # Load audio with consistent sample rate
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Use consistent parameters
        n_fft = 2048
        hop_length = 512
        
        logger.info(f"Analyzing audio engagement with n_fft: {n_fft}, hop_length: {hop_length}")
        
        # Get onset envelope with fixed parameters
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        # Segment audio with safety checks
        try:
            segments = librosa.effects.split(y, top_db=30)
            if len(segments) == 0:
                segments = np.array([[0, len(y)]])
        except Exception as e:
            logger.warning(f"Error in audio segmentation: {str(e)}")
            segments = np.array([[0, len(y)]])
        
        # Analyze patterns with error handling
        try:
            patterns = {
                'call_response': detect_call_response_fixed(y, sr, segments, n_fft, hop_length),
                'repetition': detect_repetition_fixed(y, sr, segments, n_fft, hop_length),
                'hooks': detect_hooks_fixed(y, sr, n_fft, hop_length),
                'dynamics': analyze_dynamics_fixed(y, segments, n_fft, hop_length)
            }
        except Exception as e:
            logger.warning(f"Error in pattern detection: {str(e)}")
            patterns = {
                'call_response': False,
                'repetition': False,
                'hooks': [],
                'dynamics': 'moderate'
            }
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error in audio engagement analysis: {str(e)}")
        return {
            'call_response': False,
            'repetition': False,
            'hooks': [],
            'dynamics': 'moderate'
        }


def detect_call_response_fixed(y, sr, segments, n_fft, hop_length):
    """Detect call-and-response patterns safely with fixed parameters"""
    try:
        if len(segments) < 4:
            return False
        
        segment_features = []
        for start, end in segments:
            segment = y[start:end]
            if len(segment) > 0:
                # Use fixed parameters for MFCC extraction
                mfcc = librosa.feature.mfcc(
                    y=segment, 
                    sr=sr, 
                    n_mfcc=12,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                if mfcc.size > 0:
                    segment_features.append(np.mean(mfcc, axis=1))
        
        if len(segment_features) < 2:
            return False
        
        # Ensure all feature vectors have the same length
        min_length = min(len(f) for f in segment_features)
        segment_features = [f[:min_length] for f in segment_features]
        
        similarities = []
        for i in range(0, len(segment_features)-2, 2):
            if i+3 >= len(segment_features):
                break
            sim1 = np.corrcoef(segment_features[i], segment_features[i+2])[0,1]
            sim2 = np.corrcoef(segment_features[i+1], segment_features[i+3])[0,1]
            similarities.extend([sim1, sim2])
        
        return bool(similarities) and np.mean(similarities) > 0.6
        
    except Exception as e:
        logger.warning(f"Error in call-response detection: {str(e)}")
        return False

def detect_repetition_fixed(y, sr, segments, n_fft, hop_length):
    """Detect repeated patterns with fixed parameters"""
    try:
        if len(segments) < 2:
            return False
        
        # Get chromagram with fixed parameters
        chroma = librosa.feature.chroma_cqt(
            y=y, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        if chroma.size == 0:
            return False
        
        # Compute self-similarity matrix
        sim_matrix = librosa.segment.recurrence_matrix(chroma, mode='affinity')
        
        # Check for repeated patterns
        pattern_score = np.mean(sim_matrix)
        
        return pattern_score > 0.4
        
    except Exception as e:
        logger.warning(f"Error in repetition detection: {str(e)}")
        return False

def detect_hooks_fixed(y, sr, n_fft, hop_length):
    """Detect musical hooks with fixed parameters and dimension checks"""
    try:
        # Get onset envelope with fixed parameters
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Compute mel spectrogram with fixed parameters
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Find segments with high energy and repetition
        hooks = []
        
        # Split into segments
        segments = librosa.effects.split(y, top_db=30)
        
        for start, end in segments:
            segment = y[start:end]
            if len(segment) < sr:  # Skip very short segments
                continue
                
            # Get segment features with fixed parameters
            segment_mel = librosa.feature.melspectrogram(
                y=segment, 
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            if segment_mel.shape[1] > 0:  # Check if we have valid mel spectrogram
                segment_rms = np.sqrt(np.mean(segment_mel ** 2, axis=0))
                mel_rms = np.sqrt(np.mean(mel_spec ** 2, axis=0))
                
                # Check if segment has characteristics of a hook
                if (np.mean(segment_rms) > np.mean(mel_rms) * 1.2 and
                    len(segment) > sr * 1.5):  # Longer than 1.5 seconds
                    hooks.append(start / sr)
        
        return hooks
        
    except Exception as e:
        logger.warning(f"Error in hook detection: {str(e)}")
        return []

def analyze_dynamics_fixed(y, segments, n_fft, hop_length):
    """Analyze dynamic range with fixed parameters"""
    try:
        if len(segments) == 0:
            return "moderate"
        
        # Calculate RMS energy for each segment with fixed parameters
        energies = []
        for start, end in segments:
            segment = y[start:end]
            if len(segment) == 0:
                continue
            rms = librosa.feature.rms(
                y=segment,
                frame_length=n_fft,
                hop_length=hop_length
            )[0]
            if len(rms) > 0:
                energies.append(np.mean(rms))
        
        if not energies:
            return "moderate"
        
        # Analyze dynamic range
        dynamic_range = np.max(energies) / (np.min(energies) + 1e-6)
        
        if dynamic_range > 10:
            return "highly dynamic"
        elif dynamic_range > 5:
            return "moderately dynamic"
        else:
            return "consistent"
            
    except Exception as e:
        logger.warning(f"Error in dynamics analysis: {str(e)}")
        return "moderate"


def analyze_symmetry(gray):
    """
    Analyze symmetry in grayscale image.
    
    Args:
        gray (numpy.ndarray): Grayscale image
        
    Returns:
        float: Symmetry score between 0 and 1
    """
    height, width = gray.shape
    
    # Split image into left and right halves
    mid = width // 2
    left_half = gray[:, :mid]
    right_half = gray[:, mid:mid*2]
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize if halves are different sizes
    if left_half.shape != right_half_flipped.shape:
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate difference between halves
    diff = cv2.absdiff(left_half, right_half_flipped)
    symmetry_score = 1.0 - (np.mean(diff) / 255.0)
    
    return float(symmetry_score)

def analyze_camera_frame(frame):
    """Analyze single frame for camera movement"""
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            return None
            
        if len(frame.shape) != 3:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate motion metrics
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        complexity = np.std(magnitude)
        
        return {
            'magnitude': float(np.mean(magnitude)),
            'direction': float(np.mean(direction)),
            'complexity': float(complexity)
        }
        
    except Exception as e:
        logger.warning(f"Error analyzing camera frame: {str(e)}")
        return None

def analyze_camera_movement(frame):
    """Analyze camera movement in a single frame"""
    try:
        if not validate_frame(frame):
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients for motion analysis
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate motion metrics
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        complexity = np.std(magnitude)
        
        return {
            'magnitude': float(np.mean(magnitude)),
            'direction': float(np.mean(direction)),
            'complexity': float(complexity)
        }
    except Exception as e:
        logger.warning(f"Error analyzing camera movement: {str(e)}")
        return None

def analyze_activity_metrics(frame):
    """Analyze activity metrics in a single frame with fixed audio_features parameter"""
    try:
        if not validate_frame(frame):
            return None
            
        metrics = {
            'performance_type': detect_performance_type(frame),  # Now works without audio_features
            'props': detect_props(frame),
            'location': detect_detailed_location(frame),
            'activity_level': analyze_movement_intensity(frame)
        }
        
        return metrics
    except Exception as e:
        logger.warning(f"Error analyzing activity metrics: {str(e)}")
        return None

def analyze_movement_intensity(frame):
    """Helper function to analyze movement intensity"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.mean(edges)) / 255.0
    except Exception:
        return 0.0

def get_default_setting_context():
    """Return default setting context"""
    return {
        'type': 'unknown',
        'aesthetic_match': 0.0,
        'mood_alignment': 0.0,
        'visual_elements': [],
        'atmosphere': None
    }

def get_default_activity_analysis():
    """Return default activity analysis"""
    return {
        'activity': "No activity detected",
        'performance_type': "unknown",
        'props': [],
        'location': "unknown"
    }

def analyze_camera_patterns(motion_patterns, compositions):
    """
    Analyze camera movement patterns and compositional elements.
    
    Args:
        motion_patterns (list): List of motion analysis dictionaries
        compositions (list): List of composition analysis dictionaries
        
    Returns:
        str: Detected camera strategy
    """
    if not motion_patterns or not compositions:
        return "static"

    # Calculate average motion
    avg_magnitude = np.mean([m['magnitude'] for m in motion_patterns])
    avg_complexity = np.mean([m['complexity'] for m in motion_patterns])
    
    # Analyze composition stability
    composition_changes = 0
    for i in range(1, len(compositions)):
        if abs(compositions[i]['symmetry'] - compositions[i-1]['symmetry']) > 0.2:
            composition_changes += 1
    
    # Determine camera strategy
    if avg_magnitude < 0.2:
        return "Static" if composition_changes < len(compositions) * 0.1 else "Subtle"
    elif avg_magnitude < 0.5:
        if avg_complexity > 0.3:
            return "Handheld"
        return "Smooth"
    else:
        if avg_complexity > 0.6:
            return "Dynamic"
        return "Following"

def detect_focus_region(gray):
    """
    Detect region of focus in grayscale image using Laplacian variance.
    
    Args:
        gray (numpy.ndarray): Grayscale image
        
    Returns:
        tuple: (x, y, width, height) of focus region
    """
    # Calculate Laplacian variance for focus detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_map = np.absolute(laplacian)
    
    # Apply threshold to get high-focus regions
    mean_focus = np.mean(focus_map)
    _, focus_mask = cv2.threshold(focus_map, mean_focus*1.5, 255, cv2.THRESH_BINARY)
    focus_mask = focus_mask.astype(np.uint8)
    
    # Find contours of focus regions
    contours, _ = cv2.findContours(focus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour as main focus region
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, w, h)

def analyze_focal_point_position(max_loc, shape):
    """
    Analyze position of focal point relative to frame composition.
    
    Args:
        max_loc (tuple): (y, x) coordinates of focal point
        shape (tuple): (height, width) of frame
        
    Returns:
        dict: Analysis of focal point position
    """
    height, width = shape
    y, x = max_loc
    
    # Calculate relative positions
    rel_x = x / width
    rel_y = y / height
    
    # Check rule of thirds alignment
    thirds_x = abs(rel_x - 0.33) < 0.1 or abs(rel_x - 0.67) < 0.1
    thirds_y = abs(rel_y - 0.33) < 0.1 or abs(rel_y - 0.67) < 0.1
    
    # Determine position description
    position = []
    if rel_y < 0.33:
        position.append("top")
    elif rel_y > 0.67:
        position.append("bottom")
    else:
        position.append("middle")
        
    if rel_x < 0.33:
        position.append("left")
    elif rel_x > 0.67:
        position.append("right")
    else:
        position.append("center")
    
    return {
        'position': " ".join(position),
        'relative_x': float(rel_x),
        'relative_y': float(rel_y),
        'rule_of_thirds': thirds_x or thirds_y,
        'centered': 0.4 < rel_x < 0.6 and 0.4 < rel_y < 0.6
    }

def analyze_depth(frame):
    """
    Analyze depth cues in frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        dict: Depth analysis results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradient for depth cues
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Analyze gradient distribution
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)
    
    # Detect linear perspective using Hough lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
    
    has_perspective_lines = lines is not None and len(lines) > 5
    
    return {
        'gradient_strength': float(gradient_mean),
        'gradient_variation': float(gradient_std),
        'has_perspective': has_perspective_lines,
        'depth_score': float(gradient_mean * 0.5 + (gradient_std/100) * 0.5),
        'complexity': 'high' if gradient_std > 50 else 'medium' if gradient_std > 25 else 'low'
    }

def detect_complementary_colors(hue_hist):
    """
    Detect complementary color pairs in hue histogram.
    
    Args:
        hue_hist (numpy.ndarray): Histogram of hue values
        
    Returns:
        float: Score indicating presence of complementary colors
    """
    complementary_score = 0
    
    # Normalize histogram
    hue_hist = hue_hist.flatten() / np.sum(hue_hist)
    
    # Check for complementary pairs (180 degrees apart in hue circle)
    for i in range(90):  # Half of 180 degrees
        pair_score = min(hue_hist[i], hue_hist[i + 90]) * 2
        complementary_score = max(complementary_score, pair_score)
    
    return float(complementary_score)

def detect_analogous_colors(hue_hist):
    """
    Detect analogous color relationships in hue histogram.
    
    Args:
        hue_hist (numpy.ndarray): Histogram of hue values
        
    Returns:
        float: Score indicating presence of analogous colors
    """
    # Normalize histogram
    hue_hist = hue_hist.flatten() / np.sum(hue_hist)
    
    analogous_score = 0
    window_size = 30  # 30-degree window for analogous colors
    
    # Slide window through hue circle
    for i in range(180 - window_size):
        window_sum = np.sum(hue_hist[i:i+window_size])
        analogous_score = max(analogous_score, window_sum)
    
    return float(analogous_score)

def detect_triadic_colors(hue_hist):
    """
    Detect triadic color relationships in hue histogram.
    
    Args:
        hue_hist (numpy.ndarray): Histogram of hue values
        
    Returns:
        float: Score indicating presence of triadic colors
    """
    # Normalize histogram
    hue_hist = hue_hist.flatten() / np.sum(hue_hist)
    
    triadic_score = 0
    # Check for peaks 120 degrees apart
    for i in range(60):
        triad_score = min([
            hue_hist[i],
            hue_hist[(i + 60) % 180],
            hue_hist[(i + 120) % 180]
        ]) * 3
        triadic_score = max(triadic_score, triad_score)
    
    return float(triadic_score)

def detect_monochromatic_colors(hsv):
    """
    Detect monochromatic color scheme in HSV image.
    
    Args:
        hsv (numpy.ndarray): HSV color space image
        
    Returns:
        float: Score indicating presence of monochromatic scheme
    """
    # Extract hue and saturation channels
    hue = hsv[:,:,0]
    saturation = hsv[:,:,1]
    
    # Calculate hue consistency
    hue_hist = cv2.calcHist([hue], [0], None, [180], [0,180])
    hue_hist = hue_hist.flatten() / np.sum(hue_hist)
    
    # Find dominant hue
    dominant_hue_idx = np.argmax(hue_hist)
    
    # Calculate percentage of pixels within 15 degrees of dominant hue
    hue_range = 15
    in_range_mask = np.logical_or(
        np.abs(hue - dominant_hue_idx) <= hue_range,
        np.abs(hue - dominant_hue_idx) >= 180 - hue_range
    )
    
    # Weight by saturation to ignore desaturated colors
    weighted_mask = in_range_mask * (saturation / 255.0)
    monochromatic_score = np.mean(weighted_mask)
    
    return float(monochromatic_score)

def calculate_harmony_score(harmony_patterns):
    """
    Calculate overall color harmony score from different patterns.
    
    Args:
        harmony_patterns (dict): Dictionary of different harmony pattern scores
        
    Returns:
        float: Overall harmony score
    """
    weights = {
        'complementary': 0.3,
        'analogous': 0.3,
        'triadic': 0.2,
        'monochromatic': 0.2
    }
    
    harmony_score = sum(
        score * weights[pattern]
        for pattern, score in harmony_patterns.items()
    )
    
    return float(harmony_score)

def is_fade_transition(prev_frame, curr_frame):
    """
    Detect if transition between frames is a fade.
    
    Args:
        prev_frame (numpy.ndarray): Previous frame
        curr_frame (numpy.ndarray): Current frame
        
    Returns:
        bool: True if fade transition detected
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean intensities
    prev_mean = np.mean(prev_gray)
    curr_mean = np.mean(curr_gray)
    
    # Calculate frame difference
    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = np.mean(diff)
    
    # Check for smooth intensity change
    intensity_change = abs(prev_mean - curr_mean)
    is_smooth = mean_diff < 50  # Low local differences
    
    return is_smooth and intensity_change > 30

def is_wipe_transition(diff):
    """
    Detect if frame difference indicates a wipe transition.
    
    Args:
        diff (numpy.ndarray): Frame difference image
        
    Returns:
        bool: True if wipe transition detected
    """
    # Threshold difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Analyze largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / float(h)
    
    # Check if contour forms a vertical or horizontal line
    is_vertical = aspect_ratio < 0.2
    is_horizontal = aspect_ratio > 5
    
    return is_vertical or is_horizontal

def analyze_lighting_context(hsv):
    """
    Analyze lighting context from HSV image.
    
    Args:
        hsv (numpy.ndarray): HSV color space image
        
    Returns:
        float: Lighting context score
    """
    # Extract value channel
    value = hsv[:,:,2]
    
    # Calculate lighting metrics
    mean_brightness = np.mean(value)
    std_brightness = np.std(value)
    
    # Calculate histogram for light distribution
    hist = cv2.calcHist([value], [0], None, [256], [0,256])
    hist = hist.flatten() / hist.sum()
    
    # Analyze light distribution
    low_light = np.sum(hist[:85])  # Dark regions
    mid_light = np.sum(hist[85:170])  # Mid-tones
    high_light = np.sum(hist[170:])  # Highlights
    
    # Calculate lighting balance score
    balance_score = 1.0 - (abs(low_light - high_light) + abs(mid_light - 0.33))
    
    return float(balance_score)

def analyze_texture_context(gray):
    """
    Analyze texture context from grayscale image.
    
    Args:
        gray (numpy.ndarray): Grayscale image
        
    Returns:
        float: Texture context score
    """
    # Calculate local binary pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate texture metrics
    lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / np.sum(lbp_hist)
    
    # Calculate texture entropy
    texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
    
    # Normalize entropy score
    texture_score = texture_entropy / np.log2(n_points + 2)
    
    return float(texture_score)

def analyze_spatial_context(frame):
    """Analyze spatial relationship context in frame with fixed array comparisons"""
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            return get_default_spatial_metrics()

        height, width = frame.shape[:2]
        
        # Convert to grayscale
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        except Exception:
            return get_default_spatial_metrics()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze spatial distribution
        horizontal_profile = np.mean(edges, axis=0)
        vertical_profile = np.mean(edges, axis=1)
        
        # Calculate spatial metrics safely
        try:
            horizontal_variance = float(np.var(horizontal_profile))
            vertical_variance = float(np.var(vertical_profile))
        except Exception:
            horizontal_variance = 0.0
            vertical_variance = 0.0
        
        # Detect lines for perspective analysis with explicit boolean checks
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
            has_strong_lines = False
            if lines is not None:
                if isinstance(lines, np.ndarray):
                    has_strong_lines = lines.shape[0] > 5
        except Exception:
            has_strong_lines = False
        
        # Calculate spatial score with safety checks
        try:
            spatial_score = float((horizontal_variance + vertical_variance) / 2)
        except Exception:
            spatial_score = 0.0
        
        # Safe structural analysis
        try:
            structural_elements = analyze_structural_elements(edges)
        except Exception:
            structural_elements = 0.0
            
        try:
            depth_score = analyze_depth_cues(edges)
        except Exception:
            depth_score = 0.0
        
        return {
            'horizontal_complexity': float(horizontal_variance),
            'vertical_complexity': float(vertical_variance),
            'has_strong_lines': bool(has_strong_lines),
            'spatial_score': float(spatial_score),
            'structural_elements': float(structural_elements),
            'depth_score': float(depth_score)
        }
        
    except Exception as e:
        logger.warning(f"Spatial analysis error: {str(e)}")
        return get_default_spatial_metrics()

def get_default_spatial_metrics():
    """Return default spatial metrics"""
    return {
        'horizontal_complexity': 0.0,
        'vertical_complexity': 0.0,
        'has_strong_lines': False,
        'spatial_score': 0.0,
        'structural_elements': 0.0,
        'depth_score': 0.0
    }

def analyze_structural_elements(edges):
    """Analyze structural elements in edge map"""
    try:
        # Count significant edge pixels
        significant_edges = np.sum(edges > 128)
        return float(significant_edges) / (edges.shape[0] * edges.shape[1])
    except Exception:
        return 0.0

def analyze_depth_cues(edges):
    """Analyze depth cues in edge map"""
    try:
        # Simple depth analysis based on edge distribution
        h, w = edges.shape
        near_region = edges[h//2:, :]
        far_region = edges[:h//2, :]
        
        near_density = np.sum(near_region) / near_region.size
        far_density = np.sum(far_region) / far_region.size
        
        return float(abs(near_density - far_density))
    except Exception:
        return 0.0

def analyze_lighting_pattern(frame):
    """
    Analyze lighting patterns in frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        dict: Lighting pattern analysis
    """
    # Convert to different color spaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Extract lightness channel
    lightness = lab[:,:,0]
    
    # Calculate lighting metrics
    mean_light = np.mean(lightness)
    std_light = np.std(lightness)
    
    # Analyze light direction using gradients
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Determine dominant light direction
    hist_direction = np.histogram(gradient_direction, bins=8, weights=gradient_magnitude)[0]
    dominant_direction = np.argmax(hist_direction) * 45  # 8 bins * 45 degrees
    
    return {
        'intensity': float(mean_light),
        'contrast': float(std_light),
        'direction': int(dominant_direction),
        'pattern_type': 'high_key' if mean_light > 170 else 'low_key' if mean_light < 85 else 'balanced',
        'variation': 'dramatic' if std_light > 50 else 'subtle' if std_light < 25 else 'moderate'
    }

def analyze_spatial_arrangement(edges):
    """
    Analyze spatial arrangement of elements using edge information.
    
    Args:
        edges (numpy.ndarray): Edge detection result
        
    Returns:
        dict: Spatial arrangement analysis
    """
    # Calculate edge density in different regions
    height, width = edges.shape
    regions = {
        'top': edges[:height//3],
        'middle': edges[height//3:2*height//3],
        'bottom': edges[2*height//3:],
        'left': edges[:, :width//3],
        'center': edges[:, width//3:2*width//3],
        'right': edges[:, 2*width//3:]
    }
    
    # Calculate density for each region
    densities = {region: np.mean(edges)/255.0 for region, edges in regions.items()}
    
    # Analyze balance and distribution
    horizontal_balance = 1.0 - abs(densities['left'] - densities['right'])
    vertical_balance = 1.0 - abs(densities['top'] - densities['bottom'])
    
    # Determine dominant region
    dominant_region = max(densities.items(), key=lambda x: x[1])[0]
    
    return {
        'dominant_region': dominant_region,
        'horizontal_balance': float(horizontal_balance),
        'vertical_balance': float(vertical_balance),
        'overall_balance': float((horizontal_balance + vertical_balance) / 2),
        'densities': densities
    }

def detect_studio_setting(frame):
    """
    Detect if frame shows a studio setting.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        float: Confidence score for studio setting
    """
    # Convert frame for CLIP analysis
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    # Studio-related prompts
    studio_prompts = [
        "recording studio environment",
        "studio microphone setup",
        "sound treatment panels",
        "studio monitors speakers",
        "recording booth interior"
    ]
    
    # Get CLIP text features
    text_inputs = clip_processor(text=studio_prompts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        
        # Calculate similarities
        similarities = F.cosine_similarity(
            image_features.unsqueeze(1),
            text_features.unsqueeze(0)
        )
        
        # Get max similarity score
        confidence = float(torch.max(similarities))
    
    return confidence

def detect_outdoor_setting(frame):
    """
    Detect if frame shows an outdoor setting.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        float: Confidence score for outdoor setting
    """
    # Convert frame for CLIP analysis
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    # Outdoor-related prompts
    outdoor_prompts = [
        "outdoor street scene",
        "natural outdoor environment",
        "outdoor urban setting",
        "outdoor nature background",
        "sky and buildings"
    ]
    
    text_inputs = clip_processor(text=outdoor_prompts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        similarities = F.cosine_similarity(
            image_features.unsqueeze(1),
            text_features.unsqueeze(0)
        )
        confidence = float(torch.max(similarities))
    
    # Boost confidence if sky is detected
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sky_mask = cv2.inRange(hsv, (90, 30, 170), (130, 255, 255))
    if np.mean(sky_mask) > 20:
        confidence = min(1.0, confidence * 1.2)
    
    return confidence

def detect_text_enhanced(frame):
    """Detect text regions with improved error handling and size validation"""
    try:
        # Initial frame validation
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
            
        # Store original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Minimum region size requirements
        MIN_REGION_WIDTH = 40
        MIN_REGION_HEIGHT = 10
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        text_regions = []
        for contour in contours:
            if contour is None or len(contour) < 3:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip regions that are too small
            if w < MIN_REGION_WIDTH or h < MIN_REGION_HEIGHT:
                continue
                
            # Skip regions that are too large (probably not text)
            if w > original_width * 0.9 or h > original_height * 0.9:
                continue
                
            # Check aspect ratio for potential text
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 15:
                # Ensure ROI is within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, original_width - x)
                h = min(h, original_height - y)
                
                if w > 0 and h > 0:  # Validate final dimensions
                    roi = frame[y:y+h, x:x+w]
                    if roi is not None and roi.size > 0:
                        # Analyze text properties
                        properties = analyze_text_properties(roi)
                        if properties:
                            text_regions.append({
                                'region': (x, y, w, h),
                                'properties': properties
                            })
        
        return text_regions
        
    except Exception as e:
        logger.warning(f"Error in text detection: {str(e)}")
        return []

def analyze_text_properties(roi):
    """Analyze text properties with better size handling"""
    try:
        # Skip analysis if ROI is too small
        if roi.shape[0] < 10 or roi.shape[1] < 40:
            return None
            
        # Convert color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        properties = {
            'size': roi.shape[:2],
            'contrast': float(np.std(gray)),
            'color': {
                'dominant': get_dominant_color(roi),
                'background': get_background_color(roi)
            },
            'style': {
                'bold': is_bold_text(gray),
                'italic': detect_italic(gray),
                'animated': False
            }
        }
        
        return properties
        
    except Exception as e:
        logger.warning(f"Error analyzing text properties: {str(e)}")
        return None

def detect_venue_setting(frame):
    """
    Detect if frame shows a performance venue setting.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        float: Confidence score for venue setting
    """
    # Convert frame for CLIP analysis
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    # Venue-related prompts
    venue_prompts = [
        "concert venue interior",
        "performance stage setting",
        "music venue environment",
        "stage lights and equipment",
        "audience seating area"
    ]
    
    text_inputs = clip_processor(text=venue_prompts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        similarities = F.cosine_similarity(
            image_features.unsqueeze(1),
            text_features.unsqueeze(0)
        )
        confidence = float(torch.max(similarities))
    
    # Boost confidence if stage lighting is detected
    bright_spots = detect_bright_spots(frame)
    if bright_spots > 5:
        confidence = min(1.0, confidence * 1.2)
    
    return confidence

def detect_home_setting(frame):
    """
    Detect if frame shows a home/residential setting.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        float: Confidence score for home setting
    """
    # Convert frame for CLIP analysis
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    # Home-related prompts
    home_prompts = [
        "bedroom interior",
        "living room setting",
        "home environment",
        "residential room",
        "house interior"
    ]
    
    text_inputs = clip_processor(text=home_prompts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        similarities = F.cosine_similarity(
            image_features.unsqueeze(1),
            text_features.unsqueeze(0)
        )
        confidence = float(torch.max(similarities))
    
    # Adjust confidence based on indoor lighting characteristics
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    if 80 < np.mean(l_channel) < 160:  # Typical indoor lighting range
        confidence = min(1.0, confidence * 1.1)
    
    return confidence

def detect_bright_spots(frame):
    """
    Helper function to detect bright spots (potential stage lights).
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        int: Number of detected bright spots
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold for bright spots
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to remove noise
    significant_spots = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    
    return len(significant_spots)

def detect_setting_elements(frame):
    """Detect visual elements with improved error handling"""
    try:
        if not isinstance(frame, np.ndarray):
            return []

        elements = []
        
        # Convert frame for CLIP analysis
        try:
            image = Image.fromarray(frame)
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            
            # Setting elements to check
            setting_prompts = [
                "microphone stand", "sound panels", "musical instruments",
                "stage lights", "urban buildings", "natural scenery",
                "home furniture", "recording equipment", "performance stage"
            ]
            
            text_inputs = clip_processor(text=setting_prompts, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(1),
                    text_features.unsqueeze(0)
                )
                
                # Safely iterate through similarities
                if similarities.shape[1] == len(setting_prompts):
                    for i, score in enumerate(similarities[0]):
                        if score > 0.25:
                            elements.append(setting_prompts[i])
                
        except Exception as e:
            logger.debug(f"CLIP analysis error: {str(e)}")
            
        return elements
        
    except Exception as e:
        logger.debug(f"Elements detection error: {str(e)}")
        return []

def analyze_atmosphere(frame, audio_mood=None):
    """Analyze atmospheric qualities with default audio_mood"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        atmosphere = {
            'primary_mood': 'neutral',
            'mood_strength': 0.0,
            'color_contribution': {},
            'lighting_quality': 'neutral'
        }
        
        # Analyze color mood
        color_mood = analyze_color_mood(hsv)
        
        # Analyze lighting mood
        lighting_mood = analyze_lighting_mood(frame)
        
        # Combine moods safely
        combined_mood = {}
        
        # Process color mood
        if isinstance(color_mood, dict):
            for mood, score in color_mood.items():
                if isinstance(score, (int, float)):
                    combined_mood[mood] = score * 0.4
        
        # Process audio mood if provided
        if isinstance(audio_mood, dict):
            for mood, score in audio_mood.items():
                if isinstance(score, (int, float)):
                    combined_mood[mood] = combined_mood.get(mood, 0) + score * 0.6
        
        # Determine primary mood
        if combined_mood:
            primary_mood = max(combined_mood.items(), key=lambda x: x[1])
            atmosphere['primary_mood'] = primary_mood[0]
            atmosphere['mood_strength'] = float(primary_mood[1])
        
        atmosphere['color_contribution'] = color_mood
        atmosphere['lighting_quality'] = lighting_mood.get('type', 'neutral')
        
        return atmosphere
        
    except Exception as e:
        logger.warning(f"Error in atmosphere analysis: {str(e)}")
        return {
            'primary_mood': 'neutral',
            'mood_strength': 0.0,
            'color_contribution': {},
            'lighting_quality': 'neutral'
        }
def extract_audio_mood(audio_features):
    """
    Extract mood information from audio features.
    
    Args:
        audio_features (dict): Extracted audio features
        
    Returns:
        dict: Audio mood analysis
    """
    mood = {}
    
    # Analyze tempo-based mood
    tempo = audio_features['rhythm']['tempo']
    if tempo > 140:
        mood['energetic'] = 0.8
        mood['intense'] = 0.7
    elif tempo > 120:
        mood['upbeat'] = 0.7
        mood['lively'] = 0.6
    elif tempo > 90:
        mood['moderate'] = 0.7
        mood['balanced'] = 0.6
    else:
        mood['calm'] = 0.7
        mood['relaxed'] = 0.6
    
    # Analyze key-based mood
    if audio_features['tonal']['mode'] == 'major':
        mood['positive'] = 0.6
        mood['bright'] = 0.5
    else:
        mood['melancholic'] = 0.6
        mood['introspective'] = 0.5
    
    # Analyze dynamics-based mood
    if audio_features['production']['dynamic_range'] > 20:
        mood['dramatic'] = 0.7
        mood['expressive'] = 0.6
    else:
        mood['consistent'] = 0.6
        mood['steady'] = 0.5
    
    return mood

def detect_rounded_shapes(edges):
    """Helper function to detect rounded shapes"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                return True
    return False

def detect_music_props(frame):
    """Detect music-related props with improved error handling and validation"""
    try:
        if not validate_frame(frame):
            return set()

        # Define prop categories with specific items
        props = [
            # Recording Equipment
            "microphone", "pop filter", "microphone stand",
            "audio interface", "mixing board", "studio monitors",
            # Instruments
            "guitar", "piano", "keyboard", "drums", "synthesizer",
            "drum machine", "bass guitar",
            # Audio Equipment
            "headphones", "earbuds", "speakers", "airpods",
            "studio monitors", "amplifier",
            # Technology
            "laptop", "smartphone", "tablet", "camera",
            "ring light", "LED lights"
        ]

        detected_props = set()
        
        try:
            # Convert frame for CLIP analysis
            image = Image.fromarray(frame)  # Frame should already be RGB
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            
            # Create text inputs for all props
            text_inputs = clip_processor(text=props, return_tensors="pt", padding=True)
            
            # Get similarities with error handling
            try:
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    text_features = clip_model.get_text_features(**text_inputs)
                    similarities = F.cosine_similarity(
                        image_features.unsqueeze(1),
                        text_features.unsqueeze(0)
                    )
                    
                    # Safely process similarities
                    if similarities.shape[1] == len(props):
                        for i, score in enumerate(similarities[0]):
                            if score > 0.25:  # Confidence threshold
                                prop_name = props[i]
                                # Validate detection
                                if validate_prop_detection(frame, prop_name):
                                    detected_props.add(prop_name)
                                    
            except Exception as e:
                logger.debug(f"CLIP similarity calculation error: {str(e)}")
                return set()

        except Exception as e:
            logger.debug(f"CLIP analysis error: {str(e)}")
            return set()

        # Categorize props
        categorized_props = set()
        for prop in detected_props:
            if any(recording in prop.lower() for recording in ["microphone", "interface", "mixing", "monitor"]):
                categorized_props.add(f"recording_equipment:{prop}")
            elif any(instrument in prop.lower() for instrument in ["guitar", "piano", "drum", "synth", "bass"]):
                categorized_props.add(f"instrument:{prop}")
            elif any(tech in prop.lower() for tech in ["laptop", "camera", "light", "phone"]):
                categorized_props.add(f"technology:{prop}")
            elif any(audio in prop.lower() for audio in ["headphone", "speaker", "earbud", "airpod"]):
                categorized_props.add(f"audio_equipment:{prop}")
            else:
                categorized_props.add(f"other:{prop}")

        return categorized_props

    except Exception as e:
        logger.warning(f"Error in prop detection: {str(e)}")
        return set()
def validate_prop_detection(frame, prop):
    """Validate prop detection with specific characteristics"""
    try:
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density
        edge_density = np.sum(edges) / (height * width)
        
        # Prop-specific validation
        if "microphone" in prop.lower():
            # Look for vertical lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            if lines is not None:
                vertical_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                    if 75 < angle < 105:
                        vertical_lines += 1
                return vertical_lines > 0
                
        elif any(item in prop.lower() for item in ["guitar", "piano", "keyboard"]):
            # Look for large objects with specific shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                area_ratio = (w * h) / (width * height)
                return area_ratio > 0.1
                
        elif any(item in prop.lower() for item in ["headphones", "speakers"]):
            # Look for rounded shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:
                            return True
                            
        elif any(item in prop.lower() for item in ["laptop", "smartphone", "tablet"]):
            # Look for rectangular shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    if len(approx) == 4:  # Rectangle has 4 corners
                        return True
                        
        # Default edge density check for other props
        return edge_density > 0.1
        
    except Exception as e:
        logger.warning(f"Error in prop validation: {str(e)}")
        return False

def detect_instrument_shape(edges, instrument_type):
    """Helper function to detect instrument shapes"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / float(h)
    
    if instrument_type == "guitar":
        return 0.2 < aspect_ratio < 0.5  # Typical guitar aspect ratio
    elif instrument_type == "piano":
        return 2.0 < aspect_ratio < 5.0  # Typical piano aspect ratio
    
    return False

def detect_vertical_lines(edges):
    """Helper function to detect vertical lines"""
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 75 < angle < 105:  # Near vertical
                return True
    return False

def camera_analysis(frames):
    """Enhanced camera analysis with better frame handling"""
    try:
        if len(frames) < 2:
            logger.info("Not enough frames for camera analysis")
            return "static"
            
        movements = []
        compositions = []
        prev_frame = None
        frame_count = 0
        
        # Process frames safely
        processed_frames = []
        for frame in frames:
            processed = process_frame_safely(frame)
            if processed is not None:
                processed_frames.append(processed)
                
        if len(processed_frames) < 2:
            logger.info("Not enough valid frames after processing")
            return "static"
            
        for frame in processed_frames:
            try:
                # Convert to grayscale
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                if prev_frame is not None:
                    try:
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                        
                        # Calculate optical flow with error checking
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray,
                            None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        
                        if flow is None or flow.size == 0:
                            logger.debug("Empty optical flow result")
                            continue
                            
                        # Calculate motion metrics with bounds checking
                        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                        angle = np.arctan2(flow[..., 1], flow[..., 0])
                        
                        if magnitude.size == 0 or angle.size == 0:
                            continue
                            
                        local_motion = float(np.std(magnitude))
                        global_motion = float(np.mean(magnitude))
                        direction_consistency = float(np.std(angle))
                        
                        movements.append({
                            'local_motion': local_motion,
                            'global_motion': global_motion,
                            'direction_consistency': direction_consistency
                        })
                        
                    except Exception as e:
                        logger.debug(f"Error processing frame pair {frame_count}: {str(e)}")
                        continue
                        
                prev_frame = frame.copy()
                frame_count += 1
                
            except Exception as e:
                logger.debug(f"Error processing frame {frame_count}: {str(e)}")
                continue
                
        if not movements:
            return "static"
            
        # Safe averaging with bounds checking
        try:
            avg_local_motion = np.mean([m['local_motion'] for m in movements])
            avg_global_motion = np.mean([m['global_motion'] for m in movements])
            avg_direction_consistency = np.mean([m['direction_consistency'] for m in movements])
        except Exception as e:
            logger.debug(f"Error calculating averages: {str(e)}")
            return "static"
            
        # Classification with adjusted thresholds
        if avg_global_motion < 0.1:
            return "static"
        elif avg_local_motion > 1.2 and avg_direction_consistency > 0.8:
            return "handheld"
        elif avg_global_motion > 0.4 and avg_direction_consistency < 0.5:
            return "panning"
        elif avg_local_motion > 0.6 and avg_global_motion > 0.3:
            return "dynamic"
        else:
            return "smooth"
            
    except Exception as e:
        logger.error(f"Error in camera analysis: {str(e)}")
        return "static"


def analyze_edge_distribution(edges):
    """Analyze distribution of edges in frame"""
    try:
        # Split frame into 3x3 grid
        h, w = edges.shape
        cell_h, cell_w = h // 3, w // 3
        
        distributions = []
        for i in range(3):
            for j in range(3):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                distributions.append(np.mean(cell))
                
        return np.std(distributions) / (np.mean(distributions) + 1e-6)
    except Exception:
        return 1.0

def analyze_central_focus(gray):
    """Analyze if frame maintains central focus"""
    try:
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        
        # Calculate relative focus using Laplacian variance
        center_focus = cv2.Laplacian(center_region, cv2.CV_64F).var()
        full_focus = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return center_focus / (full_focus + 1e-6)
    except Exception:
        return 1.0

def scene_analysis(frames):
    """Analyze scene and location"""
    try:
        return analyze_scene_and_location(frames, None)
    except Exception as e:
        logger.error(f"Error in scene analysis: {str(e)}")
        return {
            'description': "Error analyzing scene",
            'technical_data': {
                'primary_location': "unknown",
                'lighting_summary': "unknown",
                'setting_type': "unknown",
                'scene_changes': 0
            }
        }


def performance_analysis(frames):
    """Analyze performance engagement"""
    try:
        return detect_performance_engagement_enhanced(frames)
    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}")
        return {
            'direct_address': False,
            'gestures': False,
            'emotional_range': 0.0
        }

def emotion_analysis_safe(frames, debug=True):
    """Analyze emotions with improved face detection and error handling"""
    try:
        if not frames:
            logger.warning("No frames provided for emotion analysis")
            return ({}, "NA")

        # Track all emotions and their intensities
        emotion_scores = {}
        processed_frames = 0
        emotion_sequence = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            try:
                # Frame should already be RGB, so convert directly to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with improved parameters
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Process largest face
                    face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = face
                    
                    # Add padding to face region
                    padding = int(min(w, h) * 0.1)
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(frame_bgr.shape[1], x + w + padding)
                    y_end = min(frame_bgr.shape[0], y + h + padding)
                    
                    face_roi = frame_bgr[y_start:y_end, x_start:x_end]
                    
                    try:
                        # Analyze emotions with timeout protection
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                DeepFace.analyze,
                                img_path=face_roi,
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='opencv'
                            )
                            
                            result = future.result(timeout=2)
                            
                            # Process emotion results
                            if isinstance(result, list) and result:
                                frame_emotions = result[0].get('emotion', {})
                            elif isinstance(result, dict):
                                frame_emotions = result.get('emotion', {})
                            else:
                                continue
                                
                            # Validate and accumulate emotion scores
                            if frame_emotions:
                                emotion_sequence.append(frame_emotions)
                                for emotion, score in frame_emotions.items():
                                    if emotion not in emotion_scores:
                                        emotion_scores[emotion] = 0
                                    emotion_scores[emotion] += score
                                processed_frames += 1
                                
                    except Exception as e:
                        logger.warning(f"Error analyzing emotions in frame {frame_idx}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {str(e)}")
                continue
                
        # Generate final emotion analysis
        if processed_frames > 0:
            # Average emotion scores
            emotion_scores = {k: v/processed_frames for k, v in emotion_scores.items()}
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            # Calculate emotional range
            emotional_range = calculate_emotional_range(emotion_sequence)
            
            # Generate description
            intensity = categorize_emotional_intensity(dominant_emotion[1])
            emotion_description = f"{intensity} {dominant_emotion[0]}"
            
            if emotional_range > 0.3:
                emotion_description += f" with varied emotional expression (range: {emotional_range:.2f})"
            
            return (emotion_scores, emotion_description)
            
        return ({}, "NA")
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        return ({}, "NA")

def calculate_mood_alignment(aesthetics, audio_mood):
    """Calculate alignment between visual aesthetics and audio mood with type checking"""
    alignment_score = 0.0
    
    # Check composition alignment
    if isinstance(aesthetics, dict) and isinstance(audio_mood, dict):
        if aesthetics.get('composition', {}).get('symmetry', 0) > 0.7:
            if 'balanced' in audio_mood or 'steady' in audio_mood:
                alignment_score += 0.3
        
        # Check color harmony alignment
        if aesthetics.get('color_harmony', 0) > 0.7:
            if 'positive' in audio_mood or 'bright' in audio_mood:
                alignment_score += 0.3
        
        # Check depth and complexity alignment
        if aesthetics.get('depth', {}).get('complexity') == 'high':
            if 'dramatic' in audio_mood or 'expressive' in audio_mood:
                alignment_score += 0.4
    
    return float(alignment_score)

def safe_multiply_mood_values(mood_dict, factor):
    """Safely multiply mood values by a factor"""
    if not isinstance(mood_dict, dict):
        return {}
    
    result = {}
    for mood, value in mood_dict.items():
        if isinstance(value, (int, float)):
            result[mood] = float(value) * factor
    return result

def analyze_lighting_environment(frame):
    """Enhanced lighting analysis keeping original function name"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Calculate basic lighting metrics
        brightness = np.mean(lab[:,:,0])
        contrast = np.std(lab[:,:,0])
        
        # Calculate lighting uniformity
        blocks = np.array_split(lab[:,:,0], 16)
        block_means = [np.mean(block) for block in blocks]
        uniformity = 1.0 - (np.std(block_means) / (np.mean(block_means) + 1e-6))
        
        # Determine lighting type
        if brightness > 180:
            lighting_type = "bright"
        elif brightness > 140:
            lighting_type = "well-lit"
        elif brightness > 100:
            lighting_type = "moderate"
        else:
            lighting_type = "dark"
            
        # Analyze lighting direction
        grad_x = cv2.Sobel(lab[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(lab[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(np.mean(grad_y), np.mean(grad_x))
        direction = int(np.degrees(angle))
        
        # Analyze color temperature
        b, g, r = cv2.split(frame)
        rb_ratio = np.mean(r) / (np.mean(b) + 1e-6)
        if rb_ratio > 1.2:
            temperature = "warm"
        elif rb_ratio < 0.8:
            temperature = "cool"
        else:
            temperature = "neutral"
        
        return {
            'type': lighting_type,
            'direction': direction,
            'temperature': temperature,
            'uniformity': float(uniformity),
            'contrast': float(contrast),
            'description': f"{lighting_type} {temperature} lighting"
        }
        
    except Exception as e:
        logger.error(f"Error in lighting analysis: {str(e)}")
        return {
            'type': "unknown",
            'direction': 0,
            'temperature': "unknown",
            'uniformity': 0.0,
            'contrast': 0.0,
            'description': "unknown lighting"
        }

def analyze_spatial_environment(frame):
    """Analyze spatial characteristics with fixed array handling"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions safely
        height, width = edges.shape
        edge_density = {}
        
        for region_name, region_slice in [
            ('top', (0, height//3)),
            ('middle', (height//3, 2*height//3)),
            ('bottom', (2*height//3, height))
        ]:
            region = edges[region_slice[0]:region_slice[1], :]
            edge_density[region_name] = float(np.sum(region)) / region.size
        
        # Find lines with explicit handling
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        depth_cues = bool(lines is not None and len(lines) > 5)
        
        # Calculate edge complexity safely
        edge_complexity = float(np.mean(edges) / 255.0)
        
        # Calculate perspective strength
        perspective_strength = 0.0
        if lines is not None and len(lines) > 0:
            try:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2-y1, x2-x1))
                    angles.append(angle)
                perspective_strength = float(np.std(angles))
            except Exception:
                perspective_strength = 0.0
        
        return {
            'depth_cues': depth_cues,
            'edge_complexity': float(edge_complexity),
            'spatial_distribution': dict(edge_density),
            'perspective_strength': float(perspective_strength)
        }
        
    except Exception as e:
        logger.warning(f"Error in spatial environment analysis: {str(e)}")
        return get_default_space()

def determine_lighting_type(mean_brightness, std_brightness):
    """Helper function to determine lighting type"""
    if mean_brightness > 180:
        return 'high_key'
    elif mean_brightness < 70:
        return 'low_key'
    elif std_brightness > 60:
        return 'dramatic'
    else:
        return 'balanced'

def calculate_perspective_strength(lines):
    """Helper function to calculate perspective strength"""
    if not lines:
        return 0.0
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1))
        angles.append(angle)
    
    # Calculate variance of angles
    angle_variance = np.var(angles)
    return float(min(1.0, angle_variance * 5))

def analyze_color_temperature(frame):
    """Helper function to analyze color temperature"""
    b, g, r = cv2.split(frame)
    rb_ratio = np.mean(r) / (np.mean(b) + 1e-6)
    
    if rb_ratio > 1.2:
        return 'warm'
    elif rb_ratio < 0.8:
        return 'cool'
    else:
        return 'neutral'

def analyze_light_quality(lab):
    """Helper function to analyze light quality"""
    l_channel = lab[:,:,0]
    mean_brightness = np.mean(l_channel)
    std_brightness = np.std(l_channel)
    
    if std_brightness < 20:
        return 'flat'
    elif std_brightness > 50:
        return 'contrasty'
    else:
        return 'balanced'

def analyze_atmospheric_depth(hsv):
    """Helper function to analyze atmospheric depth"""
    saturation = hsv[:,:,1]
    value = hsv[:,:,2]
    
    # Calculate depth metrics
    sat_gradient = np.mean(np.gradient(saturation))
    val_gradient = np.mean(np.gradient(value))
    
    depth_score = (abs(sat_gradient) + abs(val_gradient)) / 2
    
    if depth_score > 0.5:
        return 'deep'
    elif depth_score > 0.2:
        return 'moderate'
    else:
        return 'shallow'

def determine_atmospheric_mood(color_temp, light_quality, depth):
    """Helper function to determine atmospheric mood"""
    moods = {
        ('warm', 'balanced', 'moderate'): 'inviting',
        ('cool', 'contrasty', 'deep'): 'dramatic',
        ('neutral', 'flat', 'shallow'): 'neutral',
        ('warm', 'contrasty', 'deep'): 'intense',
        ('cool', 'balanced', 'moderate'): 'calm'
    }
    
    return moods.get((color_temp, light_quality, depth), 'neutral')


def analyze_subgenre(features, genre_probs):
    """
    Analyze subgenre characteristics based on primary genre with improved error handling.
    """
    try:
        if not features or not genre_probs:
            return None
            
        primary_genre = max(genre_probs.items(), key=lambda x: x[1])[0]
        subgenre_details = {
            'name': None,
            'confidence': 0.0,
            'characteristics': []
        }
        
        # Get production features safely
        production = features.get('production', {})
        reverb = production.get('reverb', 0)
        
        # Get tonal features safely
        tonal = features.get('tonal', {})
        mode = tonal.get('mode', 'major')
        
        # Get spectral features safely
        spectral = features.get('spectral', {})
        spectral_centroid = np.mean(spectral.get('spectral_centroid', [0]))
        
        if primary_genre == 'Trap':
            if mode == 'minor':
                if reverb > 0.7:
                    subgenre_details['name'] = 'Ambient Trap'
                    subgenre_details['confidence'] = 0.8
                    subgenre_details['characteristics'] = ['reverb heavy', 'atmospheric']
                else:
                    subgenre_details['name'] = 'Dark Trap'
                    subgenre_details['confidence'] = 0.7
                    subgenre_details['characteristics'] = ['minor key', 'aggressive']
            else:
                subgenre_details['name'] = 'Modern Trap'
                subgenre_details['confidence'] = 0.6
                subgenre_details['characteristics'] = ['melodic', 'upbeat']
                
        elif primary_genre == 'Sad Trap':
            if reverb > 0.8:
                subgenre_details['name'] = 'Cloud Rap'
                subgenre_details['confidence'] = 0.75
                subgenre_details['characteristics'] = ['dreamy', 'atmospheric']
            else:
                subgenre_details['name'] = 'Emo Rap'
                subgenre_details['confidence'] = 0.7
                subgenre_details['characteristics'] = ['emotional', 'melodic']
                
        elif primary_genre == 'Folk':
            if spectral_centroid < 2000:
                subgenre_details['name'] = 'Traditional Folk'
                subgenre_details['confidence'] = 0.8
                subgenre_details['characteristics'] = ['acoustic', 'traditional']
            else:
                subgenre_details['name'] = 'Modern Folk'
                subgenre_details['confidence'] = 0.7
                subgenre_details['characteristics'] = ['contemporary', 'produced']
                
        elif primary_genre == 'Pop':
            compression = production.get('compression', 0)
            if compression > 0.8:
                subgenre_details['name'] = 'Commercial Pop'
                subgenre_details['confidence'] = 0.8
                subgenre_details['characteristics'] = ['polished', 'produced']
            else:
                subgenre_details['name'] = 'Indie Pop'
                subgenre_details['confidence'] = 0.7
                subgenre_details['characteristics'] = ['raw', 'authentic']
        
        return subgenre_details
        
    except Exception as e:
        logger.warning(f"Error in subgenre analysis: {str(e)}")
        return None


def detect_genre_fusion(features):
    """
    Detect fusion elements from different genres with improved error handling.
    """
    try:
        if not features:
            return []
            
        fusion_elements = []
        
        # Get rhythm patterns safely
        rhythm = features.get('rhythm', {})
        patterns = rhythm.get('rhythm_patterns', {})
        
        # Check for trap elements
        if patterns.get('trap', False):
            fusion_elements.append('trap')
        
        # Check for drill elements
        if patterns.get('drill', False):
            fusion_elements.append('drill')
        
        # Check for boom bap elements
        if patterns.get('boom_bap', False):
            fusion_elements.append('boom bap')
        
        # Check for dance elements
        if patterns.get('dance', False):
            fusion_elements.append('dance')
        
        # Get production features safely
        production = features.get('production', {})
        reverb = production.get('reverb', 0)
        distortion = production.get('distortion', 0)
        
        # Check production style elements
        if reverb > 0.7:
            fusion_elements.append('ambient')
        if distortion > 0.6:
            fusion_elements.append('rock')
        
        # Get tonal features safely
        tonal = features.get('tonal', {})
        harmony_complexity = tonal.get('harmony_complexity', 0)
        
        # Check tonal elements
        if harmony_complexity > 0.7:
            fusion_elements.append('jazz')
        
        return fusion_elements
        
    except Exception as e:
        logger.warning(f"Error in fusion detection: {str(e)}")
        return []

def analyze_beat_regularity(beat_times):
    """Analyze regularity of beat patterns"""
    try:
        if len(beat_times) < 2:
            return 0.0
            
        # Calculate intervals between beats
        intervals = np.diff(beat_times)
        
        # Calculate regularity as inverse of relative standard deviation
        mean_interval = np.mean(intervals)
        if mean_interval > 0:
            std_interval = np.std(intervals)
            regularity = 1.0 - (std_interval / mean_interval)
            return float(max(0.0, min(1.0, regularity)))
            
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error in beat regularity analysis: {str(e)}")
        return 0.0

def analyze_tonal_features(chromagram, sr):
    """Analyze tonal features of the audio"""
    try:
        tonal_features = {
            'key': 'C',
            'mode': 'major',
            'harmony_complexity': 0.0,
            'chord_progression': []
        }
        
        # Analyze key and mode
        chroma_avg = np.mean(chromagram, axis=1)
        key_idx = np.argmax(chroma_avg)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tonal_features['key'] = keys[key_idx]
        
        # Determine mode
        major_profile = np.roll([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], key_idx)
        minor_profile = np.roll([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], key_idx)
        
        major_correlation = np.corrcoef(chroma_avg, major_profile)[0,1]
        minor_correlation = np.corrcoef(chroma_avg, minor_profile)[0,1]
        
        tonal_features['mode'] = 'major' if major_correlation > minor_correlation else 'minor'
        
        # Calculate harmony complexity
        tonal_features['harmony_complexity'] = float(np.std(chromagram))
        
        return tonal_features
        
    except Exception as e:
        logger.warning(f"Error in tonal analysis: {str(e)}")
        return {
            'key': 'C',
            'mode': 'major',
            'harmony_complexity': 0.0,
            'chord_progression': []
        }

def detect_energy_peaks(energy):
    """Detect significant peaks in energy profile"""
    try:
        peaks = []
        window_size = 5
        
        for i in range(window_size, len(energy) - window_size):
            window = energy[i-window_size:i+window_size+1]
            if energy[i] == max(window) and energy[i] > np.mean(energy):
                peaks.append(i)
                
        return peaks
        
    except Exception as e:
        logger.debug(f"Error detecting energy peaks: {str(e)}")
        return []

def detect_drill_pattern_fixed(y, sr, n_fft, hop_length):
    """Detect drill rhythm patterns with fixed parameters and improved dimension checks"""
    try:
        logger.debug("Starting drill pattern detection")
        
        # Filter for bass frequencies
        nyquist = sr // 2
        filter_freq = 250 / nyquist
        b, a = scipy.signal.butter(4, filter_freq, btype='low')
        y_bass = scipy.signal.filtfilt(b, a, y)
        
        # Get onset envelope for bass
        bass_onset_env = librosa.onset.onset_strength(
            y=y_bass,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Get onset times
        onsets = librosa.onset.onset_detect(
            onset_envelope=bass_onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        # Check if we have enough onsets
        if not isinstance(onsets, np.ndarray) or len(onsets) < 4:
            return False
            
        # Get onset times in seconds
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
        
        # Calculate intervals with explicit dimension check
        intervals = np.diff(onset_times)
        if intervals.size < 1:
            return False
            
        # Calculate rhythm metrics
        std_dev = np.std(intervals)
        mean_interval = np.mean(intervals)
        
        if mean_interval == 0:
            return False
            
        rhythm_irregularity = std_dev / mean_interval
        has_irregular_rhythm = 0.2 < rhythm_irregularity < 0.4
        
        # Check for sliding bass
        stft = librosa.stft(y_bass, n_fft=n_fft, hop_length=hop_length)
        
        if stft.shape[1] < 2:  # Need at least 2 frames for diff
            return False
            
        magnitudes = np.abs(stft)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Track frequency peaks over time
        peak_freqs = []
        for frame in range(magnitudes.shape[1]):
            frame_magnitudes = magnitudes[:, frame]
            if np.any(frame_magnitudes > 0):
                peak_idx = np.argmax(frame_magnitudes)
                peak_freqs.append(frequencies[peak_idx])
                
        # Check for sliding bass
        if len(peak_freqs) > 1:
            freq_changes = np.diff(peak_freqs)
            has_sliding_bass = np.any(np.abs(freq_changes) > 10)
        else:
            has_sliding_bass = False
        
        return has_irregular_rhythm and has_sliding_bass
        
    except Exception as e:
        logger.warning(f"Error in drill pattern detection: {str(e)}")
        return False

def detect_trap_pattern_fixed(y, sr, n_fft, hop_length):
    """Detect trap rhythm patterns with fixed parameters"""
    try:
        # Filter for bass frequencies
        nyquist = sr // 2
        filter_freq = 250 / nyquist
        b, a = scipy.signal.butter(4, filter_freq, btype='low')
        y_bass = scipy.signal.filtfilt(b, a, y)
        
        # Get onset envelope for bass frequencies
        bass_onset_env = librosa.onset.onset_strength(
            y=y_bass,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Detect bass hits
        bass_onsets = librosa.onset.onset_detect(
            onset_envelope=bass_onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        if len(bass_onsets) > 0:
            bass_intervals = np.diff(bass_onsets)
            has_808_pattern = np.any(bass_intervals == 2) or np.any(bass_intervals == 4)
            
            # Check for hi-hat rolls
            y_high = librosa.effects.preemphasis(y)
            high_onset_env = librosa.onset.onset_strength(
                y=y_high,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            high_onsets = librosa.onset.onset_detect(
                onset_envelope=high_onset_env,
                sr=sr,
                hop_length=hop_length
            )
            
            has_hihat_rolls = len(high_onsets) > 0 and np.any(np.diff(high_onsets) < sr/8)
            
            return has_808_pattern and has_hihat_rolls
        
        return False
        
    except Exception as e:
        logger.warning(f"Error in trap pattern detection: {str(e)}")
        return False


def detect_boom_bap_pattern_fixed(y, sr, n_fft, hop_length):
    """Detect boom bap rhythm patterns with fixed parameters and broadcasting fix"""
    try:
        # Get kick drum onset envelope
        kick_onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            feature=librosa.feature.melspectrogram,
            fmin=20,
            fmax=200
        )
        
        # Get snare drum onset envelope
        snare_onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            feature=librosa.feature.melspectrogram,
            fmin=200,
            fmax=2000
        )
        
        # Detect onsets with fixed parameters
        kicks = librosa.onset.onset_detect(
            onset_envelope=kick_onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        snares = librosa.onset.onset_detect(
            onset_envelope=snare_onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        if len(kicks) > 1 and len(snares) > 1:
            # Convert to time domain
            kick_times = librosa.frames_to_time(kicks, sr=sr, hop_length=hop_length)
            snare_times = librosa.frames_to_time(snares, sr=sr, hop_length=hop_length)
            
            # Calculate intervals
            kick_intervals = np.diff(kick_times)
            snare_intervals = np.diff(snare_times)
            
            # Check for regular patterns
            has_regular_kick = np.std(kick_intervals) < 0.2 * np.mean(kick_intervals)
            has_backbeat_snare = np.median(snare_intervals) > 0.5
            
            # Check for swing (using kick intervals only if enough are available)
            has_swing = False
            if len(kick_intervals) >= 2:
                # Reshape to handle odd number of intervals
                even_intervals = kick_intervals[::2][:len(kick_intervals)//2]
                odd_intervals = kick_intervals[1::2][:len(kick_intervals)//2]
                if len(even_intervals) == len(odd_intervals):
                    swing_ratio = even_intervals / odd_intervals
                    has_swing = np.mean(np.abs(swing_ratio - 1.5)) < 0.2
            
            return has_regular_kick and has_backbeat_snare and has_swing
            
        return False
        
    except Exception as e:
        logger.warning(f"Error in boom bap pattern detection: {str(e)}")
        return False

def detect_dance_pattern_fixed(y, sr, n_fft, hop_length):
    """Detect dance music rhythm patterns with fixed parameters"""
    try:
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        if len(beats) > 0:
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            beat_intervals = np.diff(beat_times)
            is_regular = np.std(beat_intervals) < 0.05 * np.mean(beat_intervals)
            
            # Check for characteristic kick pattern
            kick_onset_env = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                feature=librosa.feature.melspectrogram,
                fmin=20,
                fmax=200
            )
            
            # FIXED THIS LINE
            kicks = librosa.onset.onset_detect(
                onset_envelope=kick_onset_env,
                sr=sr,
                hop_length=hop_length
            )
            
            if len(kicks) > 0:
                kick_intervals = np.diff(librosa.frames_to_time(kicks, sr=sr, hop_length=hop_length))
                has_four_on_floor = np.std(kick_intervals) < 0.1 * np.mean(kick_intervals)
                
                is_dance_tempo = 115 <= tempo <= 135
                
                return is_regular and has_four_on_floor and is_dance_tempo
                
        return False
        
    except Exception as e:
        logger.warning(f"Error in dance pattern detection: {str(e)}")
        return False



def calculate_pose_movement(prev_pose, curr_pose):
    """
    Calculate movement between consecutive poses.
    
    Args:
        prev_pose (list): Previous pose keypoints
        curr_pose (list): Current pose keypoints
        
    Returns:
        float: Movement score
    """
    if not prev_pose or not curr_pose:
        return 0.0
    
    movements = []
    for p1, p2 in zip(prev_pose, curr_pose):
        if p1 is not None and p2 is not None:
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            movements.append(distance)
    
    return np.mean(movements) if movements else 0.0

def calculate_pose_engagement(pose_data):
    """
    Calculate engagement level from pose data.
    
    Args:
        pose_data (list): List of pose keypoints
        
    Returns:
        float: Engagement score
    """
    if not pose_data:
        return 0.0
    
    # Calculate key metrics
    arm_spread = calculate_arm_spread(pose_data)
    shoulder_openness = calculate_shoulder_openness(pose_data)
    head_position = calculate_head_position(pose_data)
    stance = calculate_stance_stability(pose_data)
    
    # Weight and combine metrics
    engagement_score = (
        arm_spread * 0.3 +
        shoulder_openness * 0.3 +
        head_position * 0.2 +
        stance * 0.2
    )
    
    return float(engagement_score)

def calculate_upright_posture(pose_data):
    """
    Calculate how upright the posture is.
    
    Args:
        pose_data (list): Pose keypoints
        
    Returns:
        float: Upright posture score
    """
    if not pose_data or len(pose_data) < 17:  # Need full body keypoints
        return 0.0
    
    # Get spine keypoints (neck to hip)
    neck = pose_data[1]
    mid_hip = pose_data[8]
    
    if neck is None or mid_hip is None:
        return 0.0
    
    # Calculate spine angle relative to vertical
    dx = neck[0] - mid_hip[0]
    dy = neck[1] - mid_hip[1]
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    
    # Convert angle to score (0 = horizontal, 1 = vertical)
    return max(0.0, min(1.0, 1.0 - angle/90.0))

def calculate_shoulder_position(pose_data):
    """
    Calculate shoulder position score.
    
    Args:
        pose_data (list): Pose keypoints
        
    Returns:
        float: Shoulder position score
    """
    if not pose_data or len(pose_data) < 7:  # Need shoulder points
        return 0.0
    
    left_shoulder = pose_data[5]
    right_shoulder = pose_data[6]
    
    if left_shoulder is None or right_shoulder is None:
        return 0.0
    
    # Calculate shoulder line angle
    angle = abs(np.degrees(np.arctan2(
        right_shoulder[1] - left_shoulder[1],
        right_shoulder[0] - left_shoulder[0]
    )))
    
    # Score based on how level shoulders are (0 = vertical, 1 = horizontal)
    return max(0.0, min(1.0, 1.0 - angle/90.0))

def calculate_head_position(pose_data):
    """
    Calculate head position score.
    
    Args:
        pose_data (list): Pose keypoints
        
    Returns:
        float: Head position score
    """
    if not pose_data or len(pose_data) < 5:  # Need nose and eye points
        return 0.0
    
    nose = pose_data[0]
    left_eye = pose_data[1]
    right_eye = pose_data[2]
    
    if nose is None or left_eye is None or right_eye is None:
        return 0.0
    
    # Calculate head tilt
    eye_center = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)
    tilt_angle = abs(np.degrees(np.arctan2(
        eye_center[1] - nose[1],
        eye_center[0] - nose[0]
    )))
    
    # Score based on how level head is
    return max(0.0, min(1.0, 1.0 - tilt_angle/45.0))

def calculate_stance_stability(pose_data):
    """
    Calculate stability of stance.
    
    Args:
        pose_data (list): Pose keypoints
        
    Returns:
        float: Stance stability score
    """
    if not pose_data or len(pose_data) < 13:  # Need hip and ankle points
        return 0.0
    
    left_hip = pose_data[11]
    right_hip = pose_data[12]
    left_ankle = pose_data[13]
    right_ankle = pose_data[14]
    
    if None in [left_hip, right_hip, left_ankle, right_ankle]:
        return 0.0
    
    # Calculate base of support width
    foot_width = abs(right_ankle[0] - left_ankle[0])
    hip_width = abs(right_hip[0] - left_hip[0])
    
    # Score based on ratio of foot width to hip width
    stance_ratio = foot_width / (hip_width + 1e-6)
    
    # Normalize score (optimal stance is typically shoulder-width)
    return max(0.0, min(1.0, 1.0 - abs(stance_ratio - 1.0)))

def is_emphasis_gesture(sequence):
    """
    Detect emphasis gestures in pose sequence.
    
    Args:
        sequence (list): Sequence of pose keypoints
        
    Returns:
        bool: True if emphasis gesture detected
    """
    if len(sequence) < 5:
        return False
    
    # Track hand movements
    hand_movements = []
    for i in range(1, len(sequence)):
        prev_hands = [sequence[i-1][7], sequence[i-1][8]]  # Wrist points
        curr_hands = [sequence[i][7], sequence[i][8]]
        
        if None not in prev_hands and None not in curr_hands:
            movement = max(
                np.sqrt((curr_hands[0][0] - prev_hands[0][0])**2 + 
                       (curr_hands[0][1] - prev_hands[0][1])**2),
                np.sqrt((curr_hands[1][0] - prev_hands[1][0])**2 + 
                       (curr_hands[1][1] - prev_hands[1][1])**2)
            )
            hand_movements.append(movement)
    
    if not hand_movements:
        return False
    
    # Check for sharp, deliberate movements
    max_movement = max(hand_movements)
    return max_movement > 50 and np.std(hand_movements) > 20

def is_rhythmic_gesture_fixed(sequence):
    """Detect rhythmic gestures with dimension checks"""
    if len(sequence) < 5:
        return False
    
    # Track joint movements
    movements = []
    for i in range(1, len(sequence)):
        movement = calculate_pose_movement(sequence[i-1], sequence[i])
        movements.append(movement)
    
    if len(movements) < 2:  # Need at least 2 movements for diff
        return False
    
    # Calculate rhythm metrics with dimension check
    try:
        movement_intervals = np.diff(movements)
        if movement_intervals.size == 0:
            return False
            
        mean_movements = np.mean(movements)
        if mean_movements < 1e-6:  # Avoid division by zero
            return False
            
        regularity = 1.0 - (np.std(movement_intervals) / mean_movements)
        
        return regularity > 0.6 and mean_movements > 10
        
    except Exception as e:
        logger.warning(f"Error in rhythmic gesture detection: {str(e)}")
        return False

def is_flowing_gesture(sequence):
    """
    Detect flowing, continuous gestures.
    
    Args:
        sequence (list): Sequence of pose keypoints
        
    Returns:
        bool: True if flowing gesture detected
    """
    if len(sequence) < 5:
        return False
    
    # Track movement continuity
    movements = []
    for i in range(1, len(sequence)):
        movement = calculate_pose_movement(sequence[i-1], sequence[i])
        movements.append(movement)
    
    if not movements:
        return False
    
    # Calculate movement smoothness
    movement_changes = np.diff(movements)
    smoothness = 1.0 - (np.std(movement_changes) / (np.mean(movements) + 1e-6))
    
    return smoothness > 0.7 and np.mean(movements) > 5

def detect_rhythmic_gestures(pose_data):
    """
    Detect rhythmic patterns in pose movements.
    
    Args:
        pose_data (list): Sequence of pose keypoints
        
    Returns:
        dict: Detected rhythmic patterns
    """
    patterns = {
        'repetitive': False,
        'synchronized': False,
        'intensity': 0.0,
        'confidence': 0.0
    }
    
    if len(pose_data) < 10:
        return patterns
    
    # Track movements of key joints
    movements = []
    for i in range(1, len(pose_data)):
        movement = calculate_pose_movement(pose_data[i-1], pose_data[i])
        movements.append(movement)
    
    if movements:
        # Analyze movement patterns
        movement_intervals = np.diff(movements)
        regularity = 1.0 - (np.std(movement_intervals) / (np.mean(movements) + 1e-6))
        
        patterns['repetitive'] = regularity > 0.6
        patterns['intensity'] = float(np.mean(movements))
        patterns['confidence'] = float(regularity)
        
        # Check for rhythm synchronization
        if len(movements) > 4:
            # Look for periodic patterns
            peaks = scipy.signal.find_peaks(movements)[0]
            if len(peaks) >= 2:
                peak_intervals = np.diff(peaks)
                sync_regularity = 1.0 - (np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-6))
                patterns['synchronized'] = sync_regularity > 0.7
    
    return patterns

def detect_emphatic_gestures(pose_data):
    """
    Detect emphatic gestures in pose sequence.
    
    Args:
        pose_data (list): Sequence of pose keypoints
        
    Returns:
        dict: Detected emphatic gestures
    """
    gestures = {
        'emphasis_points': [],
        'intensity': 0.0,
        'confidence': 0.0
    }
    
    if len(pose_data) < 5:
        return gestures
    
    # Analyze movement sequences
    for i in range(4, len(pose_data)):
        sequence = pose_data[i-4:i+1]
        if is_emphasis_gesture(sequence):
            gestures['emphasis_points'].append(i)
    
    if gestures['emphasis_points']:
        # Calculate gesture metrics
        movements = [calculate_pose_movement(pose_data[i-1], pose_data[i])
                    for i in gestures['emphasis_points']]
        
        gestures['intensity'] = float(np.mean(movements))
        gestures['confidence'] = float(len(gestures['emphasis_points']) / len(pose_data))
    
    return gestures

def detect_interactive_gestures(pose_data):
    """
    Detect interactive or communicative gestures.
    
    Args:
        pose_data (list): Sequence of pose keypoints
        
    Returns:
        dict: Detected interactive gestures
    """
    gestures = {
        'pointing': False,
        'waving': False,
        'beckoning': False,
        'confidence': 0.0
    }
    
    if len(pose_data) < 10:
        return gestures
    
    # Track hand movements
    hand_tracks = []
    for i in range(1, len(pose_data)):
        prev_hands = [pose_data[i-1][7], pose_data[i-1][8]]  # Wrist points
        curr_hands = [pose_data[i][7], pose_data[i][8]]
        
        if None not in prev_hands and None not in curr_hands:
            movement = [
                (curr_hands[0][0] - prev_hands[0][0], curr_hands[0][1] - prev_hands[0][1]),
                (curr_hands[1][0] - prev_hands[1][0], curr_hands[1][1] - prev_hands[1][1])
            ]
            hand_tracks.append(movement)
    
    if hand_tracks:
        # Analyze movement patterns
        left_movements = np.array([track[0] for track in hand_tracks])
        right_movements = np.array([track[1] for track in hand_tracks])
        
        # Detect pointing (sustained directional movement)
        pointing_score = max(
            np.mean(np.abs(left_movements), axis=0).max(),
            np.mean(np.abs(right_movements), axis=0).max()
        )
        gestures['pointing'] = pointing_score > 30
        
        # Detect waving (periodic side-to-side movement)
        wave_score = max(
            np.std(left_movements[:,0]),
            np.std(right_movements[:,0])
        )
        gestures['waving'] = wave_score > 20
        
        # Detect beckoning (periodic forward-backward movement)
        beckon_score = max(
            np.std(left_movements[:,1]),
            np.std(right_movements[:,1])
        )
        gestures['beckoning'] = beckon_score > 20
        
        # Calculate overall confidence
        gesture_count = sum([gestures['pointing'], gestures['waving'], gestures['beckoning']])
        gestures['confidence'] = float(gesture_count / 3.0)
    
    return gestures

def analyze_text_content(text_regions, frame):
    """
    Analyze text content and its visual presentation.
    
    Args:
        text_regions (list): List of detected text regions
        frame (numpy.ndarray): Video frame
        
    Returns:
        dict: Text content analysis
    """
    if not text_regions:
        return None
    
    results = {
        'text_elements': [],
        'style': set(),
        'placement': [],
        'animated': False,
        'emphasis': []
    }
    
    frame_height, frame_width = frame.shape[:2]
    
    for region in text_regions:
        x, y, w, h = region['region']
        roi = frame[y:y+h, x:x+w]
        
        # Extract text content
        text = extract_text_content(frame, (x, y, w, h))
        if not text:
            continue
            
        # Analyze text properties
        properties = region['properties']
        
        # Determine text style
        style = analyze_style_attributes(properties)
        results['style'].update(style)
        
        # Analyze placement
        placement = analyze_text_placement(
            (x, y, w, h),
            (frame_width, frame_height)
        )
        
        # Analyze emphasis
        emphasis = analyze_text_emphasis(properties)
        if emphasis:
            results['emphasis'].append(emphasis)
        
        results['text_elements'].append({
            'content': text,
            'placement': placement,
            'style': list(style),
            'size': properties['size']
        })
    
    return results

def extract_text_content(frame, region):
    """Extract text content from frame region with improved error handling"""
    try:
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            # Try pytesseract first
            text = pytesseract.image_to_string(binary, config='--psm 7')
            if text.strip():
                return text.strip()
        except Exception as e:
            logger.warning(f"Pytesseract failed: {str(e)}, falling back to EasyOCR")
            try:
                # Fallback to EasyOCR
                reader = easyocr.Reader(['en'])
                results = reader.readtext(binary)
                text = ' '.join([result[1] for result in results])
                return text.strip() if text.strip() else None
            except Exception as e:
                logger.error(f"Text extraction error: {str(e)}")
                return None
                
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        return None

def analyze_text_placement(region, frame_size):
    """
    Analyze placement of text in frame.
    
    Args:
        region (tuple): (x, y, width, height) of text region
        frame_size (tuple): (width, height) of frame
        
    Returns:
        dict: Text placement analysis
    """
    x, y, w, h = region
    frame_width, frame_height = frame_size
    
    # Calculate relative position
    center_x = x + w/2
    center_y = y + h/2
    rel_x = center_x / frame_width
    rel_y = center_y / frame_height
    
    # Determine vertical position
    if rel_y < 0.33:
        vertical = "top"
    elif rel_y > 0.67:
        vertical = "bottom"
    else:
        vertical = "middle"
    
    # Determine horizontal position
    if rel_x < 0.33:
        horizontal = "left"
    elif rel_x > 0.67:
        horizontal = "right"
    else:
        horizontal = "center"
    
    return {
        'position': f"{vertical} {horizontal}",
        'relative_x': float(rel_x),
        'relative_y': float(rel_y),
        'area_ratio': float((w * h) / (frame_width * frame_height))
    }

def analyze_text_emphasis(properties):
    """
    Analyze emphasis characteristics of text.
    
    Args:
        properties (dict): Text region properties
        
    Returns:
        dict: Emphasis analysis
    """
    emphasis = {
        'type': None,
        'strength': 0.0,
        'characteristics': []
    }
    
    # Check size emphasis
    if properties['size'][0] > 50:  # Large text
        emphasis['characteristics'].append('large_size')
        emphasis['strength'] += 0.3
    
    # Check contrast emphasis
    if properties['contrast'] > 50:
        emphasis['characteristics'].append('high_contrast')
        emphasis['strength'] += 0.3
    
    # Check style emphasis
    if properties['style']['bold']:
        emphasis['characteristics'].append('bold')
        emphasis['strength'] += 0.2
    
    # Check color emphasis
    dominant_color = properties['color']['dominant']
    if np.mean(dominant_color) > 200 or np.mean(dominant_color) < 50:
        emphasis['characteristics'].append('color_emphasis')
        emphasis['strength'] += 0.2
    
    # Determine emphasis type
    if emphasis['strength'] > 0:
        if 'large_size' in emphasis['characteristics']:
            emphasis['type'] = 'size_emphasis'
        elif 'high_contrast' in emphasis['characteristics']:
            emphasis['type'] = 'contrast_emphasis'
        else:
            emphasis['type'] = 'style_emphasis'
    
    return emphasis if emphasis['strength'] > 0 else None

def analyze_scene_composition(frame):
    """Analyze compositional characteristics with proper dict merging"""
    try:
        composition_metrics = {
            'symmetry_score': 0.0,
            'rule_of_thirds_score': 0.0,
            'balance_score': 0.0,
            'depth_score': 0.0,
            'framing_score': 0.0
        }
        
        # Calculate metrics
        composition_metrics['symmetry_score'] = analyze_symmetry(frame)
        composition_metrics['rule_of_thirds_score'] = analyze_composition_thirds(frame)
        composition_metrics['balance_score'] = analyze_visual_balance(frame)
        composition_metrics['depth_score'] = analyze_depth(frame).get('depth_score', 0.0)
        composition_metrics['framing_score'] = analyze_framing(frame)
        
        return composition_metrics
        
    except Exception as e:
        logger.error(f"Error in scene composition analysis: {str(e)}")
        return {
            'symmetry_score': 0.0,
            'rule_of_thirds_score': 0.0,
            'balance_score': 0.0,
            'depth_score': 0.0,
            'framing_score': 0.0
        }

def detect_visual_points(frame):
    """
    Detect points of visual interest in frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        list: Points of visual interest
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    points = []
    
    if corners is not None:
        corners = np.int0(corners)
        height, width = frame.shape[:2]
        
        for corner in corners:
            x, y = corner.ravel()
            # Convert to relative coordinates
            points.append({
                'x': float(x / width),
                'y': float(y / height),
                'strength': float(gray[y, x] / 255.0)
            })
    
    return points

def analyze_thirds_points(frame):
    """
    Analyze rule of thirds alignment.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        dict: Rule of thirds analysis
    """
    height, width = frame.shape[:2]
    
    # Calculate thirds points
    h_third = height // 3
    w_third = width // 3
    
    thirds_points = [
        (w_third, h_third),
        (2*w_third, h_third),
        (w_third, 2*h_third),
        (2*w_third, 2*h_third)
    ]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate feature density around thirds points
    point_scores = []
    for x, y in thirds_points:
        # Analyze region around point
        region = gray[y-20:y+20, x-20:x+20]
        score = np.std(region) / 255.0
        point_scores.append(float(score))
    
    return {
        'alignment_score': float(np.mean(point_scores)),
        'point_scores': point_scores,
        'strongest_point': thirds_points[np.argmax(point_scores)]
    }

def calculate_symmetry_score(frame):
    """
    Calculate symmetry score for frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        float: Symmetry score
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Split image into left and right halves
    height, width = gray.shape
    mid = width // 2
    left_half = gray[:, :mid]
    right_half = gray[:, mid:mid*2]
    
    # Flip right half for comparison
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize if necessary
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate difference
    diff = cv2.absdiff(left_half, right_half_flipped)
    symmetry_score = 1.0 - (np.mean(diff) / 255.0)
    
    return float(symmetry_score)

def calculate_balance_score(frame):
    """
    Calculate visual balance score for frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        float: Balance score
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Split frame into quadrants
    top_left = np.mean(gray[:height//2, :width//2])
    top_right = np.mean(gray[:height//2, width//2:])
    bottom_left = np.mean(gray[height//2:, :width//2])
    bottom_right = np.mean(gray[height//2:, width//2:])
    
    # Calculate balance scores
    horizontal_balance = 1.0 - abs((top_left + bottom_left) - (top_right + bottom_right)) / 510
    vertical_balance = 1.0 - abs((top_left + top_right) - (bottom_left + bottom_right)) / 510
    
    # Combine scores
    balance_score = (horizontal_balance + vertical_balance) / 2
    
    return float(balance_score)

def detect_text_enhanced(frame):
    """Detect text regions with improved error handling"""
    try:
        if frame is None or frame.size == 0:
            return []
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        text_regions = []
        for contour in contours:
            if contour is None or len(contour) < 3:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter potential text regions based on aspect ratio and size
            if w > 0 and h > 0:  # Validate dimensions
                aspect_ratio = w / float(h)
                if 0.1 < aspect_ratio < 15 and w > 20 and h > 8:
                    roi = frame[y:y+h, x:x+w]
                    if roi is not None and roi.size > 0:
                        # Analyze text properties
                        properties = analyze_text_properties(roi)
                        if properties:
                            text_regions.append({
                                'region': (x, y, w, h),
                                'properties': properties
                            })
        
        return text_regions
        
    except Exception as e:
        logger.warning(f"Error in text detection: {str(e)}")
        return []


# Scene Analysis Functions
def analyze_location_features(frame, location_types):
    """Analyze frame for location features"""
    scores = {}
    for location, attributes in location_types.items():
        score = 0
        # Check for visual features
        for feature in attributes['features']:
            feature_score = detect_feature_presence(frame, feature)
            score += feature_score
            
        # Check contextual elements
        for context in attributes['contexts']:
            context_score = detect_context_presence(frame, context)
            score += context_score
            
        scores[location] = score / (len(attributes['features']) + len(attributes['contexts']))
    
    return scores

def analyze_location_context(frame, location_types):
    """Analyze contextual elements of location"""
    context_scores = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    for location, attributes in location_types.items():
        context_score = 0
        # Analyze lighting patterns
        lighting = analyze_lighting_pattern(frame)
        if lighting in attributes['contexts']:
            context_score += 1
            
        # Analyze spatial arrangement
        spatial = analyze_spatial_arrangement(edges)
        if spatial in attributes['contexts']:
            context_score += 1
            
        context_scores[location] = context_score / len(attributes['contexts'])
    
    return context_scores

def determine_primary_location(location_scores, context_scores):
    """Determine primary location based on scores"""
    combined_scores = {}
    
    # Combine location and context scores with weights
    for location in location_scores:
        combined_scores[location] = (
            location_scores[location] * 0.7 + 
            context_scores.get(location, 0) * 0.3
        )
    
    # Get location with highest score
    primary_location = max(combined_scores.items(), key=lambda x: x[1])
    return primary_location[0] if primary_location[1] > 0.3 else "undefined"

def determine_primary_activity(performance_types):
    """Determine primary activity from detected types"""
    if not performance_types:
        return "unknown"
        
    # Count occurrences
    type_counts = Counter(performance_types)
    
    # Get most common type
    if type_counts:
        return type_counts.most_common(1)[0][0]
    return "unknown"

# Audio Analysis Functions
def detect_trap_rhythm_fixed(y, sr, beat_frames, n_fft, hop_length):
    """Detect trap music rhythm patterns with fixed parameters"""
    try:
        if len(beat_frames) < 2:
            return False
            
        # Get tempo and beat intervals
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        intervals = np.diff(beat_times)
        
        if intervals.size == 0:
            return False
        
        # Check for typical trap characteristics
        has_808 = detect_808_fixed(y, sr, n_fft, hop_length)
        has_hihat_rolls = detect_hihat_rolls_fixed(y, sr, n_fft, hop_length)
        
        # Get tempo with fixed parameters
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        has_trap_tempo = 130 <= tempo <= 160
        
        return (has_808 and has_hihat_rolls and has_trap_tempo)
        
    except Exception as e:
        logger.warning(f"Error in trap rhythm detection: {str(e)}")
        return False

def detect_drill_rhythm_fixed(y, sr, beat_frames, n_fft, hop_length):
    """Detect drill music rhythm patterns with fixed parameters"""
    try:
        if len(beat_frames) < 2:
            return False
            
        # Analyze beat pattern
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        intervals = np.diff(beat_times)
        
        if intervals.size == 0:
            return False
            
        # Check for drill characteristics
        has_sliding_bass = detect_sliding_bass_fixed(y, sr, n_fft, hop_length)
        has_drill_pattern = detect_drill_pattern_fixed(y, sr, n_fft, hop_length)
        
        # Get tempo with fixed parameters
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        return (has_sliding_bass and has_drill_pattern and 140 <= tempo <= 150)
        
    except Exception as e:
        logger.warning(f"Error in drill rhythm detection: {str(e)}")
        return False

def detect_boom_bap_rhythm_fixed(y, sr, beat_frames, n_fft, hop_length):
    """Detect boom bap rhythm patterns with fixed parameters"""
    try:
        if len(beat_frames) < 2:
            return False
            
        # Analyze drum pattern
        has_boom = detect_kick_pattern_fixed(y, sr, n_fft, hop_length)
        has_bap = detect_snare_pattern_fixed(y, sr, n_fft, hop_length)
        
        # Check timing
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        has_swing = detect_swing_timing(beat_times)  # This function already handles dimension check
        
        return (has_boom and has_bap and has_swing)
        
    except Exception as e:
        logger.warning(f"Error in boom bap rhythm detection: {str(e)}")
        return False

def detect_dance_rhythm_fixed(y, sr, beat_frames, n_fft, hop_length):
    """Detect dance music rhythm patterns with fixed parameters"""
    try:
        if len(beat_frames) < 2:
            return False
            
        # Analyze beat consistency
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        intervals = np.diff(beat_times)
        
        if intervals.size == 0:
            return False
            
        is_regular = np.std(intervals) < 0.05
        
        # Check characteristics
        has_four_on_floor = detect_four_on_floor_fixed(y, sr, n_fft, hop_length)
        
        # Get tempo with fixed parameters
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        return (is_regular and has_four_on_floor and 120 <= tempo <= 130)
        
    except Exception as e:
        logger.warning(f"Error in dance rhythm detection: {str(e)}")
        return False
# Pose Analysis Functions
def calculate_arm_spread(pose_data):
    """Calculate arm spread from pose data"""
    if not pose_data or len(pose_data) < 14:  # Need shoulder and wrist points
        return 0
        
    # Get shoulder and wrist points
    left_shoulder = pose_data[5]
    right_shoulder = pose_data[6]
    left_wrist = pose_data[9]
    right_wrist = pose_data[10]
    
    if None in [left_shoulder, right_shoulder, left_wrist, right_wrist]:
        return 0
        
    # Calculate spread ratio
    shoulder_width = np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder))
    arm_width = np.linalg.norm(np.array(right_wrist) - np.array(left_wrist))
    
    return min(1.0, arm_width / (shoulder_width * 2.5))

# Helper functions for default values
def get_default_environment():
    """Get default environment values"""
    return {
        'lighting': get_default_lighting(),
        'space': get_default_space(),
        'atmosphere': get_default_atmosphere()
    }

def get_default_engagement_analysis():
    """Return default engagement analysis structure"""
    return {
        'visual_strategies': {'strategies': [], 'consistency': 0.0, 'details': {}},
        'audio_strategies': {'strategies': [], 'consistency': 0.0, 'details': {}},
        'performance_strategies': {'strategies': [], 'consistency': 0.0, 'details': {}},
        'content_strategies': {'strategies': [], 'consistency': 0.0, 'details': {}},
        'engagement_metrics': {
            'consistency': 0.0,
            'variety': 0.0,
            'effectiveness': 0.0
        },
        'primary_focus': "unknown",
        'strategies': []
    }


def get_default_sync_analysis():
    """Get default sync analysis results"""
    return {
        'sync_score': 0.0,
        'description': "No sync data available",
        'details': {
            'strong_sync_points': 0,
            'weak_sync_points': 0
        }
    }

def summarize_lighting_states(lighting_states):
    """Summarize lighting characteristics across frames"""
    try:
        if not lighting_states:
            return {
                'brightness_level': 0.0,
                'contrast_level': 0.0,
                'lighting_type': "unknown",
                'color_temperature': "unknown"
            }
            
        # Average brightness and contrast
        brightness_levels = [state.get('brightness_level', 0) for state in lighting_states]
        contrast_levels = [state.get('contrast_level', 0) for state in lighting_states]
        
        # Count lighting types and temperatures
        lighting_types = Counter([state.get('type', 'unknown') for state in lighting_states])
        color_temps = Counter([state.get('temperature', 'unknown') for state in lighting_states])
        
        return {
            'brightness_level': float(np.mean(brightness_levels)),
            'contrast_level': float(np.mean(contrast_levels)),
            'lighting_type': lighting_types.most_common(1)[0][0],
            'color_temperature': color_temps.most_common(1)[0][0]
        }
        
    except Exception as e:
        logger.debug(f"Error summarizing lighting states: {str(e)}")
        return {
            'brightness_level': 0.0,
            'contrast_level': 0.0,
            'lighting_type': "unknown",
            'color_temperature': "unknown"
        }

def determine_setting_type(visual_elements):
    """Determine setting type from visual elements"""
    try:
        if not visual_elements:
            return "unknown"
            
        # Define setting categories
        setting_categories = {
            'studio': ['microphone', 'audio interface', 'studio monitors', 'sound panels',
                      'recording equipment', 'mixing desk', 'studio lights'],
            'home': ['bedroom', 'living room', 'furniture', 'personal items', 
                    'desk', 'bed', 'house interior', 'residential'],
            'outdoor': ['street', 'buildings', 'nature', 'sky', 'urban', 
                       'trees', 'outdoor lighting', 'city'],
            'venue': ['stage', 'performance space', 'audience area', 'stage lights',
                     'concert', 'club', 'performance venue'],
            'bathroom': ['bathroom mirror', 'sink', 'bathroom lighting', 'tiles',
                        'bathroom fixtures', 'mirror selfie'],
            'car': ['car interior', 'vehicle', 'dashboard', 'car seat',
                    'driving', 'automobile']
        }
        
        # Count elements in each category
        category_scores = {category: 0 for category in setting_categories}
        
        for element in visual_elements:
            element_lower = element.lower()
            for category, indicators in setting_categories.items():
                if any(indicator in element_lower for indicator in indicators):
                    category_scores[category] += 1
                    
        # Get category with highest score
        if any(category_scores.values()):
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "unknown"
        
    except Exception as e:
        logger.debug(f"Error determining setting type: {str(e)}")
        return "unknown"

def get_default_scene_analysis():
    """Return default scene analysis structure"""
    return {
        'technical_data': {
            'primary_location': "unknown",
            'lighting_summary': "unknown",
            'setting_type': "unknown",
            'scene_changes': 0
        },
        'composition': {
            'symmetry_score': 0.0,
            'rule_of_thirds_score': 0.0,
            'balance_score': 0.0,
            'depth_score': 0.0
        },
        'lighting': {
            'brightness_level': 0.0,
            'contrast_level': 0.0,
            'lighting_type': "unknown",
            'color_temperature': "unknown"
        },
        'setting': {
            'environment_type': "unknown",
            'setting_quality': "unknown",
            'visual_elements': []
        },
        'transitions': [],
        'description': "No scene data available"
    }

def get_default_performance_analysis():
    return {
        'direct_address': False,
        'gestures': False,
        'emotional_range': 0.0
    }

def get_default_performance_metrics():
    """Return default performance metrics"""
    return {
        'direct_address': False,
        'gestures': False,
        'emotional_range': 0.0,
        'techniques': []
    }

def get_default_lighting():
    """Get default lighting values"""
    return {
        'brightness_level': 0.0,
        'contrast': 0.0,
        'primary_direction': 0,
        'lighting_type': "unknown",
        'lighting_quality': "unknown"
    }


def get_default_space():
    """Get default space analysis values"""
    return {
        'depth_cues': False,
        'edge_complexity': 0.0,
        'spatial_distribution': {
            'top': 0.0,
            'middle': 0.0,
            'bottom': 0.0
        },
        'perspective_strength': 0.0
    }


def get_default_atmosphere():
    """Get default atmosphere values"""
    return {
        'color_temperature': "neutral",
        'light_quality': "unknown",
        'atmospheric_depth': "shallow",
        'mood': "neutral"
    }

def get_default_effects():
    """Get default audio effects values"""
    return {
        'autotune': False,
        'reverb': 0.0,
        'delay': False,
        'distortion': 0.0,
        'filters': []
    }
def analyze_atmospheric_environment(frame):
    """Analyze atmospheric qualities of the environment"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Analyze color temperature
        color_temp = analyze_color_temperature(frame)
        
        # Analyze light quality
        light_quality = analyze_light_quality(lab)
        
        # Analyze atmospheric depth
        depth = analyze_atmospheric_depth(hsv)
        
        # Determine mood
        mood = determine_atmospheric_mood(color_temp, light_quality, depth)
        
        return {
            'color_temperature': color_temp,
            'light_quality': light_quality,
            'atmospheric_depth': depth,
            'mood': mood
        }
        
    except Exception as e:
        logger.warning(f"Error in atmospheric analysis: {str(e)}")
        return get_default_atmosphere()

def analyze_color_temperature(frame):
    """Analyze color temperature of the scene"""
    try:
        b, g, r = cv2.split(frame)
        rb_ratio = float(np.mean(r)) / (float(np.mean(b)) + 1e-6)
        
        if rb_ratio > 1.2:
            return 'warm'
        elif rb_ratio < 0.8:
            return 'cool'
        else:
            return 'neutral'
            
    except Exception:
        return 'neutral'

def analyze_light_quality(lab):
    """Analyze quality of lighting"""
    try:
        l_channel = lab[:,:,0]
        mean_brightness = float(np.mean(l_channel))
        std_brightness = float(np.std(l_channel))
        
        if std_brightness < 20:
            return 'flat'
        elif std_brightness > 50:
            return 'contrasty'
        else:
            return 'balanced'
            
    except Exception:
        return 'unknown'

def analyze_atmospheric_depth(hsv):
    """Analyze atmospheric depth cues"""
    try:
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Calculate depth metrics
        sat_gradient = float(np.mean(np.gradient(saturation)))
        val_gradient = float(np.mean(np.gradient(value)))
        
        depth_score = (abs(sat_gradient) + abs(val_gradient)) / 2
        
        if depth_score > 0.5:
            return 'deep'
        elif depth_score > 0.2:
            return 'moderate'
        else:
            return 'shallow'
            
    except Exception:
        return 'shallow'

def determine_atmospheric_mood(color_temp, light_quality, depth):
    """Determine atmospheric mood based on visual characteristics"""
    try:
        moods = {
            ('warm', 'balanced', 'moderate'): 'inviting',
            ('cool', 'contrasty', 'deep'): 'dramatic',
            ('neutral', 'flat', 'shallow'): 'neutral',
            ('warm', 'contrasty', 'deep'): 'intense',
            ('cool', 'balanced', 'moderate'): 'calm'
        }
        
        return moods.get((color_temp, light_quality, depth), 'neutral')
        
    except Exception:
        return 'neutral'

def get_default_atmosphere():
    """Return default atmospheric values"""
    return {
        'color_temperature': 'neutral',
        'light_quality': 'balanced',
        'atmospheric_depth': 'moderate',
        'mood': 'neutral'
    }

def calculate_shoulder_openness(pose_data):
    """Calculate shoulder openness from pose data"""
    if not pose_data or len(pose_data) < 7:  # Need shoulder points
        return 0
        
    # Get shoulder points
    left_shoulder = pose_data[5]
    right_shoulder = pose_data[6]
    
    if None in [left_shoulder, right_shoulder]:
        return 0
        
    # Calculate shoulder angle relative to horizontal
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    
    # Score openness (0 = shoulders very slumped, 1 = shoulders level)
    return 1.0 - min(1.0, angle / 45.0)

def detect_rhythm_patterns(y, sr):
    """Wrapper for detect_rhythm_patterns_fixed to maintain compatibility"""
    try:
        # Use standard parameters
        n_fft = 2048
        hop_length = 512
        return detect_rhythm_patterns_fixed(y, sr, n_fft, hop_length)
    except Exception as e:
        logger.error(f"Error in rhythm patterns wrapper: {str(e)}")
        return {
            'trap': False,
            'drill': False,
            'boom_bap': False,
            'dance': False
        }

def is_rhythmic_gesture(sequence):
    """Simple rhythmic gesture detection"""
    try:
        if len(sequence) < 5:
            return False
            
        # Calculate basic movement
        movements = []
        for i in range(1, len(sequence)):
            if sequence[i] is not None and sequence[i-1] is not None:
                movement = np.mean([abs(sequence[i][j] - sequence[i-1][j]) 
                                  for j in range(min(len(sequence[i]), len(sequence[i-1])))])
                movements.append(movement)
                
        if not movements:
            return False
            
        # Check for rhythm
        return np.std(movements) < np.mean(movements) * 0.5
        
    except Exception as e:
        logger.warning(f"Error in rhythmic gesture detection: {str(e)}")
        return False

def detect_duet_layout(frame):
    """Basic duet layout detection"""
    try:
        if frame is None or len(frame.shape) != 3:
            return False
            
        height, width = frame.shape[:2]
        left_half = frame[:, :width//2]
        right_half = frame[:, width//2:]
        
        return np.mean(np.abs(left_half - right_half)) > 50
        
    except Exception as e:
        logger.warning(f"Error in duet layout detection: {str(e)}")
        return False

def detect_popular_effects(frame):
    """Wrapper for existing effects detection"""
    try:
        return detect_frame_effects(frame)
    except Exception as e:
        logger.warning(f"Error detecting effects: {str(e)}")
        return []

def analyze_emotion_detection(frame):
    """Single frame emotion analysis with improved color handling and error checking"""
    try:
        if not isinstance(frame, np.ndarray):
            logger.debug("Invalid frame type in emotion detection")
            return None

        # Store original frame dimensions
        original_height, original_width = frame.shape[:2]

        # Frame should be RGB, convert to BGR for OpenCV
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.debug(f"Color conversion error: {str(e)}")
            return None

        # Convert to grayscale for face detection
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.debug(f"Grayscale conversion error: {str(e)}")
            return None

        # Initialize face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                logger.debug("Failed to load face cascade classifier")
                return None
        except Exception as e:
            logger.debug(f"Face cascade initialization error: {str(e)}")
            return None

        # Detect faces with different parameters
        faces = []
        try:
            scale_factors = [1.1, 1.2, 1.3]
            min_neighbors_values = [3, 4, 5]
            
            for scale in scale_factors:
                for min_neighbors in min_neighbors_values:
                    detected = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30)
                    )
                    if len(detected) > 0:
                        faces.extend(detected)
                        break
                if faces:
                    break
        except Exception as e:
            logger.debug(f"Face detection error: {str(e)}")
            return None

        if not faces:
            logger.debug("No faces detected")
            return None

        try:
            # Process largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            # Validate face region coordinates
            if x < 0 or y < 0 or x + w > original_width or y + h > original_height:
                logger.debug("Invalid face coordinates")
                return None

            # Add padding to face region
            padding = int(min(w, h) * 0.1)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(original_width, x + w + padding)
            y_end = min(original_height, y + h + padding)

            # Extract face ROI
            face_roi = frame_bgr[y_start:y_end, x_start:x_end]

            # Validate ROI
            if face_roi.shape[0] < 48 or face_roi.shape[1] < 48:
                logger.debug("Face ROI too small")
                return None

            # Analyze emotions with timeout protection
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        DeepFace.analyze,
                        img_path=face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    emotions = future.result(timeout=2)
                    
                    # Process emotion results
                    if isinstance(emotions, list) and emotions:
                        emotion_data = emotions[0].get('emotion', {})
                    elif isinstance(emotions, dict):
                        emotion_data = emotions.get('emotion', {})
                    else:
                        logger.debug("Invalid emotion analysis result")
                        return None

                    # Calculate confidence
                    if emotion_data:
                        confidence = max(emotion_data.values())
                        if confidence < 0.2:  # Minimum confidence threshold
                            logger.debug("Low emotion confidence")
                            return None

                        return {
                            'emotions': emotion_data,
                            'position': (x_start, y_start, x_end - x_start, y_end - y_start),
                            'confidence': float(confidence),
                            'frame_size': {
                                'width': original_width,
                                'height': original_height
                            }
                        }

            except Exception as e:
                logger.debug(f"Emotion analysis error: {str(e)}")
                return None

        except Exception as e:
            logger.debug(f"Face processing error: {str(e)}")
            return None

    except Exception as e:
        logger.warning(f"Error in emotion detection: {str(e)}")
        return None

def calculate_shoulder_tension(pose_data):
    """Calculate shoulder tension from pose data"""
    if not pose_data or len(pose_data) < 7:
        return 0
        
    # Get shoulder and neck points
    neck = pose_data[1]
    left_shoulder = pose_data[5]
    right_shoulder = pose_data[6]
    
    if None in [neck, left_shoulder, right_shoulder]:
        return 0
        
    # Calculate shoulder elevation relative to neck
    left_elevation = neck[1] - left_shoulder[1]
    right_elevation = neck[1] - right_shoulder[1]
    
    # Normalize and score tension
    max_elevation = 30  # pixels
    tension = (left_elevation + right_elevation) / (2 * max_elevation)
    
    return max(0, min(1, tension))

def analyze_arm_position(pose_data):
    """Analyze arm position relative to body"""
    if not pose_data or len(pose_data) < 11:
        return 'unknown'
        
    # Get arm points
    shoulders = np.array([pose_data[5], pose_data[6]])
    elbows = np.array([pose_data[7], pose_data[8]])
    wrists = np.array([pose_data[9], pose_data[10]])
    
    if np.any(shoulders is None) or np.any(elbows is None) or np.any(wrists is None):
        return 'unknown'
        
    # Calculate average arm extension
    shoulder_to_elbow = np.linalg.norm(elbows - shoulders, axis=1)
    elbow_to_wrist = np.linalg.norm(wrists - elbows, axis=1)
    arm_extension = np.mean(shoulder_to_elbow + elbow_to_wrist)
    
    # Classify position
    if arm_extension > 100:  # pixels
        return 'open'
    elif arm_extension < 50:
        return 'closed'
    return 'neutral'

# Audio Effects Functions
def analyze_vocal_effects(y, sr):
    """Analyze vocal effects and processing"""
    effects = {}
    
    # Detect autotune/pitch correction
    effects['autotune'] = detect_autotune(y, sr)
    
    # Measure reverb amount
    effects['reverb'] = measure_reverb(y)
    
    # Detect delay/echo
    effects['delay'] = detect_delay(y, sr)
    
    # Analyze other effects
    effects['distortion'] = measure_distortion(y)
    effects['filters'] = detect_vocal_filters(y, sr)
    
    return effects

def detect_autotune(y, sr):
    """Detect presence of autotune/pitch correction"""
    # Extract pitch
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    # Analyze pitch stability
    pitch_std = np.std(pitches[magnitudes > np.max(magnitudes)*0.1])
    
    # Higher stability suggests pitch correction
    return pitch_std < 1.0

def measure_reverb(y):
    """Measure amount of reverb in audio"""
    # Use scipy.signal instead of removed librosa function
    window = scipy.signal.windows.hann(2048)
    reverb = scipy.signal.convolve(y, window, mode='full')
    
    # Measure decay time
    decay = np.mean(np.abs(reverb[len(y):]))
    return min(1.0, decay * 10)

def measure_distortion(y):
    """Estimate amount of distortion in audio signal"""
    # Calculate zero crossing rate as a distortion indicator
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    
    # High zero crossing rate can indicate distortion
    # Normalize to 0-1 range
    distortion = (zcr_mean - 0.1) / 0.4  # 0.1 to 0.5 is typical range
    return max(0, min(1, distortion))

def measure_compression(y):
    """Estimate amount of dynamic range compression"""
    # Calculate crest factor (peak to RMS ratio)
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    crest_factor = peak / (rms + 1e-8)
    
    # Convert to compression estimate (0-1 range)
    compression = 1.0 - (np.log10(crest_factor) / np.log10(20))
    return max(0, min(1, compression))

def detect_delay(y, sr):
    """Detect presence of delay/echo effects"""
    # Calculate autocorrelation
    correlation = correlate(y, y)
    correlation = correlation[len(y)-1:]
    
    # Find peaks in correlation
    peaks = scipy.signal.find_peaks(correlation, distance=sr//4)[0]
    
    return len(peaks) > 2  # Multiple strong peaks suggest delay

def detect_vocal_filters(y, sr):
    """Detect frequency filters applied to vocals"""
    # Calculate spectrogram
    S = np.abs(librosa.stft(y))
    
    # Analyze frequency distribution
    freq_profile = np.mean(S, axis=1)
    
    filters = []
    if np.mean(freq_profile[:sr//4]) < 0.1:
        filters.append('highpass')
    if np.mean(freq_profile[3*sr//4:]) < 0.1:
        filters.append('lowpass')
        
    return filters


# Scene Analysis Functions
def detect_scene_transition(prev_frame, curr_frame):
    """Detect and classify scene transitions with validation"""
    try:
        # Validate both frames
        if not validate_frame(prev_frame) or not validate_frame(curr_frame):
            return None
            
        # Calculate frame difference (frames already in RGB format)
        diff = cv2.absdiff(prev_frame, curr_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > 100:
            # Check for fade
            prev_brightness = np.mean(prev_frame)
            curr_brightness = np.mean(curr_frame)
            if abs(prev_brightness - curr_brightness) > 50:
                return 'fade'
            return 'cut'
            
        elif mean_diff > 50:
            # Check for wipe/slide
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_diff, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100)
            if lines is not None and len(lines) > 5:
                return 'wipe'
                
        return None
        
    except Exception as e:
        logger.warning(f"Error detecting scene transition: {str(e)}")
        return None

def analyze_environment(frame):
    try:
        if not isinstance(frame, np.ndarray):
            return get_default_environment()
            
        context = get_default_environment()  # Start with default
        
        # Convert color spaces safely
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        except Exception:
            return get_default_environment()
        
        # Update each component safely
        try:
            lighting_result = analyze_lighting_environment(frame)
            context['lighting'].clear()
            context['lighting'].update(lighting_result)
        except Exception as e:
            logger.warning(f"Lighting analysis error: {str(e)}")
            
        try:
            space_result = analyze_spatial_environment(frame)
            context['space'].clear()
            context['space'].update(space_result)
        except Exception as e:
            logger.warning(f"Spatial analysis error: {str(e)}")
            
        try:
            atmosphere_result = analyze_atmospheric_environment(frame)
            context['atmosphere'].clear()
            context['atmosphere'].update(atmosphere_result)
        except Exception as e:
            logger.warning(f"Atmosphere analysis error: {str(e)}")
        
        return context
        
    except Exception as e:
        logger.warning(f"Error in environment analysis: {str(e)}")
        return get_default_environment()

def analyze_primary_location(locations):
    """Determine primary location from sequence"""
    if not locations:
        return None
        
    # Count location frequencies
    location_counter = Counter(locations)
    
    # Get most common location
    primary = location_counter.most_common(1)[0]
    
    # Check if location is consistent enough
    if primary[1] / len(locations) > 0.3:
        return primary[0]
    return "multiple locations"

def summarize_lighting(lighting_states):
    """Summarize lighting conditions across video"""
    if not lighting_states:
        return None
        
    # Count lighting types
    types = Counter([state['type'] for state in lighting_states])
    intensities = Counter([state['intensity'] for state in lighting_states])
    
    # Get dominant characteristics
    main_type = types.most_common(1)[0][0]
    main_intensity = intensities.most_common(1)[0][0]
    
    return f"{main_intensity} {main_type} lighting"

def summarize_environment(contexts):
    """Summarize environmental context with fixed dictionary handling"""
    if not contexts:
        return None

    try:
        # Extract atmosphere types safely
        atmospheres = []
        for context in contexts:
            if isinstance(context, dict):
                atm = context.get('atmosphere', {})
                if isinstance(atm, dict):
                    mood = atm.get('mood', 'neutral')
                    if isinstance(mood, str):
                        atmospheres.append(mood)

        # Count atmosphere occurrences
        if atmospheres:
            atmosphere_counter = Counter(atmospheres)
            return atmosphere_counter.most_common(1)[0][0]
        
        return "neutral"
        
    except Exception as e:
        logger.warning(f"Error in environment summarization: {str(e)}")
        return "neutral"

def analyze_scene_dynamics(transitions):
    """Analyze scene transition patterns"""
    if not transitions:
        return "static shot"
        
    # Calculate transition frequency
    avg_duration = len(transitions) / transitions[-1]['frame']
    
    if avg_duration > 0.5:
        return "rapid cuts"
    elif avg_duration > 0.2:
        return "moderate pacing"
    return "slow pacing"

def analyze_text_properties(roi):
    """Analyze text properties with better size handling"""
    try:
        # Skip analysis if ROI is too small
        if roi.shape[0] < 10 or roi.shape[1] < 40:
            return None
            
        # Convert color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        properties = {
            'size': roi.shape[:2],
            'contrast': float(np.std(gray)),
            'color': {
                'dominant': get_dominant_color(roi),
                'background': get_background_color(roi)
            },
            'style': {
                'bold': is_bold_text(gray),
                'italic': detect_italic(gray),
                'animated': False
            }
        }
        
        return properties
        
    except Exception as e:
        logger.warning(f"Error analyzing text properties: {str(e)}")
        return None

def get_dominant_emotion(emotion_scores):
    """Get dominant emotion from emotion scores"""
    try:
        if not emotion_scores:
            return ""
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    except Exception:
        return ""

# Color and Visual Analysis Helpers
def get_dominant_color(roi):
    """Get dominant color from a region of interest"""
    # Reshape ROI for k-means
    pixels = roi.reshape(-1, 3)
    
    # Use k-means to find dominant colors
    kmeans = KMeans(n_clusters=1, n_init=1)
    kmeans.fit(pixels)
    
    return kmeans.cluster_centers_[0]

def get_background_color(roi):
    """Get background color by analyzing edges"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    mask = cv2.dilate(edges, None)
    
    # Get background pixels (non-edge areas)
    background = roi.copy()
    background[mask > 0] = [0, 0, 0]
    
    # Get mean color of non-edge areas
    if np.sum(mask == 0) > 0:
        return np.mean(background[mask == 0], axis=0)
    return np.array([0, 0, 0])

def is_bold_text(gray):
    """Detect if text is bold based on thickness"""
    # Calculate text thickness using gradient magnitude
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Threshold for bold text
    return np.mean(gradient_magnitude) > 50

def detect_italic(gray):
    """Detect if text is italic based on angle"""
    # Use Hough transform to detect lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
    
    if lines is not None:
        # Calculate average angle of detected lines
        angles = [line[0][1] for line in lines]
        mean_angle = np.mean(angles)
        return abs(mean_angle - np.pi/2) > 0.1
    return False

def detect_props(frame):
    """Detect music-related props with improved error handling"""
    try:
        if not validate_frame(frame):
            return set()

        # List of common music props to check
        props = [
            "microphone", "guitar", "piano", "drum",
            "headphones", "turntable", "synthesizer",
            "laptop", "speakers", "amplifier"
        ]
        
        detected_props = set()

        # Convert frame for CLIP analysis
        try:
            image = Image.fromarray(frame)  # Frame should already be RGB from validate_frame
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            text_inputs = clip_processor(text=props, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                
                # Validate tensor shapes before calculating similarities
                if image_features.shape[0] > 0 and text_features.shape[0] > 0:
                    similarities = F.cosine_similarity(
                        image_features.unsqueeze(1),
                        text_features.unsqueeze(0)
                    )
                    
                    # Check if similarities tensor has the expected shape
                    if similarities.shape[1] == len(props):
                        for i, score in enumerate(similarities[0]):
                            if score > 0.25:  # Confidence threshold
                                detected_props.add(props[i])

        except Exception as e:
            logger.warning(f"Error in CLIP analysis: {str(e)}")
            return set()

        return detected_props

    except Exception as e:
        logger.warning(f"Error detecting props: {str(e)}")
        return set()

def analyze_location_appeal(frame):
    """Analyze location appeal with validation"""
    try:
        if not validate_frame(frame):
            return None
            
        aesthetics = analyze_visual_aesthetics(frame)
        features = detect_location_features(frame)
        
        if (aesthetics and 'composition' in aesthetics and 
            aesthetics['composition'].get('score', 0) > 0.7 and 
            len(features) > 2):
            return "visually appealing location"
        elif len(features) > 3:
            return "unique location"
        return None
        
    except Exception as e:
        logger.warning(f"Error analyzing location appeal: {str(e)}")
        return None

def analyze_frame_features(frame):
    """Extract comprehensive frame features with validation"""
    try:
        if not validate_frame(frame):
            return None
            
        features = {
            'composition': analyze_composition(frame),
            'colors': analyze_color_distribution(frame),
            'lighting': analyze_detailed_lighting(frame),
            'texture': analyze_texture(frame),
            'motion_blur': detect_motion_blur(frame)
        }
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting frame features: {str(e)}")
        return None


def analyze_video_frames(frames):
    """
    Analyze video frames with improved validation and error handling.
    
    Args:
        frames (list): List of video frames
        
    Returns:
        dict: Analysis results for emotions, camera work, and scene composition
    """
    try:
        if not frames:
            logger.warning("No frames provided for analysis")
            return {
                'emotions': ({}, "NA"),
                'camera': "static",
                'scene': {
                    'description': "No scene data available",
                    'technical_data': {
                        'primary_location': "unknown",
                        'lighting_summary': "unknown",
                        'setting_type': "unknown",
                        'scene_changes': 0
                    }
                }
            }
        
        # Process each analysis type with proper validation
        results = {}
        
        # Emotion analysis - Fix: Pass analysis function correctly
        try:
            emotion_frames = process_frames_batch_fixed(frames, analyze_emotion_detection)
            results['emotions'] = emotion_analysis(emotion_frames)
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            results['emotions'] = ({}, "NA")
        
        # Camera analysis - Fix: Pass analysis function correctly
        try:
            camera_frames = process_frames_batch_fixed(frames, analyze_camera_movement)
            results['camera'] = camera_analysis(camera_frames)
        except Exception as e:
            logger.error(f"Error in camera analysis: {str(e)}")
            results['camera'] = "static"
        
        # Scene analysis - Fix: Pass analysis function correctly
        try:
            scene_frames = process_frames_batch_fixed(frames, analyze_scene_composition)
            results['scene'] = scene_analysis(scene_frames)
        except Exception as e:
            logger.error(f"Error in scene analysis: {str(e)}")
            results['scene'] = {
                'description': "Error analyzing scene",
                'technical_data': {
                    'primary_location': "unknown",
                    'lighting_summary': "unknown",
                    'setting_type': "unknown",
                    'scene_changes': 0
                }
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in video frame analysis: {str(e)}")
        return {
            'emotions': ({}, "NA"),
            'camera': "static",
            'scene': {
                'description': "Error in analysis",
                'technical_data': {
                    'primary_location': "unknown",
                    'lighting_summary': "unknown",
                    'setting_type': "unknown",
                    'scene_changes': 0
                }
            }
        }
def detect_location_features(frame):
    """Detect distinctive location features"""
    features = []
    
    # Analyze frame regions
    height, width = frame.shape[:2]
    regions = {
        'top': frame[:height//3],
        'middle': frame[height//3:2*height//3],
        'bottom': frame[2*height//3:]
    }
    
    # Analyze each region for features
    for region_name, region in regions.items():
        # Detect lines and edges
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            if len(lines) > 10:
                features.append('structural_elements')
        
        # Analyze textures
        lbp = local_binary_pattern(gray, 8, 1, 'uniform')
        if np.std(lbp) > 30:
            features.append('textured_surfaces')
        
        # Check for distinctive colors
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        if np.std(hsv[:,:,0]) > 40:
            features.append('color_variety')
    
    return list(set(features))

def analyze_texture_pattern(lbp, pattern_type):
    """Helper function to analyze texture patterns"""
    hist = np.histogram(lbp, bins=10)[0]
    hist = hist.astype(float) / hist.sum()
    
    # Pattern-specific analysis
    if pattern_type == 'tile':
        return 1.0 - np.std(hist)  # Regular patterns have low std
    elif pattern_type == 'natural':
        return np.std(hist)  # Natural textures have high variation
    elif pattern_type == 'technical':
        return float(hist[0] > 0.3)  # Technical surfaces often have uniform areas
    else:
        return 0.5  # Default score

def analyze_texture(frame):
    """Analyze texture patterns in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate texture metrics
    texture_metrics = {
        'variance': np.var(lbp),
        'uniformity': len(np.unique(lbp)) / (n_points + 2),
        'contrast': np.std(gray)
    }
    
    return texture_metrics

def detect_motion_blur(frame):
    """Detect and quantify motion blur"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.var(laplacian)
    
    # Classify blur level
    if blur_score < 100:
        return 'high_blur'
    elif blur_score < 500:
        return 'moderate_blur'
    return 'low_blur'

# Emotion Analysis Helpers
def calculate_scene_emotion_score(scene_data):
    """Calculate emotional scores from scene data"""
    scores = {}
    
    if 'color_mood' in scene_data:
        for mood, value in scene_data['color_mood'].items():
            scores[mood] = value * 0.4  # Weight color mood
            
    if 'lighting_mood' in scene_data:
        for mood, value in scene_data['lighting_mood'].items():
            scores[mood] = scores.get(mood, 0) + value * 0.3  # Weight lighting
            
    if 'composition_mood' in scene_data:
        for mood, value in scene_data['composition_mood'].items():
            scores[mood] = scores.get(mood, 0) + value * 0.3  # Weight composition
    
    return scores

def categorize_emotional_intensity(score):
    """Categorize emotional intensity based on score"""
    if score > 0.8:
        return "very"
    elif score > 0.6:
        return "notably"
    elif score > 0.4:
        return "moderately"
    elif score > 0.2:
        return "slightly"
    else:
        return "minimally"

# Mood Analysis Functions
def analyze_color_mood(hsv):
    """Analyze mood based on color properties"""
    try:
        # Extract HSV channels
        hue = hsv[:,:,0]
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Calculate average values
        avg_saturation = np.mean(saturation)
        avg_value = np.mean(value)
        
        # Initialize mood scores
        moods = {
            'energetic': 0.0,
            'calm': 0.0,
            'dark': 0.0,
            'vibrant': 0.0
        }
        
        # Score based on color properties
        if avg_saturation > 150:  # High saturation
            moods['energetic'] = min(1.0, avg_saturation / 255)
            moods['vibrant'] = min(1.0, avg_saturation / 200)
        else:  # Low saturation
            moods['calm'] = 1.0 - (avg_saturation / 255)
        
        if avg_value < 100:  # Dark image
            moods['dark'] = 1.0 - (avg_value / 255)
        
        return moods
        
    except Exception as e:
        logger.warning(f"Error in color mood analysis: {str(e)}")
        return {'neutral': 1.0}

def analyze_lighting_mood(frame):
    """Analyze mood based on lighting characteristics"""
    try:
        # Convert to LAB for better lighting analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        
        # Calculate lighting metrics
        brightness = np.mean(l_channel)
        contrast = np.std(l_channel)
        
        # Initialize mood scores
        moods = {
            'dramatic': 0.0,
            'intimate': 0.0,
            'atmospheric': 0.0,
            'natural': 0.0
        }
        
        # Score based on lighting characteristics
        if contrast > 60:
            moods['dramatic'] = min(1.0, contrast / 100)
        
        if brightness < 100:
            moods['intimate'] = 1.0 - (brightness / 255)
            moods['atmospheric'] = 0.6
        else:
            moods['natural'] = brightness / 255
        
        return moods
        
    except Exception as e:
        logger.warning(f"Error in lighting mood analysis: {str(e)}")
        return {'neutral': 1.0}

def analyze_composition_mood(frame):
    """Analyze mood based on compositional elements"""
    try:
        height, width = frame.shape[:2]
        
        # Initialize mood scores
        moods = {
            'balanced': 0.0,
            'dynamic': 0.0,
            'structured': 0.0,
            'chaotic': 0.0
        }
        
        # Analyze composition metrics
        try:
            thirds_score = analyze_composition_thirds(frame)
            symmetry_score = analyze_symmetry(frame)
            
            if symmetry_score > 0.7:
                moods['balanced'] = symmetry_score
                moods['structured'] = 0.7
            
            if thirds_score > 0.6:
                moods['dynamic'] = thirds_score
            else:
                moods['chaotic'] = 1.0 - thirds_score
                
        except Exception as e:
            logger.warning(f"Error in composition metrics: {str(e)}")
        
        return moods
        
    except Exception as e:
        logger.warning(f"Error in composition mood analysis: {str(e)}")
        return {'neutral': 1.0}

def analyze_movement_mood(frame):
    """Analyze mood based on movement in frame"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge intensity
        edges = cv2.Canny(gray, 50, 150)
        edge_intensity = np.mean(edges) / 255.0
        
        # Initialize mood scores
        moods = {
            'energetic': 0.0,
            'calm': 0.0,
            'flowing': 0.0,
            'static': 0.0
        }
        
        # Score based on edge intensity
        if edge_intensity > 0.3:
            moods['energetic'] = min(1.0, edge_intensity * 2)
        else:
            moods['calm'] = 1.0 - edge_intensity
            moods['static'] = 0.8
        
        return moods
        
    except Exception as e:
        logger.warning(f"Error in movement mood analysis: {str(e)}")
        return {'neutral': 1.0}

def analyze_eq_profile_fixed(y, sr, n_fft, hop_length):
    """Analyze EQ profile with fixed parameters"""
    try:
        logger.info(f"Analyzing EQ profile with n_fft: {n_fft}, hop_length: {hop_length}")
        
        # Compute STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(D)
        
        logger.info(f"EQ STFT shape: {S.shape}")
        
        # Define frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mids': (250, 500),
            'mids': (500, 2000),
            'high_mids': (2000, 4000),
            'highs': (4000, 20000)
        }
        
        # Get frequencies
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        profile = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                band_energy = np.mean(S[mask])
                profile[band_name] = {
                    'energy': float(band_energy),
                    'prominence': float(band_energy / (np.mean(S) + 1e-8))
                }
            else:
                profile[band_name] = {
                    'energy': 0.0,
                    'prominence': 0.0
                }
        
        return profile
        
    except Exception as e:
        logger.error(f"Error in EQ profile analysis: {str(e)}")
        return {band: {'energy': 0.0, 'prominence': 0.0} for band in bands}


def analyze_bass_character_fixed(y, sr, n_fft, hop_length):
    """Analyze bass character with fixed parameters"""
    try:
        logger.info(f"Analyzing bass character with n_fft: {n_fft}, hop_length: {hop_length}")
        
        # Filter for bass frequencies
        nyquist = sr // 2
        filter_freq = 250 / nyquist
        b, a = scipy.signal.butter(4, filter_freq, btype='low')
        y_bass = scipy.signal.filtfilt(b, a, y)
        
        # Compute STFT for bass
        D_bass = librosa.stft(y_bass, n_fft=n_fft, hop_length=hop_length)
        S_bass = np.abs(D_bass)
        
        logger.info(f"Bass STFT shape: {S_bass.shape}")
        
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y_bass,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Calculate characteristics
        rms = librosa.feature.rms(y=y_bass, frame_length=n_fft, hop_length=hop_length)[0]
        
        return {
            'intensity': float(np.mean(rms)),
            'variation': float(np.std(rms)),
            'character': 'punchy' if np.max(rms) / (np.mean(rms) + 1e-8) > 2 else 'smooth'
        }
        
    except Exception as e:
        logger.error(f"Error in bass character analysis: {str(e)}")
        return {
            'intensity': 0.0,
            'variation': 0.0,
            'character': 'unknown'
        }


def detect_strobe_effect(hsv):
    """
    Detect strobe-like lighting effects.
    
    Args:
        hsv (numpy.ndarray): HSV color space image
        
    Returns:
        bool: True if strobe effect detected
    """
    # Analyze value channel for rapid changes
    value = hsv[:,:,2]
    
    # Calculate local variance in brightness
    kernel_size = 5
    local_var = cv2.blur(value, (kernel_size, kernel_size))
    var_of_var = np.var(local_var)
    
    # High local variance suggests strobe effect
    return var_of_var > 2000

def detect_color_wash(hsv):
    """
    Detect color wash effects.
    
    Args:
        hsv (numpy.ndarray): HSV color space image
        
    Returns:
        bool: True if color wash effect detected
    """
    # Analyze hue and saturation
    hue = hsv[:,:,0]
    saturation = hsv[:,:,1]
    
    # Calculate hue consistency and saturation level
    hue_hist = cv2.calcHist([hue], [0], None, [180], [0,180])
    hue_hist = hue_hist.flatten() / hue_hist.sum()
    
    # Color wash typically has dominant hue and high saturation
    dominant_hue_ratio = np.max(hue_hist)
    avg_saturation = np.mean(saturation)
    
    return dominant_hue_ratio > 0.3 and avg_saturation > 100

def detect_spotlight(lab):
    """
    Detect spotlight effects in lighting.
    
    Args:
        lab (numpy.ndarray): LAB color space image
        
    Returns:
        bool: True if spotlight effect detected
    """
    # Analyze luminance channel
    luminance = lab[:,:,0]
    
    # Calculate brightness distribution
    hist = cv2.calcHist([luminance], [0], None, [256], [0,256])
    hist = hist.flatten() / hist.sum()
    
    # Find peaks in brightness distribution
    peaks, _ = scipy.signal.find_peaks(hist, height=0.02, distance=50)
    
    # Spotlight typically creates bimodal brightness distribution
    if len(peaks) >= 2:
        peak_diff = np.abs(np.diff(peaks))
        return np.any(peak_diff > 100)
    
    return False

def detect_color_grading(hsv):
    """
    Detect color grading effects.
    
    Args:
        hsv (numpy.ndarray): HSV color space image
        
    Returns:
        bool: True if color grading detected
    """
    # Analyze hue and saturation distributions
    hue = hsv[:,:,0]
    saturation = hsv[:,:,1]
    
    # Calculate color statistics
    hue_hist = cv2.calcHist([hue], [0], None, [180], [0,180])
    hue_hist = hue_hist.flatten() / hue_hist.sum()
    
    # Color grading often shows as selective color enhancement
    hue_peaks, _ = scipy.signal.find_peaks(hue_hist, height=0.05)
    
    # Check for selective saturation
    sat_mean = np.mean(saturation)
    sat_std = np.std(saturation)
    
    return len(hue_peaks) <= 3 and sat_std > 30

def detect_vignette(lab):
    """
    Detect vignette effects.
    
    Args:
        lab (numpy.ndarray): LAB color space image
        
    Returns:
        bool: True if vignette effect detected
    """
    # Get luminance channel
    luminance = lab[:,:,0]
    height, width = luminance.shape
    
    # Create distance matrix from center
    y, x = np.ogrid[0:height, 0:width]
    center_y, center_x = height//2, width//2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    distance = distance / np.max(distance)
    
    # Calculate correlation between distance and brightness
    brightness_distance_corr = np.corrcoef(
        distance.flatten(),
        luminance.flatten()
    )[0,1]
    
    # Strong negative correlation suggests vignette
    return brightness_distance_corr < -0.3

def detect_blur_effect(frame):
    """
    Detect intentional blur effects.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        bool: True if artistic blur detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (blur metric)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate local variance to detect selective focus
    local_var = cv2.blur(gray, (20, 20))
    var_of_var = np.var(local_var)
    
    # High local variance but low overall sharpness suggests artistic blur
    return laplacian_var < 500 and var_of_var > 1000

def detect_chromatic_aberration(frame):
    """
    Detect chromatic aberration effects.
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        bool: True if chromatic aberration detected
    """
    # Split into color channels
    b, g, r = cv2.split(frame)
    
    # Calculate edge maps for each channel
    edges_b = cv2.Canny(b, 100, 200)
    edges_g = cv2.Canny(g, 100, 200)
    edges_r = cv2.Canny(r, 100, 200)
    
    # Calculate edge differences
    diff_rg = cv2.absdiff(edges_r, edges_g)
    diff_rb = cv2.absdiff(edges_r, edges_b)
    diff_gb = cv2.absdiff(edges_g, edges_b)
    
    # High edge differences suggest chromatic aberration
    total_diff = np.mean(diff_rg) + np.mean(diff_rb) + np.mean(diff_gb)
    
    return total_diff > 30

def calculate_overall_confidence(context):
    """Calculate overall confidence in emotional analysis"""
    confidences = [
        context['facial']['confidence'],
        context['body']['confidence'],
        context['gesture']['confidence'],
        context['scene']['confidence']
    ]
    
    # Weight different confidence scores
    weights = [0.4, 0.3, 0.2, 0.1]
    
    return sum(c * w for c, w in zip(confidences, weights))

def analyze_emotion_progression(emotion_profile):
    """Analyze how emotions progress/change"""
    if not emotion_profile or len(emotion_profile) < 2:
        return None
        
    changes = []
    prev_emotion = None
    
    for emotion, score in emotion_profile.items():
        if prev_emotion and abs(score - prev_emotion[1]) > 0.3:
            changes.append(f"shifting from {prev_emotion[0]} to {emotion}")
        prev_emotion = (emotion, score)
    
    if changes:
        return f"with {' and '.join(changes)}"
    return None

def analyze_text_style_and_content(text_regions, frame):
    """
    Analyze text style and content type.
    
    Args:
        text_regions (list): List of detected text regions
        frame (numpy.ndarray): Video frame
        
    Returns:
        dict: Analysis of text style and content
    """
    if not text_regions:
        return None
    
    # Extract frame timestamp (assuming video analysis context)
    frame_time = frame.shape[0]  # Placeholder for actual timestamp
    
    # Analyze largest text region (usually main content)
    largest_region = max(text_regions, key=lambda x: 
                        x['region'][2] * x['region'][3])
    
    # Get region properties
    properties = largest_region['properties']
    
    # Determine text type and style
    text_type = classify_text_type(properties)
    style_attributes = analyze_style_attributes(properties)
    
    return {
        'type': text_type,
        'content': extract_text_content(frame, largest_region['region']),
        'style': style_attributes,
        'timing': frame_time
    }

def classify_text_type(properties):
    """
    Classify type of text (lyrics, quote, title, etc.).
    
    Args:
        properties (dict): Text region properties
        
    Returns:
        str: Classified text type
    """
    # Size-based classification
    height = properties['size'][0]
    if height > 50:
        return 'title'
    elif height > 30:
        return 'quote'
    else:
        return 'lyrics'

def analyze_style_attributes(properties):
    """
    Analyze text style attributes.
    
    Args:
        properties (dict): Text region properties
        
    Returns:
        set: Set of style attributes
    """
    attributes = set()
    
    # Add basic style attributes
    if properties['style']['bold']:
        attributes.add('bold')
    if properties['style']['italic']:
        attributes.add('italic')
    
    # Analyze color-based attributes
    if properties['contrast'] > 50:
        attributes.add('high_contrast')
    
    # Analyze based on color
    dominant_color = properties['color']['dominant']
    if np.mean(dominant_color) > 200:
        attributes.add('bright')
    elif np.mean(dominant_color) < 50:
        attributes.add('dark')
    
    return attributes

def detect_trending_elements(frame):
    """
    Detect trending visual elements in frame.
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        list: List of detected trending elements
    """
    trends = []
    
    # Analyze frame characteristics
    height, width = frame.shape[:2]
    
    # Check for split screen
    left_half = frame[:, :width//2]
    right_half = frame[:, width//2:]
    if np.mean(np.abs(left_half - right_half)) > 50:
        trends.append('split_screen')
    
    # Check for duet-style layout
    if detect_duet_layout(frame):
        trends.append('duet_style')
    
    # Check for popular effects
    if detect_popular_effects(frame):
        trends.extend(['slow_motion', 'time_warp'])
    
    return trends

def detect_storytelling_elements(frame):
    """Detect storytelling elements with validation"""
    try:
        if not validate_frame(frame):
            return []
            
        elements = []
        
        # Analyze scene composition
        composition = analyze_scene_composition(frame)
        if composition and composition.get('focal_point'):
            elements.append('focused_subject')
        
        # Detect lighting with proper color conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if np.std(gray) > 60:
            elements.append('dramatic_lighting')
        
        # Detect emotional cues (frame already RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        if np.mean(hsv[:,:,1]) > 150:
            elements.append('intense_colors')
        
        if np.mean(lab[:,:,0]) < 100:
            elements.append('moody_lighting')
        elif np.mean(lab[:,:,0]) > 200:
            elements.append('uplifting_lighting')
        
        return elements
        
    except Exception as e:
        logger.warning(f"Error detecting storytelling elements: {str(e)}")
        return []

def summarize_storytelling(elements):
    """
    Summarize detected storytelling elements.
    
    Args:
        elements (list): List of storytelling elements
        
    Returns:
        dict: Summary of storytelling approach
    """
    if not elements:
        return {}
    
    # Count element frequencies
    element_counts = Counter(elements)
    
    # Determine primary storytelling approach
    primary_elements = [elem for elem, count in element_counts.most_common(3)
                       if count > len(elements) * 0.2]
    
    return {
        'primary_elements': primary_elements,
        'style': determine_story_style(primary_elements),
        'consistency': len(primary_elements) / len(set(elements))
    }

def determine_story_style(elements):
    """
    Determine overall storytelling style.
    
    Args:
        elements (list): Primary storytelling elements
        
    Returns:
        str: Identified storytelling style
    """
    style_indicators = {
        'dramatic': ['dramatic_lighting', 'emotional_intensity'],
        'narrative': ['scene_progression', 'character_focus'],
        'artistic': ['composition_focus', 'visual_metaphor'],
        'documentary': ['natural_lighting', 'candid_moments']
    }
    
    # Count matches for each style
    style_matches = {
        style: sum(1 for elem in elements if elem in indicators)
        for style, indicators in style_indicators.items()
    }
    
    # Return most matched style
    return max(style_matches.items(), key=lambda x: x[1])[0]

def detect_duet_layout(frame):
    """
    Detect if frame uses duet-style layout.
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        bool: True if duet layout detected
    """
    height, width = frame.shape[:2]
    
    # Check for vertical split
    left_half = frame[:, :width//2]
    right_half = frame[:, width//2:]
    
    # Calculate difference between halves
    diff = np.mean(np.abs(left_half - right_half))
    
    return diff > 50

def detect_popular_effects(frame):
    """
    Detect popular video effects.
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        list: Detected effects
    """
    effects = []
    
    # Convert to HSV for effect detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Check for color manipulation
    if np.std(hsv[:,:,0]) > 50:
        effects.append('color_filter')
    
    # Check for exposure effects
    if np.mean(hsv[:,:,2]) > 200:
        effects.append('bright_exposure')
    elif np.mean(hsv[:,:,2]) < 50:
        effects.append('dark_exposure')
    
    return effects

def detect_dramatic_lighting(frame):
    """
    Detect dramatic lighting in frame.
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        bool: True if dramatic lighting detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate lighting contrast
    contrast = np.std(gray)
    
    # Check for high contrast (dramatic lighting)
    return contrast > 60

def detect_emotional_cues(frame):
    """
    Detect emotional visual cues in frame.
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        list: Detected emotional cues
    """
    cues = []
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Analyze color mood
    if np.mean(hsv[:,:,1]) > 150:  # High saturation
        cues.append('intense_colors')
    
    # Analyze lighting mood
    if np.mean(lab[:,:,0]) < 100:  # Dark lighting
        cues.append('moody_lighting')
    elif np.mean(lab[:,:,0]) > 200:  # Bright lighting
        cues.append('uplifting_lighting')
    
    return cues

def determine_primary_focus(engagement_data):
    """Determine primary engagement focus"""
    try:
        # Count strategies by type
        strategy_counts = {
            'visual': len(engagement_data['visual_strategies'].get('strategies', [])),
            'audio': len(engagement_data['audio_strategies'].get('strategies', [])),
            'performance': len(engagement_data['performance_strategies'].get('strategies', [])),
            'content': len(engagement_data['content_strategies'].get('strategies', []))
        }
        
        if not any(strategy_counts.values()):
            return "unknown"
            
        # Get type with most strategies
        primary_focus = max(strategy_counts.items(), key=lambda x: x[1])[0]
        
        return primary_focus
        
    except Exception as e:
        logger.debug(f"Error determining primary focus: {str(e)}")
        return "unknown"


def determine_primary_engagement(engagement_features):
    """
    Determines primary engagement strategy from features.
    
    Args:
        engagement_features (dict): Dictionary of engagement features
        
    Returns:
        str: Primary engagement strategy
    """
    try:
        scores = {
            'visual': sum([
                bool(engagement_features.get('visual', {}).get('text_overlay', False)),
                bool(engagement_features.get('visual', {}).get('effects', False)),
                bool(engagement_features.get('visual', {}).get('hooks', False))
            ]),
            'audio': sum([
                bool(engagement_features.get('audio', {}).get('call_response', False)),
                bool(engagement_features.get('audio', {}).get('repetition', False)),
                len(engagement_features.get('audio', {}).get('hooks', [])) > 0
            ]),
            'performance': sum([
                bool(engagement_features.get('performance', {}).get('direct_address', False)),
                bool(engagement_features.get('performance', {}).get('gestures', False))
            ])
        }
        
        if not any(scores.values()):
            return "unknown"
            
        return max(scores.items(), key=lambda x: x[1])[0]
        
    except Exception as e:
        logger.error(f"Error determining primary engagement: {str(e)}")
        return "unknown"

def calculate_engagement_score(engagement_features):
    """
    Calculates overall engagement score from features.
    
    Args:
        engagement_features (dict): Dictionary of engagement features
        
    Returns:
        float: Engagement score between 0 and 1
    """
    score = 0
    total_weight = 0
    
    # Visual engagement (weight: 0.4)
    visual_score = sum([
        bool(engagement_features['visual']['text_overlay']) * 0.3,
        bool(engagement_features['visual']['effects']) * 0.3,
        len(engagement_features['visual']['hooks']) * 0.1
    ])
    score += visual_score * 0.4
    total_weight += 0.4
    
    # Audio engagement (weight: 0.3)
    audio_score = sum([
        bool(engagement_features['audio']['call_response']) * 0.3,
        bool(engagement_features['audio']['repetition']) * 0.3,
        min(len(engagement_features['audio']['hooks']) * 0.1, 0.4)
    ])
    score += audio_score * 0.3
    total_weight += 0.3
    
    # Performance engagement (weight: 0.3)
    performance_score = sum([
        bool(engagement_features['performance']['direct_address']) * 0.5,
        bool(engagement_features['performance']['gestures']) * 0.5
    ])
    score += performance_score * 0.3
    total_weight += 0.3
    
    return score / total_weight if total_weight > 0 else 0

def aggregate_setting_features(features):
    """
    Aggregates setting features across frames.
    
    Args:
        features (list): List of setting feature dictionaries
        
    Returns:
        dict: Aggregated setting characteristics
    """
    if not features:
        return {}
        
    aggregated = {
        'brightness': np.mean([f['brightness'] for f in features]),
        'texture': np.mean([f['texture'] for f in features]),
        'edge_density': np.mean([f['edge_density'] for f in features])
    }
    
    return aggregated

def describe_location(primary_location, setting_details):
    """
    Generates human-readable location description.
    
    Args:
        primary_location (str): Main location type
        setting_details (dict): Setting characteristics
        
    Returns:
        str: Detailed location description
    """
    description = []
    
    # Basic location
    description.append(primary_location)
    
    # Add detail based on setting characteristics
    if setting_details.get('brightness', 0) > 150:
        description.append("well-lit")
    elif setting_details.get('brightness', 0) < 100:
        description.append("dimly lit")
    
    if setting_details.get('edge_density', 0) > 0.3:
        description.append("with many objects/details")
    
    if setting_details.get('texture', 0) > 50:
        description.append("textured environment")
    
    return " ".join(description)

def classify_transition(prev_frame, curr_frame):
    """
    Classifies the type of transition between frames.
    
    Args:
        prev_frame (numpy.ndarray): Previous video frame
        curr_frame (numpy.ndarray): Current video frame
        
    Returns:
        str: Transition type description
    """
    diff = cv2.absdiff(prev_frame, curr_frame)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    if mean_diff > 100:
        if max_diff > 200:
            return "hard cut"
        return "fade"
    elif np.std(diff) > 50:
        return "swipe"
    else:
        return "smooth transition"

def analyze_text_patterns(text_regions):
    """
    Analyzes patterns in detected text regions.
    
    Args:
        text_regions (list): List of detected text regions
        
    Returns:
        bool: True if consistent text overlay is detected
    """
    if not text_regions:
        return False
    
    # Count frames with text
    frames_with_text = len([r for r in text_regions if r])
    text_ratio = frames_with_text / len(text_regions)
    
    return text_ratio > 0.3  # Text present in >30% of frames

def analyze_effect_patterns(effects):
    """
    Analyzes patterns in detected visual effects.
    
    Args:
        effects (list): List of detected effects
        
    Returns:
        str: Description of dominant effects
    """
    if not effects:
        return "no effects"
    
    # Flatten effect list
    all_effects = [e for sublist in effects for e in sublist]
    effect_counter = Counter(all_effects)
    
    if not effect_counter:
        return "no effects"
    
    # Get most common effects
    common_effects = [effect for effect, count in effect_counter.most_common(2)
                     if count > len(effects) * 0.2]  # Present in >20% of frames
    
    if common_effects:
        return ", ".join(common_effects)
    return "minimal effects"

def analyze_visual_hooks(frame):
    """Analyze visual hooks with validation"""
    try:
        if not validate_frame(frame):
            return []
            
        hooks = []
        
        # Check composition
        composition = analyze_scene_composition(frame)
        if composition and composition.get('focal_point'):
            hooks.append('strong_composition')
            
        # Check lighting
        lighting = analyze_lighting_pattern(frame)
        if lighting and lighting.get('type') in ['dramatic', 'high_key']:
            hooks.append('dramatic_lighting')
            
        # Check visual effects
        effects = detect_frame_effects(frame)
        if effects:
            hooks.extend(['visual_effect_' + effect for effect in effects])
            
        # Check for text
        text_regions = detect_text_enhanced(frame)
        if text_regions:
            hooks.append('text_element')
            
        return hooks
        
    except Exception as e:
        logger.warning(f"Error detecting visual hooks: {str(e)}")
        return []


def analyze_direct_address(metrics):
    """Analyze if performance directly addresses viewer"""
    if not metrics:
        return False

    try:
        # Count frames with direct address
        center_frames = 0
        total_frames = len(metrics)

        for m in metrics:
            if not m or 'position' not in m:
                continue

            x, y, w, h = m['position']
            
            # Calculate relative position
            if 'frame_size' in m:
                frame_width = m['frame_size'].get('width', 1)
                frame_height = m['frame_size'].get('height', 1)
            else:
                continue

            # Check horizontal centering
            face_center_x = (x + w/2) / frame_width
            if 0.3 < face_center_x < 0.7:
                # Check vertical position
                face_center_y = (y + h/2) / frame_height
                if 0.2 < face_center_y < 0.8:
                    # Check face size
                    face_size_ratio = (w * h) / (frame_width * frame_height)
                    if face_size_ratio > 0.05:  # Face is large enough
                        center_frames += 1

        # Return True if face is centered in enough frames
        return center_frames > total_frames * 0.3 if total_frames > 0 else False

    except Exception as e:
        logger.warning(f"Error in direct address analysis: {str(e)}")
        return False

def analyze_gesture_patterns(metrics):
    """
    Analyzes patterns in performer gestures.
    
    Args:
        metrics (list): List of performance metrics
        
    Returns:
        bool: True if significant gesturing is detected
    """
    if not metrics:
        return False
    
    # Calculate position changes
    position_changes = []
    for i in range(1, len(metrics)):
        prev_pos = metrics[i-1]['position']
        curr_pos = metrics[i]['position']
        change = np.sqrt((prev_pos[0] - curr_pos[0])**2 + (prev_pos[1] - curr_pos[1])**2)
        position_changes.append(change)
    
    return np.mean(position_changes) > 20  # Significant average movement

def analyze_emotional_range(metrics):
    """
    Analyzes range of detected emotions.
    
    Args:
        metrics (list): List of performance metrics
        
    Returns:
        float: Score indicating emotional range (0-1)
    """
    if not metrics:
        return 0.0
    
    # Track emotion changes
    emotions = [max(m['emotions'].items(), key=lambda x: x[1])[0] for m in metrics]
    emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
    
    return emotion_changes / (len(metrics) - 1) if len(metrics) > 1 else 0.0

def detect_frame_effects(frame):
    """Detect visual effects with validation"""
    try:
        if not validate_frame(frame):
            return []
        
        # Preserve original size
        original_shape = frame.shape
        
        effects = []
        
        # Convert to HSV (frame is already RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Check various effects
        if detect_color_grading(hsv):
            effects.append("color_grading")
            
        if detect_vignette(lab):
            effects.append("vignette")
            
        if detect_blur_effect(frame):
            effects.append("blur")
            
        if detect_chromatic_aberration(frame):
            effects.append("chromatic_aberration")
            
        # Basic effects
        if np.std(hsv[:,:,0]) > 50:
            effects.append("color_filter")
            
        if np.mean(hsv[:,:,2]) > 200:
            effects.append("overexposed")
        elif np.mean(hsv[:,:,2]) < 50:
            effects.append("underexposed")
        
        return effects
        
    except Exception as e:
        logger.warning(f"Error detecting frame effects: {str(e)}")
        return []

def summarize_transitions(transitions):
    """
    Summarizes detected transition types.
    
    Args:
        transitions (list): List of transition types
        
    Returns:
        str: Summary of transition patterns
    """
    if not transitions:
        return "no transitions"
    
    transition_counter = Counter(transitions)
    
    # Get most common transitions
    common = transition_counter.most_common(2)
    if not common:
        return "minimal transitions"
    
    return f"{common[0][0]}" + (f" and {common[1][0]}" if len(common) > 1 else "")

def analyze_edit_rhythm(scene_changes, frame_rate):
    """
    Analyzes rhythm of video edits.
    
    Args:
        scene_changes (list): List of frame numbers where scenes change
        frame_rate (int): Video frame rate
        
    Returns:
        str: Description of editing rhythm
    """
    if not scene_changes:
        return "single shot"
    
    # Calculate time between cuts
    cut_intervals = np.diff([x/frame_rate for x in scene_changes])
    avg_interval = np.mean(cut_intervals)
    std_interval = np.std(cut_intervals)
    
    if avg_interval < 1.0:
        return "fast paced"
    elif avg_interval < 3.0:
        return "moderate pace"
    else:
        return "slow paced"

def analyze_effect_sequences(effects, frame_rate):
    """
    Analyzes sequences of visual effects.
    
    Args:
        effects (list): List of (frame_number, effects) tuples
        frame_rate (int): Video frame rate
        
    Returns:
        dict: Analysis of effect patterns
    """
    if not effects:
        return {'pattern': 'no effects', 'frequency': 0}
    
    # Convert frame numbers to timestamps
    effect_times = [frame/frame_rate for frame, _ in effects]
    
    # Analyze timing patterns
    intervals = np.diff(effect_times)
    avg_interval = np.mean(intervals) if intervals.size > 0 else 0
    
    if avg_interval < 1.0:
        pattern = "rapid effects"
    elif avg_interval < 3.0:
        pattern = "regular effects"
    else:
        pattern = "sparse effects"
    
    return {
        'pattern': pattern,
        'frequency': len(effects)/(effect_times[-1] - effect_times[0]) if len(effect_times) > 1 else 0
    }

# Camera strategy patterns
CAMERA_STRATEGIES = {
    'Static': {
        'motion_threshold': 0.2,
        'features': ['fixed position', 'stable frame', 'tripod mounted']
    },
    'Handheld': {
        'motion_threshold': 0.5,
        'features': ['slight movement', 'natural shake', 'following motion']
    },
    'Dynamic': {
        'motion_threshold': 0.8,
        'features': ['intentional movement', 'panning', 'tracking shots']
    }
}

# Engagement strategy patterns
ENGAGEMENT_STRATEGIES = {
    'Text': ['lyrics', 'quotes', 'captions', 'text overlay'],
    'Visual': ['transitions', 'effects', 'filters', 'split screen'],
    'Performance': ['dancing', 'gestures', 'facial expressions'],
    'Interactive': ['responding to comments', 'direct address', 'questions'],
    'Narrative': ['storytelling', 'multiple scenes', 'progression']
}

def analyze_color_distribution(frame):
    """Analyze color distribution in a frame"""
    # Convert to RGB for better color analysis
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = rgb_frame.reshape(-1, 3)
    
    # Perform k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=5, n_init=1)
    kmeans.fit(pixels)
    
    # Get the colors and their percentages
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    percentages = np.bincount(labels) / len(labels)
    
    return list(zip(colors, percentages))

def analyze_visual_aesthetics(frame):
    """Analyze frame aesthetics with improved error handling"""
    try:
        if not isinstance(frame, np.ndarray):
            return None

        aesthetic_scores = {
            'composition': analyze_composition(frame),
            'color_harmony': analyze_color_harmony(frame),
            'visual_balance': analyze_visual_balance(frame),
            'depth': analyze_depth(frame),
            'score': 0.0  # Default score
        }
        
        # Calculate overall score
        valid_scores = [score for score in [
            aesthetic_scores['color_harmony'],
            aesthetic_scores['visual_balance']
        ] if isinstance(score, (int, float))]
        
        if valid_scores:
            aesthetic_scores['score'] = sum(valid_scores) / len(valid_scores)
            
        return aesthetic_scores
        
    except Exception as e:
        logger.debug(f"Aesthetics analysis error: {str(e)}")
        return {
            'composition': None,
            'color_harmony': 0.0,
            'visual_balance': 0.0,
            'depth': None,
            'score': 0.0
        }

def analyze_composition(frame):
    """Advanced composition analysis"""
    scores = {
        'rule_of_thirds': analyze_composition_thirds(frame),
        'symmetry': analyze_symmetry(frame),
        'leading_lines': detect_leading_lines(frame),
        'framing': analyze_framing(frame),
        'focal_point': detect_focal_point(frame)
    }
    return scores

def analyze_composition_thirds(frame):
    """Enhanced rule of thirds analysis"""
    height, width = frame.shape[:2]
    h_third = height // 3
    w_third = width // 3
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Analyze intersection points
    intersection_points = [
        (w_third, h_third),
        (2*w_third, h_third),
        (w_third, 2*h_third),
        (2*w_third, 2*h_third)
    ]
    
    intersection_scores = []
    for x, y in intersection_points:
        region = edges[y-20:y+20, x-20:x+20]
        score = np.sum(region) / (40 * 40)
        intersection_scores.append(score)
    
    return np.mean(intersection_scores)

def detect_leading_lines(frame):
    """Detect leading lines in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0
    
    # Analyze line directions and convergence
    line_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        line_angles.append(angle)
    
    # Score based on line convergence
    angle_histogram = np.histogram(line_angles, bins=36)[0]
    return np.max(angle_histogram) / len(lines)

def analyze_framing(frame):
    """Analyze subject framing"""
    # Detect main subject (face or body)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
    
    if len(faces) == 0:
        return 0
    
    height, width = frame.shape[:2]
    framing_scores = []
    
    for (x, y, w, h) in faces:
        # Calculate relative position
        center_x = x + w/2
        center_y = y + h/2
        
        # Score based on centering and headroom
        horizontal_score = 1 - abs((center_x / width) - 0.5) * 2
        vertical_score = 1 - abs((center_y / height) - 0.4) * 2
        
        framing_scores.append((horizontal_score + vertical_score) / 2)
    
    return max(framing_scores)

def detect_focal_point(frame):
    """Detect and analyze main focal point"""
    # Convert to LAB color space for better analysis
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Calculate focus map using Laplacian
    focus_map = cv2.Laplacian(l_channel, cv2.CV_64F)
    focus_map = np.absolute(focus_map)
    
    # Find region of highest focus
    kernel_size = 20
    kernel = np.ones((kernel_size,kernel_size)) / (kernel_size*kernel_size)
    focus_map_blurred = cv2.filter2D(focus_map, -1, kernel)
    
    # Get coordinates of maximum focus
    max_loc = np.unravel_index(focus_map_blurred.argmax(), focus_map_blurred.shape)
    
    return analyze_focal_point_position(max_loc, frame.shape[:2])

def analyze_color_harmony(frame):
    """Analyze color harmony and relationships"""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract hue values and create histogram
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
    
    # Detect color harmony patterns
    harmony_patterns = {
        'complementary': detect_complementary_colors(hue_hist),
        'analogous': detect_analogous_colors(hue_hist),
        'triadic': detect_triadic_colors(hue_hist),
        'monochromatic': detect_monochromatic_colors(hsv)
    }
    
    return calculate_harmony_score(harmony_patterns)


def analyze_performance_engagement(frames):
    """Analyze performance engagement with comprehensive metrics"""
    try:
        if not frames:
            return get_default_performance_metrics()
            
        performance_metrics = {
            'direct_address': False,
            'gestures': False,
            'emotional_range': 0.0,
            'performance_techniques': [],
            'confidence_score': 0.0,
            'energy_score': 0.0,
            'frame_metrics': []
        }
        
        direct_address_frames = 0
        gesture_frames = 0
        emotional_scores = []
        
        for frame_idx, frame in enumerate(frames):
            try:
                # Get frame metrics
                metrics = analyze_performance_metrics(frame)
                if metrics:
                    performance_metrics['frame_metrics'].append(metrics)
                    
                    # Check for direct address
                    if analyze_direct_address_enhanced(metrics).get('direct_address', False):
                        direct_address_frames += 1
                        
                    # Detect gestures
                    if detect_performance_gestures(frame):
                        gesture_frames += 1
                        
                    # Track emotional expression
                    if metrics.get('emotions'):
                        emotional_scores.append(max(metrics['emotions'].values()))
                        
                    # Add detected techniques
                    techniques = detect_performance_techniques(frame)
                    if techniques:
                        performance_metrics['performance_techniques'].extend(techniques)
                        
            except Exception as e:
                logger.debug(f"Error processing frame {frame_idx}: {str(e)}")
                continue
                
        # Calculate final metrics
        total_frames = len(frames)
        if total_frames > 0:
            performance_metrics['direct_address'] = direct_address_frames / total_frames > 0.3
            performance_metrics['gestures'] = gesture_frames / total_frames > 0.25
            
            if emotional_scores:
                performance_metrics['emotional_range'] = calculate_emotional_range_score(emotional_scores)
                
        # Remove duplicate techniques
        performance_metrics['performance_techniques'] = list(set(performance_metrics['performance_techniques']))
        
        # Calculate overall scores
        performance_metrics['confidence_score'] = calculate_confidence_score(performance_metrics['frame_metrics'])
        performance_metrics['energy_score'] = calculate_energy_score(performance_metrics['frame_metrics'])
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}")
        return get_default_performance_metrics()

def detect_performance_gestures(frame):
    """Detect performance-related gestures and movements"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion metrics
        edges = cv2.Canny(gray, 100, 200)
        movement_intensity = np.mean(edges) / 255.0
        
        # Detect specific gesture patterns
        has_gestures = False
        
        # Look for raised hands or movement above shoulders
        upper_half = edges[:edges.shape[0]//2, :]
        upper_movement = np.mean(upper_half) / 255.0
        
        # Look for side-to-side movement
        left_side = edges[:, :edges.shape[1]//3]
        right_side = edges[:, 2*edges.shape[1]//3:]
        side_movement = (np.mean(left_side) + np.mean(right_side)) / (2 * 255.0)
        
        # Combine metrics
        has_gestures = (
            movement_intensity > 0.15 or
            upper_movement > 0.2 or
            side_movement > 0.15
        )
        
        return has_gestures
        
    except Exception as e:
        logger.debug(f"Error in gesture detection: {str(e)}")
        return False

def calculate_confidence_score(frame_metrics):
    """Calculate overall performance confidence score"""
    try:
        if not frame_metrics:
            return 0.0
            
        confidence_scores = []
        for metrics in frame_metrics:
            if not metrics:
                continue
                
            # Factor in emotions
            if metrics.get('emotions'):
                emotion_intensity = max(metrics['emotions'].values()) / 100.0
                confidence_scores.append(emotion_intensity * 0.4)
                
            # Factor in direct address
            if metrics.get('position'):
                x, y, w, h = metrics['position']
                frame_height, frame_width = metrics.get('frame_size', (1, 1))
                face_size_ratio = (w * h) / (frame_width * frame_height)
                if face_size_ratio > 0.05:
                    confidence_scores.append(0.3)
                    
        return np.mean(confidence_scores) if confidence_scores else 0.0
        
    except Exception as e:
        logger.debug(f"Error calculating confidence score: {str(e)}")
        return 0.0

def calculate_energy_score(frame_metrics):
    """Calculate overall performance energy score"""
    try:
        if not frame_metrics:
            return 0.0
            
        energy_scores = []
        for metrics in frame_metrics:
            if not metrics:
                continue
                
            # Calculate movement energy
            if metrics.get('position'):
                gray = cv2.cvtColor(metrics.get('frame', np.zeros((100, 100, 3))), cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                movement_intensity = np.mean(edges) / 255.0
                energy_scores.append(movement_intensity * 0.5)
                
            # Factor in emotional intensity
            if metrics.get('emotions'):
                emotion_intensity = max(metrics['emotions'].values()) / 100.0
                energy_scores.append(emotion_intensity * 0.5)
                
        return np.mean(energy_scores) if energy_scores else 0.0
        
    except Exception as e:
        logger.debug(f"Error calculating energy score: {str(e)}")
        return 0.0

def calculate_emotional_range_score(emotional_scores):
    """Calculate emotional range from sequence of scores"""
    try:
        if not emotional_scores:
            return 0.0
            
        # Calculate variance in emotional intensity
        emotion_variance = np.std(emotional_scores)
        
        # Get difference between highest and lowest
        emotion_range = max(emotional_scores) - min(emotional_scores)
        
        # Combine metrics
        range_score = (emotion_variance * 0.5 + emotion_range * 0.5)
        
        return float(range_score)
        
    except Exception as e:
        logger.debug(f"Error calculating emotional range: {str(e)}")
        return 0.0

def analyze_visual_engagement(frame):
    """
    Analyze visual engagement elements in a frame.
    
    Args:
        frame (np.ndarray): Video frame
        
    Returns:
        dict: Visual engagement features
    """
    try:
        # Validate frame here directly instead of using analyze_frame_safely
        if not validate_frame(frame):
            return {
                'text_overlay': False,
                'effects': [],
                'hooks': [],
                'composition': []
            }
            
        engagement = {
            'text_overlay': False,
            'effects': [],
            'hooks': [],
            'composition': []
        }
        
        # Detect text overlay
        text_regions = detect_text_enhanced(frame)
        if text_regions:
            engagement['text_overlay'] = True
            
        # Detect visual effects
        effects = detect_frame_effects(frame)
        if effects:
            engagement['effects'] = effects
            
        # Detect visual hooks
        hooks = detect_visual_hooks(frame)
        if hooks:
            engagement['hooks'] = hooks
            
        # Analyze composition
        composition = analyze_scene_composition(frame)
        if composition:
            engagement['composition'] = [composition]
            
        return engagement
        
    except Exception as e:
        logger.warning(f"Error in visual engagement analysis: {str(e)}")
        return {
            'text_overlay': False,
            'effects': [],
            'hooks': [],
            'composition': []
        }

def analyze_visual_balance(frame):
    """Analyze visual balance of the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Split frame into quadrants
    top_left = np.mean(gray[:height//2, :width//2])
    top_right = np.mean(gray[:height//2, width//2:])
    bottom_left = np.mean(gray[height//2:, :width//2])
    bottom_right = np.mean(gray[height//2:, width//2:])
    
    # Calculate balance scores
    horizontal_balance = 1 - abs((top_left + bottom_left) - (top_right + bottom_right)) / 255
    vertical_balance = 1 - abs((top_left + top_right) - (bottom_left + bottom_right)) / 255
    
    return (horizontal_balance + vertical_balance) / 2

def describe_color_scheme(dominant_colors):
    """Convert color analysis into human-readable description"""
    descriptions = []
    for color, percentage in dominant_colors:
        if percentage > 0.2:  # Only describe significant colors
            color_name = get_color_name(color)
            intensity = "dominant" if percentage > 0.4 else "prominent"
            descriptions.append(f"{color_name} ({intensity})")
    
    return ", ".join(descriptions)

def get_color_name(rgb):
    """Convert RGB values to color name"""
    colors = {
        'Red': [255,0,0],
        'Green': [0,255,0],
        'Blue': [0,0,255],
        'White': [255,255,255],
        'Black': [0,0,0],
        'Yellow': [255,255,0],
        'Purple': [128,0,128],
        'Orange': [255,165,0],
        'Pink': [255,192,203],
        'Gray': [128,128,128]
    }
    
    min_dist = float('inf')
    closest_color = 'Unknown'
    for name, value in colors.items():
        dist = np.linalg.norm(rgb - np.array(value))
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    
    return closest_color


def detect_visual_effects(frame):
    """Detect visual effects in frame"""
    effects = []
    
    # Convert to HSV for effect detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Check for color effects
    if np.std(hsv[:,:,0]) > 50:
        effects.append('color_filter')
    
    # Check for brightness/contrast effects
    if np.std(hsv[:,:,2]) > 80:
        effects.append('exposure_effect')
    
    return effects

def detect_visual_hooks(frame):
    """Detect visual hooks like transitions, split screens etc"""
    hooks = []
    height, width = frame.shape[:2]
    
    # Check for split screen
    left_half = frame[:, :width//2]
    right_half = frame[:, width//2:]
    if np.mean(np.abs(left_half - right_half)) > 50:
        hooks.append('split_screen')
    
    return hooks

def extract_audio_features(audio_path):
    """Extract comprehensive audio features with improved detection"""
    try:
        # Load audio with consistent parameters
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Define consistent frame sizes
        n_fft = 2048
        hop_length = 512
        
        features = {
            'rhythm': {
                'tempo': 0.0,
                'beat_strength': 0.0,
                'beat_regularity': 0.0,
                'rhythm_patterns': {
                    'trap': False,
                    'drill': False,
                    'boom_bap': False,
                    'dance': False
                }
            },
            'tonal': {
                'key': 'C',
                'mode': 'major',
                'harmony_complexity': 0.0,
                'chord_progression': []
            },
            'spectral': {
                'spectral_centroid': [],
                'spectral_bandwidth': [],
                'spectral_rolloff': [],
                'mfccs': []
            },
            'production': {
                'dynamic_range': 0.0,
                'compression': 0.0,
                'reverb': 0.0,
                'distortion': 0.0,
                'eq_profile': {},
                'bass_character': {}
            },
            'engagement': {
                'call_response': False,
                'repetition': False,
                'hooks': [],
                'dynamics': 'moderate',
                'energy_profile': {}
            }
        }

        # Extract rhythm features
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            
            features['rhythm']['tempo'] = float(tempo)
            features['rhythm']['beat_strength'] = float(np.mean(onset_env))
            
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                features['rhythm']['beat_regularity'] = analyze_beat_regularity(beat_times)
                
            # Detect rhythm patterns
            features['rhythm']['rhythm_patterns'] = detect_rhythm_patterns_fixed(y, sr, n_fft, hop_length)
            
        except Exception as e:
            logger.warning(f"Error in rhythm analysis: {str(e)}")

        # Extract tonal features
        try:
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
            features['tonal'].update(analyze_tonal_features(chromagram, sr))
            
        except Exception as e:
            logger.warning(f"Error in tonal analysis: {str(e)}")

        # Extract spectral features
        try:
            features['spectral'] = extract_spectral_features_fixed(y, sr, n_fft, hop_length)
            
        except Exception as e:
            logger.warning(f"Error in spectral analysis: {str(e)}")

        # Extract production features
        try:
            production_features = analyze_production_features(y, sr)
            # Add EQ and bass character analysis directly
            production_features['eq_profile'] = analyze_eq_profile_fixed(y, sr, n_fft=n_fft, hop_length=hop_length)
            production_features['bass_character'] = analyze_bass_character_fixed(y, sr, n_fft=n_fft, hop_length=hop_length)
            features['production'] = production_features
            
        except Exception as e:
            logger.warning(f"Error in production analysis: {str(e)}")

        # Analyze engagement patterns
        try:
            # Change this call to only use audio_path
            features['engagement'] = detect_audio_engagement_fixed(audio_path)
            
        except Exception as e:
            logger.warning(f"Error in engagement analysis: {str(e)}")

        return features

    except Exception as e:
        logger.error(f"Error in audio feature extraction: {str(e)}")
        return None


def extract_rhythm_features(y, sr, n_fft=2048, hop_length=512):
    """
    Extract comprehensive rhythm features with proper frame size handling and validation.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        n_fft (int): FFT window size (deprecated, kept for compatibility)
        hop_length (int): Number of samples between frames (deprecated, kept for compatibility)
        
    Returns:
        dict: Dictionary containing rhythm features
    """
    try:
        # Validate input signal
        if y is None or len(y) == 0:
            logger.error("Empty or invalid audio signal")
            return get_default_rhythm_features()
            
        # Ensure minimum signal length (3 seconds)
        min_length = sr * 3
        if len(y) < min_length:
            logger.info("Padding short audio signal")
            y = np.pad(y, (0, min_length - len(y)))
            
        # Use correct frame sizes for librosa
        frame_length = 254  # Specific size required by librosa
        hop_length = 128   # Corresponding hop length
        
        # Initialize features dictionary
        features = {
            'tempo': 0.0,
            'beat_strength': 0.0,
            'beat_regularity': 0.0,
            'rhythm_patterns': {},
            'additional_metrics': {}
        }
        
        try:
            # Get onset envelope with correct parameters
            onset_env = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                n_fft=frame_length,
                hop_length=hop_length,
                aggregate=np.median,
                center=True
            )
            
            # Normalize onset envelope
            onset_env = librosa.util.normalize(onset_env)
            
            # Get tempo and beats
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                start_bpm=120,
                tightness=100
            )
            features['tempo'] = float(tempo)
            
            # Calculate beat strength
            if len(onset_env) > 0:
                features['beat_strength'] = float(np.mean(onset_env))
            
            # Calculate beat regularity if we have enough beats
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                features['beat_regularity'] = calculate_beat_regularity(beat_times)
                
                # Additional beat metrics
                features['additional_metrics'].update(
                    calculate_additional_beat_metrics(beat_times)
                )
            
            # Analyze rhythm patterns using the new method
            features['rhythm_patterns'] = detect_rhythm_patterns(y, sr)
            
            # Add tempo stability metric
            if len(onset_env) > 1:
                tempo_stability = calculate_tempo_stability(
                    onset_env, sr, hop_length
                )
                features['additional_metrics']['tempo_stability'] = tempo_stability
            
            # Validate all features
            features = validate_rhythm_features(features)
            
        except librosa.ParameterError as e:
            logger.error(f"Librosa parameter error: {str(e)}")
            return get_default_rhythm_features()
        except Exception as e:
            logger.error(f"Error in rhythm analysis: {str(e)}")
            return get_default_rhythm_features()
            
        return features
        
    except Exception as e:
        logger.error(f"Fatal error in rhythm feature extraction: {str(e)}")
        return get_default_rhythm_features()

def calculate_additional_beat_metrics(beat_times):
    """
    Calculate additional beat-related metrics.
    
    Args:
        beat_times (np.ndarray): Array of beat times in seconds
        
    Returns:
        dict: Dictionary of additional beat metrics
    """
    try:
        metrics = {
            'beat_count': 0,
            'average_bpm': 0.0,
            'bpm_stability': 0.0,
            'groove_consistency': 0.0
        }
        
        if len(beat_times) < 2:
            return metrics
            
        # Calculate intervals
        intervals = np.diff(beat_times)
        
        # Beat count
        metrics['beat_count'] = len(beat_times)
        
        # Average BPM
        average_interval = np.mean(intervals)
        if average_interval > 0:
            metrics['average_bpm'] = float(60.0 / average_interval)
            
        # BPM stability
        if len(intervals) > 1:
            metrics['bpm_stability'] = float(1.0 - (np.std(intervals) / np.mean(intervals)))
            
        # Groove consistency (pattern of every 2-4 beats)
        if len(intervals) >= 4:
            pattern_lengths = [2, 3, 4]
            consistencies = []
            
            for length in pattern_lengths:
                if len(intervals) >= length * 2:
                    patterns = intervals[:len(intervals) - (len(intervals) % length)].reshape(-1, length)
                    if len(patterns) >= 2:
                        pattern_consistency = 1.0 - np.mean([
                            np.std(patterns[:, i]) / np.mean(patterns[:, i])
                            for i in range(length)
                        ])
                        consistencies.append(pattern_consistency)
                        
            if consistencies:
                metrics['groove_consistency'] = float(max(consistencies))
                
        # Normalize all metrics to 0-1 range
        for key in metrics:
            if isinstance(metrics[key], (float, np.float32, np.float64)):
                metrics[key] = float(max(0.0, min(1.0, metrics[key])))
                
        return metrics
        
    except Exception as e:
        logger.warning(f"Error calculating additional beat metrics: {str(e)}")
        return {
            'beat_count': 0,
            'average_bpm': 0.0,
            'bpm_stability': 0.0,
            'groove_consistency': 0.0
        }


def calculate_beat_regularity(beat_times):
    """
    Calculate beat regularity from beat times.
    
    Args:
        beat_times (np.ndarray): Array of beat times in seconds
        
    Returns:
        float: Beat regularity score between 0 and 1
    """
    try:
        if len(beat_times) < 2:
            return 0.0
            
        # Calculate intervals between beats
        intervals = np.diff(beat_times)
        
        # Remove outliers (more than 2 standard deviations from mean)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        valid_intervals = intervals[
            np.abs(intervals - mean_interval) <= 2 * std_interval
        ]
        
        if len(valid_intervals) < 2:
            return 0.0
            
        # Calculate regularity as inverse of coefficient of variation
        regularity = 1.0 - (np.std(valid_intervals) / np.mean(valid_intervals))
        
        # Normalize to 0-1 range
        return float(max(0.0, min(1.0, regularity)))
        
    except Exception as e:
        logger.warning(f"Error calculating beat regularity: {str(e)}")
        return 0.0



def extract_tonal_features(y, sr):
    """Advanced tonal analysis"""
    harmonic = librosa.effects.harmonic(y)
    
    features = {
        'key': estimate_key(y, sr),
        'mode': estimate_mode(y, sr),
        'chroma': np.mean(librosa.feature.chroma_cqt(y=harmonic, sr=sr), axis=1).tolist(),
        'harmony_complexity': analyze_harmonic_complexity(y, sr)
    }
    
    return features

def extract_spectral_features_fixed(y, sr, n_fft, hop_length):
    """Detailed spectral analysis with fixed parameters"""
    try:
        features = {
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0])),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0])),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0])),
            'mfccs': librosa.feature.mfcc(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).mean(axis=1).tolist()
        }
        
        return features
    except Exception as e:
        logger.error(f"Error in spectral feature extraction: {str(e)}")
        return {
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_rolloff': 0.0,
            'mfccs': []
        }

def analyze_production_features(y, sr):
   """Analyze production characteristics with corrected frame sizes"""
   try:
       # Use smaller frame sizes
       frame_length = 256
       hop_length = 128
       
       # Analyze production characteristics
       characteristics = {
           'reverb': measure_reverb(y),
           'compression': measure_compression(y),
           'distortion': measure_distortion(y),
           'eq_profile': analyze_eq_profile_fixed(y, sr, n_fft=frame_length, hop_length=hop_length),
           'bass_character': analyze_bass_character_fixed(y, sr, n_fft=frame_length, hop_length=hop_length)
       }
       
       return characteristics
       
   except Exception as e:
       logger.error(f"Error in production analysis: {str(e)}")
       return {
           'reverb': 0.0,
           'compression': 0.0,
           'distortion': 0.0,
           'eq_profile': {},
           'bass_character': {}
       }

def detect_rhythm_patterns_fixed(y, sr, n_fft, hop_length):
    """Detect rhythm patterns with fixed parameters and improved logging"""
    try:
        logger.info(f"Detecting rhythm patterns with n_fft: {n_fft}, hop_length: {hop_length}")
        
        patterns = {
            'trap': False,
            'drill': False,
            'boom_bap': False,
            'dance': False
        }

        # Detect each pattern with logging
        try:
            patterns['trap'] = detect_trap_pattern_fixed(y, sr, n_fft, hop_length)
            logger.debug("Completed trap pattern detection")
        except Exception as e:
            logger.warning(f"Error in trap pattern detection: {str(e)}")

        try:
            patterns['drill'] = detect_drill_pattern_fixed(y, sr, n_fft, hop_length)
            logger.debug("Completed drill pattern detection")
        except Exception as e:
            logger.warning(f"Error in drill pattern detection: {str(e)}")

        try:
            patterns['boom_bap'] = detect_boom_bap_pattern_fixed(y, sr, n_fft, hop_length)
            logger.debug("Completed boom bap pattern detection")
        except Exception as e:
            logger.warning(f"Error in boom bap pattern detection: {str(e)}")

        try:
            patterns['dance'] = detect_dance_pattern_fixed(y, sr, n_fft, hop_length)
            logger.debug("Completed dance pattern detection")
        except Exception as e:
            logger.warning(f"Error in dance pattern detection: {str(e)}")

        logger.info("Completed rhythm pattern detection")
        return patterns
        
    except Exception as e:
        logger.error(f"Error in rhythm pattern detection: {str(e)}")
        return {
            'trap': False,
            'drill': False,
            'boom_bap': False,
            'dance': False
        }

def analyze_beat_regularity(beat_frames):
    """Analyze regularity of beat patterns"""
    if len(beat_frames) < 2:
        return 0
        
    beat_intervals = np.diff(beat_frames)
    regularity = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
    return max(0, min(1, regularity))

def estimate_key(y, sr):
    """Estimate musical key with consistent frame sizes"""
    try:
        frame_length = 254
        hop_length = 128
        
        # Compute chromagram
        chromagram = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hop_length
        )
        
        if chromagram.size == 0:
            return 'C'
            
        # Calculate key weights
        key_weights = np.mean(chromagram, axis=1)
        key_index = np.argmax(key_weights)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return keys[key_index]
        
    except Exception as e:
        logger.warning(f"Error in key estimation: {str(e)}")
        return 'C'


def estimate_mode(y, sr):
    """Estimate whether music is major or minor with consistent frame sizes"""
    try:
        frame_length = 254
        hop_length = 128
        
        # Extract harmonic component
        harmonic = librosa.effects.harmonic(y)
        
        # Compute chromagram
        chromagram = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=sr,
            hop_length=hop_length
        )
        
        if chromagram.size == 0:
            return 'major'
            
        # Check major/minor thirds
        major_third = np.mean(chromagram[4])
        minor_third = np.mean(chromagram[3])
        
        return "major" if major_third > minor_third else "minor"
        
    except Exception as e:
        logger.warning(f"Error in mode estimation: {str(e)}")
        return 'major'

def analyze_harmonic_complexity(y, sr, hop_length=128):
    """Analyze harmonic complexity with consistent frame sizes"""
    try:
        frame_length = 254
        
        # Extract harmonic component
        harmonic = librosa.effects.harmonic(y)
        
        # Compute chromagram with consistent frame size
        chroma = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=sr,
            hop_length=hop_length
        )
        
        if chroma.size == 0:
            return 0.0
            
        # Calculate entropy of pitch class distribution
        pitch_entropy = stats.entropy(np.mean(chroma, axis=1))
        
        # Normalize to 0-1 range
        max_entropy = np.log(12)  # Maximum entropy for 12 pitch classes
        complexity = pitch_entropy / max_entropy
        
        return float(complexity)
        
    except Exception as e:
        logger.warning(f"Error in harmonic complexity analysis: {str(e)}")
        return 0.0

def calculate_tempo_stability(onset_env, sr, hop_length):
    """
    Calculate tempo stability over time.
    
    Args:
        onset_env (np.ndarray): Onset strength envelope
        sr (int): Sampling rate
        hop_length (int): Number of samples between frames
        
    Returns:
        float: Tempo stability score between 0 and 1
    """
    try:
        if len(onset_env) < sr / hop_length * 3:  # At least 3 seconds
            return 0.0
            
        # Analyze tempo in sliding windows
        window_length = int(sr / hop_length * 2)  # 2-second windows
        hop_size = window_length // 2  # 50% overlap
        
        tempos = []
        for i in range(0, len(onset_env) - window_length, hop_size):
            window = onset_env[i:i + window_length]
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=window,
                sr=sr,
                hop_length=hop_length,
                start_bpm=120
            )
            tempos.append(tempo)
            
        if not tempos:
            return 0.0
            
        # Calculate stability as inverse of tempo variance
        stability = 1.0 - (np.std(tempos) / np.mean(tempos))
        
        return float(max(0.0, min(1.0, stability)))
        
    except Exception as e:
        logger.warning(f"Error calculating tempo stability: {str(e)}")
        return 0.0

def get_default_rhythm_features():
    """
    Return default rhythm features structure.
    
    Returns:
        dict: Default rhythm features
    """
    return {
        'tempo': 0.0,
        'beat_strength': 0.0,
        'beat_regularity': 0.0,
        'rhythm_patterns': {
            'trap': False,
            'drill': False,
            'boom_bap': False,
            'dance': False
        },
        'additional_metrics': {
            'beat_count': 0,
            'average_bpm': 0.0,
            'bpm_stability': 0.0,
            'groove_consistency': 0.0,
            'tempo_stability': 0.0
        }
    }

def validate_rhythm_features(features):
    """
    Validate rhythm features and ensure all fields are present.
    
    Args:
        features (dict): Rhythm features to validate
        
    Returns:
        dict: Validated rhythm features
    """
    try:
        default_features = get_default_rhythm_features()
        
        # Ensure all keys exist
        for key in default_features:
            if key not in features:
                features[key] = default_features[key]
                
        # Validate numeric values
        for key in ['tempo', 'beat_strength', 'beat_regularity']:
            if key in features:
                try:
                    features[key] = float(features[key])
                    features[key] = max(0.0, min(1.0, features[key]))
                except (TypeError, ValueError):
                    features[key] = 0.0
                    
        # Validate rhythm patterns
        if 'rhythm_patterns' in features:
            default_patterns = default_features['rhythm_patterns']
            for pattern in default_patterns:
                if pattern not in features['rhythm_patterns']:
                    features['rhythm_patterns'][pattern] = False
                    
        # Validate additional metrics
        if 'additional_metrics' in features:
            default_metrics = default_features['additional_metrics']
            for metric in default_metrics:
                if metric not in features['additional_metrics']:
                    features['additional_metrics'][metric] = default_metrics[metric]
                    
        return features
        
    except Exception as e:
        logger.error(f"Error validating rhythm features: {str(e)}")
        return get_default_rhythm_features()

def calculate_dynamic_range_fixed(y, n_fft, hop_length):
    """Calculate dynamic range of audio with fixed parameters"""
    try:
        # Calculate RMS energy with fixed parameters
        rms = librosa.feature.rms(
            y=y, 
            frame_length=n_fft, 
            hop_length=hop_length
        )[0]
        
        # Calculate dynamic range in dB
        if len(rms) > 0:
            db_range = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-8))
            return float(db_range)
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating dynamic range: {str(e)}")
        return 0.0

def estimate_compression(y):
    """Estimate amount of dynamic range compression"""
    # Calculate crest factor (peak to RMS ratio)
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    crest_factor = peak / (rms + 1e-8)
    
    # Convert to compression estimate (0-1 range)
    compression = 1.0 - (np.log10(crest_factor) / np.log10(20))  # 20 is typical uncompressed crest factor
    return max(0, min(1, compression))

def estimate_distortion(y):
    """Estimate presence of distortion"""
    # Calculate zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # High zero crossing rate can indicate distortion
    zcr_mean = np.mean(zcr)
    
    # Normalize to 0-1 range (empirically determined thresholds)
    distortion = (zcr_mean - 0.1) / 0.4  # 0.1 to 0.5 is typical range
    return max(0, min(1, distortion))

def analyze_stereo_width(y):
    """Analyze stereo width if audio is stereo"""
    if len(y.shape) < 2:
        return 0
        
    # Calculate correlation between channels
    correlation = np.corrcoef(y[0], y[1])[0,1]
    
    # Convert to width measure (0-1 range)
    width = 1.0 - abs(correlation)
    return width



def analyze_facial_emotions_detailed(emotions):
    """Enhanced facial emotion analysis"""
    if not emotions:
        return {'confidence': 0, 'emotions': {}}
        
    # Convert basic emotions to more nuanced interpretations
    nuanced_emotions = {
        'happy': {
            'enthusiastic': emotions.get('happy', 0) * 0.8,
            'content': emotions.get('happy', 0) * 0.6,
            'excited': emotions.get('happy', 0) * 0.7
        },
        'sad': {
            'melancholic': emotions.get('sad', 0) * 0.8,
            'reflective': emotions.get('sad', 0) * 0.6,
            'nostalgic': emotions.get('sad', 0) * 0.5
        },
        'neutral': {
            'focused': emotions.get('neutral', 0) * 0.7,
            'composed': emotions.get('neutral', 0) * 0.8,
            'reserved': emotions.get('neutral', 0) * 0.6
        },
        'angry': {
            'intense': emotions.get('angry', 0) * 0.8,
            'passionate': emotions.get('angry', 0) * 0.6,
            'determined': emotions.get('angry', 0) * 0.7
        }
    }
    
    return {
        'confidence': max(emotions.values()),
        'emotions': nuanced_emotions
    }

def analyze_body_language(pose_data):
    """Analyze emotional indicators in body language"""
    if not pose_data:
        return {'confidence': 0, 'indicators': {}}
    
    indicators = {
        'openness': calculate_pose_openness(pose_data),
        'energy': calculate_pose_energy(pose_data),
        'confidence': calculate_pose_confidence(pose_data),
        'engagement': calculate_pose_engagement(pose_data)
    }
    
    return {
        'confidence': np.mean(list(indicators.values())),
        'indicators': indicators
    }

def analyze_gesture_emotions(pose_data):
    """Analyze emotional content of gestures"""
    if not pose_data:
        return {'confidence': 0, 'gestures': {}}
    
    gestures = {
        'expressive': detect_expressive_gestures(pose_data),
        'rhythmic': detect_rhythmic_gestures(pose_data),
        'emphatic': detect_emphatic_gestures(pose_data),
        'interactive': detect_interactive_gestures(pose_data)
    }
    
    return {
        'confidence': np.mean([g['confidence'] for g in gestures.values()]),
        'gestures': gestures
    }

def calculate_pose_openness(pose_data):
    """Calculate how open/closed the body posture is"""
    # Check arm positions
    arms_spread = calculate_arm_spread(pose_data)
    shoulders_open = calculate_shoulder_openness(pose_data)
    
    openness_score = (arms_spread * 0.6 + shoulders_open * 0.4)
    return max(0, min(1, openness_score))

def calculate_pose_energy(pose_data):
    """Calculate energy level from pose movement"""
    if len(pose_data) < 2:
        return 0
    
    # Calculate movement between consecutive poses
    movements = []
    for i in range(1, len(pose_data)):
        movement = calculate_pose_movement(pose_data[i-1], pose_data[i])
        movements.append(movement)
    
    # Normalize and return energy score
    return np.mean(movements) if movements else 0

def calculate_pose_confidence(pose_data):
    """Analyze pose for confidence indicators"""
    if not pose_data:
        return 0
    
    indicators = {
        'upright_posture': calculate_upright_posture(pose_data),
        'shoulder_position': calculate_shoulder_position(pose_data),
        'head_position': calculate_head_position(pose_data),
        'stance_stability': calculate_stance_stability(pose_data)
    }
    
    # Weight and combine indicators
    weights = {
        'upright_posture': 0.3,
        'shoulder_position': 0.3,
        'head_position': 0.2,
        'stance_stability': 0.2
    }
    
    confidence_score = sum(score * weights[indicator] 
                         for indicator, score in indicators.items())
    
    return confidence_score

def detect_expressive_gestures(pose_data):
    """Detect emotionally expressive gestures"""
    if len(pose_data) < 5:  # Need sequence of poses
        return {'confidence': 0, 'types': []}
    
    gesture_types = []
    confidences = []
    
    # Analyze gesture sequences
    for i in range(4, len(pose_data)):
        sequence = pose_data[i-4:i+1]
        
        # Check for different gesture types
        if is_emphasis_gesture(sequence):
            gesture_types.append('emphasis')
            confidences.append(0.8)
        if is_rhythmic_gesture(sequence):
            gesture_types.append('rhythmic')
            confidences.append(0.7)
        if is_flowing_gesture(sequence):
            gesture_types.append('flowing')
            confidences.append(0.9)
    
    return {
        'confidence': np.mean(confidences) if confidences else 0,
        'types': list(set(gesture_types))
    }

def analyze_scene_emotion(frame):
    """Analyze emotional context from scene with safe multiplication"""
    try:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Initialize empty result
        result = {
            'color_mood': {'score': 0.0, 'moods': {}},
            'lighting_mood': {'score': 0.0, 'moods': {}},
            'composition_mood': {'score': 0.0, 'moods': {}},
            'overall_mood': None
        }
        
        # Analyze color mood
        color_result = analyze_color_mood(hsv)
        if isinstance(color_result, dict):
            # Process each mood value individually
            processed_moods = {}
            for mood, value in color_result.items():
                if isinstance(value, (int, float)):
                    processed_moods[mood] = float(value)
            result['color_mood']['moods'] = processed_moods
            result['color_mood']['score'] = sum(processed_moods.values()) / len(processed_moods) if processed_moods else 0.0
        
        # Analyze lighting mood
        lighting_result = analyze_lighting_mood(frame)
        if isinstance(lighting_result, dict):
            # Process each mood value individually
            processed_moods = {}
            for mood, value in lighting_result.items():
                if isinstance(value, (int, float)):
                    processed_moods[mood] = float(value)
            result['lighting_mood']['moods'] = processed_moods
            result['lighting_mood']['score'] = sum(processed_moods.values()) / len(processed_moods) if processed_moods else 0.0
        
        # Analyze composition mood
        comp_result = analyze_composition_mood(frame)
        if isinstance(comp_result, dict):
            # Process each mood value individually
            processed_moods = {}
            for mood, value in comp_result.items():
                if isinstance(value, (int, float)):
                    processed_moods[mood] = float(value)
            result['composition_mood']['moods'] = processed_moods
            result['composition_mood']['score'] = sum(processed_moods.values()) / len(processed_moods) if processed_moods else 0.0
        
        # Calculate overall mood
        moods_combined = {}
        for mood_type in ['color_mood', 'lighting_mood', 'composition_mood']:
            for mood, value in result[mood_type]['moods'].items():
                if mood not in moods_combined:
                    moods_combined[mood] = 0.0
                moods_combined[mood] += value / 3  # Equal weighting
        
        if moods_combined:
            result['overall_mood'] = max(moods_combined.items(), key=lambda x: x[1])[0]
        
        return result
        
    except Exception as e:
        logger.warning(f"Error in scene emotion analysis: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {
            'color_mood': {'score': 0.0, 'moods': {}},
            'lighting_mood': {'score': 0.0, 'moods': {}},
            'composition_mood': {'score': 0.0, 'moods': {}},
            'overall_mood': None
        }

def combine_emotional_indicators(context):
    """Combine different emotional signals with proper type handling"""
    # Weight different sources
    weights = {
        'facial': 0.4,
        'body': 0.3,
        'gesture': 0.2,
        'scene': 0.1
    }
    
    combined_emotions = {}
    
    # Process facial emotions safely
    if context['facial']['confidence'] > 0.3:
        for category, emotions in context['facial']['emotions'].items():
            if isinstance(emotions, dict):
                for emotion, score in emotions.items():
                    if isinstance(score, (int, float)):
                        combined_emotions[emotion] = score * weights['facial']
    
    # Add body language indicators safely
    if context['body']['confidence'] > 0.3:
        for indicator, score in context['body']['indicators'].items():
            if isinstance(score, (int, float)):
                combined_emotions[indicator] = score * weights['body']
    
    # Add gesture information safely
    if context['gesture']['confidence'] > 0.3:
        for gesture_type, details in context['gesture']['gestures'].items():
            if isinstance(details, dict) and 'confidence' in details:
                if details['confidence'] > 0.5 and 'types' in details:
                    for gesture in details['types']:
                        combined_emotions[gesture] = details['confidence'] * weights['gesture']
    
    # Add scene mood safely
    if context['scene'].get('confidence', 0) > 0.3:
        scene_moods = context['scene'].get('moods', {})
        for mood_type, mood_data in scene_moods.items():
            if isinstance(mood_data, dict) and 'score' in mood_data:
                if isinstance(mood_data['score'], (int, float)):
                    combined_emotions[mood_type] = mood_data['score'] * weights['scene']
    
    return {
        'primary_emotion': get_primary_emotion(combined_emotions),
        'emotional_state': generate_emotional_state_description(combined_emotions),
        'confidence': calculate_overall_confidence(context)
    }

def get_primary_emotion(emotions):
    """Determine primary emotion from combined indicators"""
    if not emotions:
        return None
        
    # Sort emotions by intensity
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    # Get top emotion and its intensity
    primary = sorted_emotions[0]
    intensity = categorize_emotional_intensity(primary[1])
    
    return {
        'emotion': primary[0],
        'intensity': intensity,
        'score': primary[1]
    }

def generate_emotional_state_description(emotions):
    """Generate human-readable emotional state description"""
    if not emotions:
        return "Neutral or unclear emotional state"
    
    # Get top emotions
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Generate description
    description_parts = []
    
    # Primary emotion
    primary = top_emotions[0]
    intensity = categorize_emotional_intensity(primary[1])
    description_parts.append(f"{intensity} {primary[0]}")
    
    # Secondary emotions
    if len(top_emotions) > 1 and top_emotions[1][1] > 0.3:
        description_parts.append(f"with {top_emotions[1][0]} undertones")
    
    return " ".join(description_parts)

def detect_genre_from_audio(audio_path):
    """Enhanced genre detection with better error handling"""
    try:
        # Extract features with timeout protection
        features = extract_audio_features(audio_path)
        
        if not features or not any(features.values()):
            logger.warning("No audio features extracted")
            return "Unknown Genre"
        
        # Get genre probabilities with safe feature access
        genre_probs = {}
        
        # Extract relevant features safely with default values
        tempo = features.get('rhythm', {}).get('tempo', 0)
        beat_strength = features.get('rhythm', {}).get('beat_strength', 0)
        beat_regularity = features.get('rhythm', {}).get('beat_regularity', 0)
        
        # Get spectral characteristics safely
        spectral = features.get('spectral', {})
        spectral_centroid = np.mean(spectral.get('spectral_centroid', [0]))
        
        # Get production characteristics safely
        production = features.get('production', {})
        compression = production.get('compression', 0)
        reverb = production.get('reverb', 0)
        
        # Check for trap characteristics
        trap_score = 0
        if 130 <= tempo <= 160 and beat_regularity > 0.7:
            trap_score += 0.4
        if features.get('rhythm', {}).get('rhythm_patterns', {}).get('trap', False):
            trap_score += 0.6
        genre_probs['Trap'] = trap_score
        
        # Check for sad trap characteristics
        sad_trap_score = 0
        if 120 <= tempo <= 150:
            sad_trap_score += 0.3
        if features.get('tonal', {}).get('mode') == 'minor':
            sad_trap_score += 0.4
        if reverb > 0.6:
            sad_trap_score += 0.3
        genre_probs['Sad Trap'] = sad_trap_score
        
        # Check for folk characteristics
        folk_score = 0
        if 80 <= tempo <= 120:
            folk_score += 0.3
        if spectral_centroid < 3000:
            folk_score += 0.4
        if compression < 0.4:
            folk_score += 0.3
        genre_probs['Folk'] = folk_score
        
        # Check for pop characteristics
        pop_score = 0
        if 100 <= tempo <= 130:
            pop_score += 0.3
        if beat_regularity > 0.8:
            pop_score += 0.3
        if compression > 0.6:
            pop_score += 0.4
        genre_probs['Pop'] = pop_score
        
        # Normalize probabilities safely
        total = sum(genre_probs.values())
        if total > 0:
            genre_probs = {k: v/total for k, v in genre_probs.items()}
            
        # Get subgenre details with timeout protection
        try:
            subgenre_details = analyze_subgenre(features, genre_probs)
        except Exception as e:
            logger.warning(f"Error in subgenre analysis: {str(e)}")
            subgenre_details = None
            
        # Detect fusion elements with timeout protection
        try:
            fusion_elements = detect_genre_fusion(features)
        except Exception as e:
            logger.warning(f"Error in fusion detection: {str(e)}")
            fusion_elements = []
        
        # Generate description
        description = generate_genre_description(genre_probs, subgenre_details, fusion_elements)
        return description
        
    except Exception as e:
        logger.error(f"Error in genre detection: {str(e)}")
        return "Unknown Genre"

def classify_genre(features):
    """Classify music genre based on extracted features"""
    try:
        if not features:
            logger.info("No features provided for genre classification")
            return "Unknown Genre"
            
        genre_probs = {}
        
        # Extract relevant features safely with default values
        tempo = features.get('rhythm', {}).get('tempo', 0)
        beat_strength = features.get('rhythm', {}).get('beat_strength', 0)
        beat_regularity = features.get('rhythm', {}).get('beat_regularity', 0)
        
        logger.debug(f"Extracted rhythm features - Tempo: {tempo}, Beat Strength: {beat_strength}, Regularity: {beat_regularity}")
        
        # Get spectral characteristics with safe defaults
        spectral = features.get('spectral', {})
        spectral_centroid = np.mean(spectral.get('spectral_centroid', [0]))
        spectral_rolloff = np.mean(spectral.get('spectral_rolloff', [0]))
        
        logger.debug(f"Extracted spectral features - Centroid: {spectral_centroid}, Rolloff: {spectral_rolloff}")
        
        # Get production characteristics safely
        production = features.get('production', {})
        compression = production.get('compression', 0)
        reverb = production.get('reverb', 0)
        
        logger.debug(f"Extracted production features - Compression: {compression}, Reverb: {reverb}")
        
        # Check for trap characteristics
        trap_score = 0
        if 130 <= tempo <= 160 and beat_regularity > 0.7:
            trap_score += 0.4
        if features.get('rhythm', {}).get('rhythm_patterns', {}).get('trap', False):
            trap_score += 0.6
        genre_probs['Trap'] = trap_score
        
        # Check for sad trap characteristics
        sad_trap_score = 0
        if 120 <= tempo <= 150:
            sad_trap_score += 0.3
        if features.get('tonal', {}).get('mode') == 'minor':
            sad_trap_score += 0.4
        if reverb > 0.6:
            sad_trap_score += 0.3
        genre_probs['Sad Trap'] = sad_trap_score
        
        # Check for folk characteristics
        folk_score = 0
        if 80 <= tempo <= 120:
            folk_score += 0.3
        if spectral_centroid < 3000:
            folk_score += 0.4
        if compression < 0.4:
            folk_score += 0.3
        genre_probs['Folk'] = folk_score
        
        # Check for pop characteristics
        pop_score = 0
        if 100 <= tempo <= 130:
            pop_score += 0.3
        if beat_regularity > 0.8:
            pop_score += 0.3
        if compression > 0.6:
            pop_score += 0.4
        genre_probs['Pop'] = pop_score
        
        logger.debug(f"Initial genre probabilities: {genre_probs}")
        
        # Normalize probabilities
        total = sum(genre_probs.values())
        if total > 0:
            genre_probs = {k: v/total for k, v in genre_probs.items()}
            # Get genre with highest probability
            primary_genre = max(genre_probs.items(), key=lambda x: x[1])[0]
            logger.info(f"Classified genre: {primary_genre} (confidence: {genre_probs[primary_genre]:.2f})")
            return primary_genre
        else:
            logger.info("No clear genre detected")
            return "Unknown Genre"
        
    except Exception as e:
        logger.error(f"Error in genre classification: {str(e)}\n{traceback.format_exc()}")
        return "Unknown Genre"

def generate_genre_description(genre_probs, subgenre_details, fusion_elements):
    """Generate detailed genre description with improved error handling."""
    try:
        if not genre_probs:
            logger.info("No genre probabilities provided")
            return "Unknown Genre"
            
        description_parts = []
        
        # Get primary genre safely
        try:
            primary_genre = max(genre_probs.items(), key=lambda x: x[1])
            logger.debug(f"Primary genre: {primary_genre[0]} (confidence: {primary_genre[1]:.2f})")
        except Exception as e:
            logger.warning(f"Could not determine primary genre: {str(e)}")
            return "Unknown Genre"
            
        if primary_genre[1] < 0.4:
            logger.info(f"Low genre confidence ({primary_genre[1]:.2f}), classifying as genre-bending")
            return "Genre-bending mix"
        
        # Add primary genre
        description_parts.append(primary_genre[0])
        
        # Add subgenre if confident
        if subgenre_details and isinstance(subgenre_details, dict):
            if subgenre_details.get('confidence', 0) > 0.6:
                subgenre_name = subgenre_details.get('name')
                if subgenre_name:
                    description_parts.insert(0, subgenre_name)
        
        # Add fusion elements
        if fusion_elements:
            fusion_desc = "with " + " and ".join(fusion_elements)
            description_parts.append(fusion_desc)
        
        return " ".join(filter(None, description_parts))
        
    except Exception as e:
        logger.warning(f"Error generating genre description: {str(e)}")
        return "Unknown Genre"

def detect_song_structure(audio_path):
    """Enhanced song part detection with fixed parameters"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)  # Use consistent sample rate
        
        # Define fixed parameters
        n_fft = 2048
        hop_length = 512
        
        logger.info(f"Analyzing song structure with n_fft: {n_fft}, hop_length: {hop_length}")
        
        # Extract segments
        segments = librosa.effects.split(y, top_db=30)
        
        # Analyze each segment
        segment_features = []
        for segment in segments:
            seg_audio = y[segment[0]:segment[1]]
            
            # Extract detailed features with fixed parameters
            rms = np.mean(librosa.feature.rms(
                y=seg_audio,
                frame_length=n_fft,
                hop_length=hop_length
            )[0])
            
            spectral = np.mean(librosa.feature.spectral_centroid(
                y=seg_audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )[0])
            
            # Get tempo with fixed parameters
            onset_env = librosa.onset.onset_strength(
                y=seg_audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length
            )
            
            # HPSS with fixed parameters
            harmonic, percussive = librosa.effects.hpss(
                seg_audio,
                margin=2.0
            )
            
            segment_features.append({
                'energy': rms,
                'spectral_content': spectral,
                'tempo': tempo,
                'harmonic_ratio': np.mean(np.abs(harmonic))/np.mean(np.abs(percussive))
            })
        
        # Analyze structure
        parts = []
        
        # Get overall song characteristics
        if segment_features:  # Add check for empty list
            total_energy = np.mean([f['energy'] for f in segment_features])
            avg_tempo = np.mean([f['tempo'] for f in segment_features])
            
            # Determine emotional quality
            emotional_intensity = np.mean([f['harmonic_ratio'] for f in segment_features])
            if emotional_intensity > 1.5:
                parts.append("Emotional")
            
            # Determine section type
            energies = [f['energy'] for f in segment_features]
            if energies:  # Add check for empty list
                max_energy = max(energies)
                if total_energy > 0.8 * max_energy:
                    parts.append("Hook")
                elif total_energy < 0.4 * max_energy:
                    parts.append("Stripped back")
                    parts.append("Verse")
                else:
                    parts.append("Bridge" if emotional_intensity > 1.2 else "Verse")
            
            # Add modifiers based on tempo and energy
            if avg_tempo > 120:
                parts.insert(0, "Sped up")
            if total_energy > 0.7:
                parts.insert(0, "Energetic")
            elif total_energy < 0.3:
                parts.insert(0, "Soft")
        
        return ", ".join(parts) if parts else "Unknown Structure"
    
    except Exception as e:
        logger.error(f"Error in song structure detection: {str(e)}")
        return "Unknown Structure"


def analyze_rhythm_patterns(y, sr, beats, frame_length=254, hop_length=128):
    """Analyze rhythm patterns with consistent frame sizes"""
    try:
        patterns = {
            'trap': False,
            'drill': False,
            'boom_bap': False,
            'dance': False
        }
        
        # Call each pattern detection function
        patterns['trap'] = detect_trap_pattern(y, sr, beats, frame_length, hop_length)
        patterns['drill'] = detect_drill_pattern(y, sr, beats, frame_length, hop_length)
        patterns['boom_bap'] = detect_boom_bap_pattern(y, sr, beats, frame_length, hop_length)
        patterns['dance'] = detect_dance_pattern(y, sr, beats, frame_length, hop_length)
        
        return patterns
        
    except Exception as e:
        logger.warning(f"Error in pattern detection: {str(e)}")
        return {
            'trap': False,
            'drill': False,
            'boom_bap': False,
            'dance': False
        }



def analyze_production_style(y, sr):
    """Analyze production characteristics"""
    # Split into harmonic and percussive
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Define standard frame parameters
    n_fft = 2048
    hop_length = 512
    
    # Analyze characteristics
    characteristics = {
        'reverb': measure_reverb(y),
        'compression': measure_compression(y),
        'distortion': measure_distortion(y),
        'eq_profile': analyze_eq_profile_fixed(y, sr, n_fft=n_fft, hop_length=hop_length),
        'bass_character': analyze_bass_character_fixed(y_percussive, sr, n_fft=n_fft, hop_length=hop_length),
        'vocal_effects': detect_vocal_effects(y_harmonic, sr)
    }
    
    return characteristics

def generate_genre_description(genre_probs, subgenre_details, fusion_elements):
    """Generate detailed genre description"""
    description_parts = []
    
    # Get primary genre
    primary_genre = max(genre_probs.items(), key=lambda x: x[1])
    
    if primary_genre[1] < 0.4:
        return "Genre-bending mix"
    
    # Build description
    description_parts.append(primary_genre[0])
    
    # Add subgenre if confident
    if subgenre_details['confidence'] > 0.6:
        description_parts.insert(0, subgenre_details['name'])
    
    # Add fusion elements
    if fusion_elements:
        fusion_desc = "with " + " and ".join(fusion_elements)
        description_parts.append(fusion_desc)
    
    return " ".join(description_parts)


def detect_camera_strategy(frames):
    """Camera movement analysis with pre-extracted frames"""
    motion_patterns = []
    compositions = []
    prev_frame = None
    
    for frame in frames:
        try:
            if prev_frame is not None:
                # Motion analysis
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_magnitude = np.mean(magnitude)
                motion_direction = np.mean(angle)
                motion_complexity = np.std(magnitude)
                
                motion_patterns.append({
                    'magnitude': motion_magnitude,
                    'direction': motion_direction,
                    'complexity': motion_complexity
                })
            
            # Analyze frame composition
            composition = analyze_scene_composition(frame)
            compositions.append(composition)
            
            prev_frame = frame
            
        except Exception as e:
            logger.warning(f"Error analyzing frame for camera strategy: {str(e)}")
            continue
    
    return analyze_camera_patterns(motion_patterns, compositions)


# Feature Detection Functions
def detect_feature_presence(frame, feature):
    """Detect presence of specific features in frame"""
    # Convert frame to CLIP input
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    # Get CLIP text features
    text_inputs = clip_processor(text=[feature], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        similarity = F.cosine_similarity(image_features, text_features)
        
    return float(similarity[0])

def detect_context_presence(frame, context):
    """Detect presence of contextual elements"""
    # Similar to feature detection but with broader context
    score = detect_feature_presence(frame, context) * 0.7
    
    # Add additional context analysis
    context_score = analyze_contextual_elements(frame, context)
    score += context_score * 0.3
    
    return score

def analyze_contextual_elements(frame, context):
    """Analyze broader contextual elements in frame"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Context-specific analysis
    if 'lighting' in context.lower():
        return analyze_lighting_context(hsv)
    elif 'texture' in context.lower():
        return analyze_texture_context(gray)
    elif 'space' in context.lower():
        return analyze_spatial_context(frame)
    return 0.0

# Lighting Analysis Functions
def detect_lighting_type(frame):
    """Detect type of lighting in frame"""
    # Convert to LAB for better lighting analysis
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Analyze lighting distribution
    hist = cv2.calcHist([l_channel], [0], None, [256], [0,256])
    hist = hist.flatten() / hist.sum()
    
    # Calculate metrics
    avg_intensity = np.average(np.arange(256), weights=hist)
    std_intensity = np.sqrt(np.average((np.arange(256) - avg_intensity)**2, weights=hist))
    
    # Classify lighting type
    if std_intensity < 30:
        return "uniform"
    elif np.argmax(hist) > 200:
        return "high-key"
    elif np.argmax(hist) < 50:
        return "low-key"
    return "dramatic"

def detect_lighting_effects(frame):
    """Detect lighting effects in frame"""
    effects = []
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Check for various effects
    if detect_strobe_effect(hsv):
        effects.append('strobe')
    if detect_color_wash(hsv):
        effects.append('color_wash')
    if detect_spotlight(lab):
        effects.append('spotlight')
        
    return effects

def assess_lighting_quality(frame):
    """Assess technical quality of lighting"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Calculate quality metrics
    contrast = np.std(l_channel)
    uniformity = 1.0 - (np.std(l_channel) / np.mean(l_channel))
    
    if contrast > 60 and uniformity > 0.7:
        return "professional"
    elif contrast > 40 and uniformity > 0.5:
        return "good"
    return "basic"

# Audio Analysis Functions

def detect_808_fixed(y, sr, n_fft, hop_length):
    """Detect presence of 808 bass with fixed frame sizes"""
    try:
        # Filter for sub-bass frequencies (20-100Hz)
        y_bass = librosa.effects.preemphasis(y, coef=0.95)
        
        # Calculate spectral rolloff with consistent parameters
        rolloff = librosa.feature.spectral_rolloff(
            y=y_bass, 
            sr=sr, 
            n_fft=n_fft,
            hop_length=hop_length,
            roll_percent=0.85
        )
        
        return np.mean(rolloff) < 100  # Strong sub-bass presence
        
    except Exception as e:
        logger.warning(f"Error detecting 808: {str(e)}")
        return False

def detect_hihat_rolls_fixed(y, sr, n_fft, hop_length):
    """Detect hi-hat roll patterns with fixed parameters"""
    try:
        # Extract high frequencies
        y_high = librosa.effects.preemphasis(y)
        
        # Detect onsets in high frequencies
        onset_env = librosa.onset.onset_strength(
            y=y_high, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        # Analyze onset patterns
        if len(onsets) > 0:
            onset_intervals = np.diff(onsets)
            return np.any(onset_intervals < sr/8)  # Detect rapid sequences
            
        return False
        
    except Exception as e:
        logger.warning(f"Error detecting hihat rolls: {str(e)}")
        return False

def detect_sliding_bass_fixed(y, sr, n_fft, hop_length):
    """Detect sliding 808 bass patterns with fixed parameters"""
    try:
        # Extract bass frequencies
        y_bass = librosa.effects.preemphasis(y, coef=0.95)
        
        # Track pitch changes
        pitches, magnitudes = librosa.piptrack(
            y=y_bass,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Add dimension check
        if pitches.size == 0:
            return False
            
        # Calculate pitch changes with dimension check
        mean_pitches = np.mean(pitches, axis=1)
        if mean_pitches.size < 2:
            return False
            
        pitch_changes = np.diff(mean_pitches)
        
        return np.any(np.abs(pitch_changes) > 10)
        
    except Exception as e:
        logger.warning(f"Error detecting sliding bass: {str(e)}")
        return False

def detect_kick_pattern_fixed(y, sr, n_fft, hop_length):
    """Detect kick drum patterns with fixed parameters"""
    try:
        # Emphasize low frequencies
        y_low = librosa.effects.preemphasis(y, coef=0.95)
        
        # Detect onsets in low frequencies
        onset_env = librosa.onset.onset_strength(
            y=y_low, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        if len(onsets) > 0:
            onset_intervals = np.diff(onsets)
            return np.std(onset_intervals) < 0.2  # Regular pattern
        return False
        
    except Exception as e:
        logger.warning(f"Error detecting kick pattern: {str(e)}")
        return False

def detect_snare_pattern_fixed(y, sr, n_fft, hop_length):
    """Detect snare patterns with fixed parameters"""
    try:
        # Emphasize mid frequencies
        y_mid = librosa.effects.preemphasis(y, coef=0.5)
        
        # Detect onsets
        onset_env = librosa.onset.onset_strength(
            y=y_mid, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        if len(onsets) > 0:
            onset_intervals = np.diff(onsets)
            return np.median(onset_intervals) > sr/2  # Typical backbeat pattern
        return False
        
    except Exception as e:
        logger.warning(f"Error detecting snare pattern: {str(e)}")
        return False

def detect_four_on_floor_fixed(y, sr, n_fft, hop_length):
    """Detect four-on-the-floor beat pattern with fixed parameters"""
    try:
        # Extract kick drum hits
        y_kick = librosa.effects.preemphasis(y, coef=0.95)
        onset_env = librosa.onset.onset_strength(
            y=y_kick, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        if len(onsets) > 0:
            # Check for regular intervals
            intervals = np.diff(onsets)
            return np.std(intervals) < 0.1 and np.mean(intervals) > sr/3
        return False
        
    except Exception as e:
        logger.warning(f"Error detecting four on floor pattern: {str(e)}")
        return False

# Performance Analysis Functions
def detect_performance_type(frame):
    """Detect type of performance with improved error handling"""
    try:
        if not validate_frame(frame):
            return "unknown"

        # Convert frame for CLIP analysis
        try:
            image = Image.fromarray(frame)  # Frame should already be RGB
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)

            # Performance types to check
            performances = [
                "singing performance",
                "dancing performance",
                "rapping performance",
                "instrumental performance"
            ]

            text_inputs = clip_processor(text=performances, return_tensors="pt", padding=True)

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(1),
                    text_features.unsqueeze(0)
                )

                if similarities.numel() > 0:
                    max_idx = similarities[0].argmax().item()
                    if max_idx < len(performances):
                        return performances[max_idx].split()[0]

        except Exception as e:
            logger.debug(f"CLIP analysis error: {str(e)}")
            return "unknown"

        return "unknown"

    except Exception as e:
        logger.debug(f"Error in performance type detection: {str(e)}")
        return "unknown"


def detect_detailed_location(frame):
    """Enhanced location detection keeping original function name"""
    try:
        # Location categories with specific visual markers
        locations = {
            'bathroom': {
                'markers': ['bathroom mirror', 'sink', 'toilet', 'bathroom tiles'],
                'colors': [(255, 255, 255), (230, 230, 230)],  # White, off-white
                'texture_pattern': 'tile'
            },
            'bedroom': {
                'markers': ['bed', 'bedroom mirror', 'dresser', 'bedroom wall'],
                'colors': [(200, 200, 200), (240, 220, 180)],  # Grey, beige
                'texture_pattern': 'fabric'
            },
            'studio': {
                'markers': ['microphone stand', 'studio monitors', 'audio interface', 'acoustic panels'],
                'colors': [(50, 50, 50), (30, 30, 30)],  # Dark colors
                'texture_pattern': 'technical'
            },
            'outdoor': {
                'markers': ['sky', 'buildings', 'trees', 'street'],
                'colors': [(135, 206, 235), (34, 139, 34)],  # Sky blue, green
                'texture_pattern': 'natural'
            },
            'venue': {
                'markers': ['stage', 'performance lights', 'speakers', 'venue crowd'],
                'colors': [(20, 20, 20), (100, 100, 100)],  # Dark colors
                'texture_pattern': 'stage'
            }
        }

        scores = {loc: 0.0 for loc in locations.keys()}
        
        # Convert frame for CLIP analysis
        image = Image.fromarray(frame)
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        
        # Analyze each location
        for location, features in locations.items():
            # Check visual markers using CLIP
            text_inputs = clip_processor(text=features['markers'], return_tensors="pt", padding=True)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(1),
                    text_features.unsqueeze(0)
                )
                marker_score = float(torch.max(similarities))
                scores[location] += marker_score * 0.4
                
            # Check colors
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            color_score = 0
            for color in features['colors']:
                color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
                mask = cv2.inRange(hsv, color_hsv * 0.7, color_hsv * 1.3)
                color_score += np.sum(mask) / (frame.shape[0] * frame.shape[1])
            scores[location] += color_score * 0.3
            
            # Check texture patterns
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            texture_score = analyze_texture_pattern(lbp, features['texture_pattern'])
            scores[location] += texture_score * 0.3
            
        # Get location with highest score
        best_location = max(scores.items(), key=lambda x: x[1])
        
        # Only return location if confidence is high enough
        if best_location[1] > 0.35:
            return best_location[0]
        return "unknown"
        
    except Exception as e:
        logger.error(f"Error in location detection: {str(e)}")
        return "unknown"

def analyze_prop_usage(props):
    """Analyze how props are used with improved error handling"""
    try:
        if not props:
            return []
            
        # Convert to list if set was passed
        props_list = list(props) if isinstance(props, set) else props
        
        # Count prop frequencies
        prop_counter = Counter(props_list)
        
        # Get consistently used props (present in > 20% of frames)
        threshold = len(props_list) * 0.2
        main_props = [
            prop for prop, count in prop_counter.most_common()
            if count > threshold
        ]
        
        return main_props[:5]  # Limit to top 5 most common props
        
    except Exception as e:
        logger.warning(f"Error analyzing prop usage: {str(e)}")
        return []

def analyze_setting_patterns(settings):
    """Analyze patterns in setting usage"""
    if not settings:
        return None
        
    # Count setting frequencies
    setting_types = Counter([s.get('atmosphere') for s in settings if s])
    
    # Get primary setting
    if setting_types:
        primary_setting = setting_types.most_common(1)[0][0]
        return primary_setting
    return None

def detect_frame_effects_advanced(frame):
    """Detect advanced video effects"""
    effects = []
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Check for various effects
    if detect_color_grading(hsv):
        effects.append('color_graded')
    if detect_vignette(lab):
        effects.append('vignette')
    if detect_blur_effect(frame):
        effects.append('blur_effect')
    if detect_chromatic_aberration(frame):
        effects.append('chromatic_aberration')
        
    return effects

def calculate_frame_difference(prev_frame, curr_frame):
    """Calculate difference between frames"""
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    return np.mean(diff)

def detect_transition_type(prev_frame, curr_frame):
    """Detect type of transition between frames"""
    diff = cv2.absdiff(prev_frame, curr_frame)
    mean_diff = np.mean(diff)
    
    if mean_diff > 100:  # Substantial change
        if is_fade_transition(prev_frame, curr_frame):
            return 'fade'
        elif is_wipe_transition(diff):
            return 'wipe'
        return 'cut'
    return None

def analyze_editing_rhythm(frame_changes, fps):
    """Analyze rhythm of frame changes"""
    if not frame_changes:
        return 'static'
        
    # Calculate change frequency
    changes_per_second = len([c for c in frame_changes if c > 30]) / (len(frame_changes) / fps)
    
    if changes_per_second > 2.0:
        return 'fast_paced'
    elif changes_per_second > 0.5:
        return 'moderate'
    return 'slow_paced'

def calculate_engagement_effectiveness(engagement_features):
    """Calculate overall engagement effectiveness score"""
    scores = []
    
    # Visual engagement (40%)
    if engagement_features['visual']:
        visual_score = (
            bool(engagement_features['visual']['text_overlay']) * 0.4 +
            len(engagement_features['visual']['effects']) * 0.3 +
            len(engagement_features['visual']['hooks']) * 0.3
        ) * 0.4
        scores.append(visual_score)
    
    # Audio engagement (30%)
    if engagement_features['audio']:
        audio_score = (
            bool(engagement_features['audio']['call_response']) * 0.3 +
            bool(engagement_features['audio']['repetition']) * 0.4 +
            min(len(engagement_features['audio']['hooks']) * 0.1, 0.3)
        ) * 0.3
        scores.append(audio_score)
    
    # Performance engagement (30%)
    if engagement_features['performance']:
        performance_score = (
            bool(engagement_features['performance']['direct_address']) * 0.4 +
            bool(engagement_features['performance']['gestures']) * 0.3 +
            float(engagement_features['performance']['emotional_range']) * 0.3
        ) * 0.3
        scores.append(performance_score)
    
    return sum(scores) / len(scores) if scores else 0.0

def detect_performance_engagement_enhanced(frames):
    """Enhanced performance engagement detection now accepting pre-extracted frames"""
    performance_metrics = []
    
    for frame in frames:
        try:
            # Analyze performance metrics
            metrics = analyze_performance_metrics(frame)
            if metrics:
                performance_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"Error analyzing frame: {str(e)}")
            continue
    
    return {
        'direct_address': analyze_direct_address(performance_metrics),
        'gestures': analyze_gesture_patterns(performance_metrics),
        'emotional_range': analyze_emotional_range(performance_metrics)
    }

def calculate_sync_scores(video_motion, onset_times, onset_frames):
    """Calculate audio-visual synchronization scores"""
    try:
        if not video_motion or not onset_times.size:
            return {
                'sync_score': 0.0,
                'description': "Insufficient data for sync analysis",
                'details': {'strong_sync_points': 0, 'weak_sync_points': 0}
            }
        
        # Convert motion times to same scale as onset times
        motion_times = np.array([m[0] for m in video_motion])
        motion_values = np.array([m[1] for m in video_motion])
        
        # Find matching points
        sync_points = []
        for onset_time in onset_times:
            # Find closest motion time
            idx = np.argmin(np.abs(motion_times - onset_time))
            if abs(motion_times[idx] - onset_time) < 0.1:  # Within 100ms
                sync_points.append((onset_time, motion_values[idx]))
        
        if not sync_points:
            return {
                'sync_score': 0.0,
                'description': "No sync points detected",
                'details': {'strong_sync_points': 0, 'weak_sync_points': 0}
            }
        
        # Calculate sync quality
        sync_scores = [s[1] for s in sync_points]
        strong_syncs = sum(1 for s in sync_scores if s > np.mean(sync_scores))
        weak_syncs = len(sync_scores) - strong_syncs
        
        sync_score = strong_syncs / (len(onset_frames) + 1e-6)
        
        return {
            'sync_score': float(sync_score),
            'description': "Good sync" if sync_score > 0.5 else "Poor sync",
            'details': {
                'strong_sync_points': strong_syncs,
                'weak_sync_points': weak_syncs
            }
        }
        
    except Exception as e:
        logger.warning(f"Error calculating sync scores: {str(e)}")
        return {
            'sync_score': 0.0,
            'description': "Error in sync analysis",
            'details': {'strong_sync_points': 0, 'weak_sync_points': 0}
        }

def analyze_performance_metrics(frame):
    """Analyze performance metrics with fixed error handling"""
    try:
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray):
            return None
            
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return None
            
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Initialize face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        metrics = {}
        if len(faces) > 0:
            # Get largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Validate face region
            if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                face_roi = frame_bgr[y:y+h, x:x+w]
                
                try:
                    emotions = DeepFace.analyze(
                        img_path=face_roi,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    
                    if isinstance(emotions, list) and len(emotions) > 0:
                        metrics['emotions'] = emotions[0].get('emotion', {})
                        metrics['position'] = (x, y, w, h)
                except:
                    metrics['emotions'] = {}
                    metrics['position'] = (x, y, w, h)
                    
        return metrics
        
    except Exception as e:
        logger.warning(f"Performance metrics error: {str(e)}")
        return None

def detect_performance_techniques(frame):
    """Detect specific performance techniques used"""
    try:
        techniques = {
            'detected_techniques': [],
            'confidence_scores': {},
            'performance_quality': 0.0
        }

        # Convert frame for CLIP analysis
        image = Image.fromarray(frame)
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)

        # Define performance techniques to check
        performance_categories = {
            'vocal_techniques': [
                "singing to camera",
                "lip syncing performance",
                "vocal expression",
                "mouth movements synchronized"
            ],
            'physical_performance': [
                "dancing to music",
                "choreographed moves",
                "freestyle movement",
                "rhythmic gestures"
            ],
            'camera_interaction': [
                "direct eye contact",
                "camera engagement",
                "facial expressions to camera",
                "performing to viewer"
            ],
            'energy_style': [
                "high energy performance",
                "laid back style",
                "dramatic expression",
                "subtle performance"
            ]
        }

        # Process each category
        for category, prompts in performance_categories.items():
            text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(1),
                    text_features.unsqueeze(0)
                )

                # Get highest scoring technique in category
                max_score = torch.max(similarities)
                if max_score > 0.25:
                    best_technique = prompts[torch.argmax(similarities)]
                    techniques['detected_techniques'].append(f"{category}:{best_technique}")
                    techniques['confidence_scores'][best_technique] = float(max_score)

        # Validate detected techniques
        validated_techniques = []
        for technique in techniques['detected_techniques']:
            category = technique.split(':')[0]
            if validate_technique(frame, category):
                validated_techniques.append(technique)

        # Calculate overall performance quality
        if validated_techniques:
            confidence_scores = [techniques['confidence_scores'][t.split(':')[1]] 
                               for t in validated_techniques]
            techniques['performance_quality'] = float(np.mean(confidence_scores))

        # Update with validated techniques only
        techniques['detected_techniques'] = validated_techniques

        return techniques

    except Exception as e:
        logger.warning(f"Error in technique detection: {str(e)}")
        return {'detected_techniques': [], 'confidence_scores': {}, 'performance_quality': 0.0}

def validate_technique(frame, category):
    """Validate detected performance technique"""
    try:
        if category == 'vocal_techniques':
            return validate_vocal_technique(frame)
        elif category == 'physical_performance':
            return validate_physical_performance(frame)
        elif category == 'camera_interaction':
            return validate_camera_interaction(frame)
        elif category == 'energy_style':
            return validate_energy_style(frame)
        return False

    except Exception as e:
        logger.debug(f"Error in technique validation: {str(e)}")
        return False

def validate_vocal_technique(frame):
    """Validate vocal performance technique"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            
            # Check mouth region (lower third of face)
            mouth_region = face_roi[2*h//3:, :]
            
            # Calculate motion in mouth region
            edges = cv2.Canny(mouth_region, 100, 200)
            movement = np.mean(edges) / 255.0
            
            return movement > 0.15
            
        return False
        
    except Exception as e:
        logger.debug(f"Error in vocal technique validation: {str(e)}")
        return False

def validate_physical_performance(frame):
    """Validate physical performance technique"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion metrics
        edges = cv2.Canny(gray, 100, 200)
        
        # Look for significant movement
        movement_intensity = np.mean(edges) / 255.0
        
        # Check different regions for movement
        h, w = edges.shape
        upper_body = edges[:2*h//3, :]
        lower_body = edges[2*h//3:, :]
        
        upper_movement = np.mean(upper_body) / 255.0
        lower_movement = np.mean(lower_body) / 255.0
        
        return movement_intensity > 0.2 or upper_movement > 0.15 or lower_movement > 0.15
        
    except Exception as e:
        logger.debug(f"Error in physical performance validation: {str(e)}")
        return False

def validate_camera_interaction(frame):
    """Validate camera interaction technique"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face and eyes
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Check face position (centered)
            face_center_x = x + w/2
            frame_center_x = frame.shape[1]/2
            is_centered = abs(face_center_x - frame_center_x) < frame.shape[1]/6
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(face_roi)
            has_eye_contact = len(eyes) >= 2
            
            return is_centered and has_eye_contact
            
        return False
        
    except Exception as e:
        logger.debug(f"Error in camera interaction validation: {str(e)}")
        return False

def validate_energy_style(frame):
    """Validate energy style technique"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion and edge metrics
        edges = cv2.Canny(gray, 100, 200)
        movement_intensity = np.mean(edges) / 255.0
        
        # Calculate contrast
        contrast = np.std(gray) / 255.0
        
        # Combined energy score
        energy_score = (movement_intensity + contrast) / 2
        
        return energy_score > 0.15
        
    except Exception as e:
        logger.debug(f"Error in energy style validation: {str(e)}")
        return False


def analyze_direct_address_enhanced(metrics):
    """Enhanced analysis of direct address with additional metrics"""
    try:
        if not metrics:
            return {
                'direct_address': False,
                'confidence': 0.0,
                'consistency': 0.0
            }

        center_frames = 0
        face_sizes = []
        positions = []
        total_frames = len(metrics)

        for m in metrics:
            if not m or 'position' not in m:
                continue

            x, y, w, h = m['position']
            
            # Get frame dimensions
            if 'frame_size' in m:
                frame_width = m['frame_size'].get('width', 1)
                frame_height = m['frame_size'].get('height', 1)
            else:
                continue

            # Calculate relative positions and sizes
            face_center_x = (x + w/2) / frame_width
            face_center_y = (y + h/2) / frame_height
            face_size_ratio = (w * h) / (frame_width * frame_height)

            # Track metrics
            positions.append((face_center_x, face_center_y))
            face_sizes.append(face_size_ratio)

            # Check for direct address
            if (0.3 < face_center_x < 0.7 and 
                0.2 < face_center_y < 0.8 and 
                face_size_ratio > 0.05):
                center_frames += 1

        # Calculate metrics
        direct_address = center_frames > total_frames * 0.3 if total_frames > 0 else False
        
        # Calculate confidence based on face size and position consistency
        avg_face_size = np.mean(face_sizes) if face_sizes else 0
        position_std = np.std([x for x, y in positions]) if positions else 1
        
        confidence = min(1.0, (avg_face_size * 5 + (1 - position_std)) / 2)
        
        # Calculate consistency
        consistency = center_frames / total_frames if total_frames > 0 else 0

        return {
            'direct_address': direct_address,
            'confidence': float(confidence),
            'consistency': float(consistency),
            'face_size_avg': float(avg_face_size),
            'position_stability': float(1 - position_std)
        }

    except Exception as e:
        logger.warning(f"Error in enhanced direct address analysis: {str(e)}")
        return {
            'direct_address': False,
            'confidence': 0.0,
            'consistency': 0.0,
            'face_size_avg': 0.0,
            'position_stability': 0.0
        }
def analyze_gesture_patterns_enhanced(metrics):
    """Enhanced analysis of gesture patterns"""
    if not metrics:
        return False
    
    gesture_frames = 0
    for m in metrics:
        if 'pose' in m:
            # Check for active gestures
            if m['pose'] in ['gesturing', 'dancing', 'dynamic']:
                gesture_frames += 1
                
            # Check for significant pose changes
            if i > 0 and 'pose' in metrics[i-1]:
                if m['pose'] != metrics[i-1]['pose']:
                    gesture_frames += 1
    
    return gesture_frames > len(metrics) * 0.3  # Significant gestures in >30% of frames

def detect_gaze_direction(face_roi):
    """Detect gaze direction from face ROI"""
    # Convert to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(eyes) >= 2:
        # Calculate eye positions relative to face
        eye_positions = [(x + w/2, y + h/2) for (x, y, w, h) in eyes[:2]]
        avg_x = np.mean([pos[0] for pos in eye_positions])
        
        # Determine gaze direction based on eye positions
        face_center = face_roi.shape[1] / 2
        if abs(avg_x - face_center) < face_roi.shape[1] * 0.1:
            return 'forward'
        elif avg_x < face_center:
            return 'left'
        else:
            return 'right'
    return 'unknown'

def detect_performance_pose(frame):
    """Detect performance-related pose"""
    # Get pose keypoints
    pose_detector = cv2.dnn.readNetFromTensorflow('pose_estimation_model.pb')
    frameHeight, frameWidth = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    pose_detector.setInput(blob)
    output = pose_detector.forward()
    
    # Extract key points
    points = []
    for i in range(output.shape[1]):
        heatMap = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        if conf > 0.1:
            x = int((frameWidth * point[0]) / output.shape[3])
            y = int((frameHeight * point[1]) / output.shape[2])
            points.append((x, y))
        else:
            points.append(None)
    
    # Analyze pose type
    if len(points) >= 15:  # Enough points detected
        if detect_dancing_pose(points):
            return 'dancing'
        elif detect_gesturing_pose(points):
            return 'gesturing'
        elif detect_dynamic_pose(points):
            return 'dynamic'
    return 'static'

def detect_dancing_pose(points):
    """Detect if pose suggests dancing"""
    if not all(points[i] for i in [7, 8, 11, 12]):  # Need arms and legs
        return False
    
    # Calculate joint angles and movements
    arm_angles = calculate_limb_angles(points, [5, 6, 7, 8])  # Shoulders to hands
    leg_angles = calculate_limb_angles(points, [11, 12, 13, 14])  # Hips to feet
    
    # Check for dance-like pose
    return (
        np.std(arm_angles) > 30 and  # Variable arm positions
        np.std(leg_angles) > 20  # Variable leg positions
    )

def detect_gesturing_pose(points):
    """Detect if pose suggests gesturing"""
    if not all(points[i] for i in [7, 8]):  # Need hands
        return False
    
    # Calculate hand positions relative to body
    hand_positions = np.array([points[7], points[8]])
    shoulder_positions = np.array([points[5], points[6]])
    
    # Check for hands away from body
    distances = np.linalg.norm(hand_positions - shoulder_positions, axis=1)
    return np.any(distances > 50)  # Hands extended from body

def detect_dynamic_pose(points):
    """Detect if pose suggests dynamic movement"""
    if len(points) < 15:
        return False
    
    # Calculate overall pose spread
    valid_points = [p for p in points if p is not None]
    if len(valid_points) < 2:
        return False
        
    points_array = np.array(valid_points)
    spread = np.std(points_array, axis=0)
    
    return np.mean(spread) > 30  # Significant pose spread

def calculate_limb_angles(points, indices):
    """Calculate angles between limb segments"""
    angles = []
    for i in range(0, len(indices)-1):
        if points[indices[i]] and points[indices[i+1]]:
            p1 = np.array(points[indices[i]])
            p2 = np.array(points[indices[i+1]])
            angle = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
            angles.append(np.degrees(angle))
    return angles

def calculate_emotional_range(emotion_sequence):
    """Calculate the range of emotional expression"""
    try:
        if not emotion_sequence:
            return 0.0
            
        # Track emotion variations
        variations = []
        for i in range(1, len(emotion_sequence)):
            prev_emotions = emotion_sequence[i-1]
            curr_emotions = emotion_sequence[i]
            
            # Calculate change in emotional state
            total_change = 0
            for emotion in prev_emotions:
                if emotion in curr_emotions:
                    change = abs(prev_emotions[emotion] - curr_emotions[emotion])
                    total_change += change
                    
            variations.append(total_change / 100.0)  # Normalize to 0-1
            
        return np.mean(variations) if variations else 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating emotional range: {str(e)}")
        return 0.0


def detect_body_pose(frame):
    """Detect body pose for emotional analysis"""
    pose_detector = cv2.dnn.readNetFromTensorflow('pose_estimation_model.pb')
    frameHeight, frameWidth = frame.shape[:2]
    
    # Prepare frame for pose detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    pose_detector.setInput(blob)
    output = pose_detector.forward()
    
    # Extract key points
    points = []
    for i in range(output.shape[1]):
        heatMap = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]
        points.append((int(x), int(y)) if conf > 0.1 else None)
    
    return points

def analyze_pose_emotion(pose_points):
    """Analyze emotional indicators from body pose"""
    if not pose_points:
        return None
        
    # Analyze pose characteristics
    head_tilt = calculate_head_tilt(pose_points)
    shoulder_tension = calculate_shoulder_tension(pose_points)
    arm_position = analyze_arm_position(pose_points)
    
    # Map poses to emotional indicators
    emotions = {
        'confidence': 0,
        'enthusiasm': 0,
        'tension': 0,
        'relaxation': 0
    }
    
    # Update based on pose analysis
    if head_tilt > 30:
        emotions['confidence'] += 0.5
    if shoulder_tension > 0.7:
        emotions['tension'] += 0.6
    if arm_position == 'open':
        emotions['enthusiasm'] += 0.7
        emotions['confidence'] += 0.3
    elif arm_position == 'closed':
        emotions['tension'] += 0.4
    
    return emotions

def analyze_scene_emotion(frame):
    """Analyze emotional context from scene"""
    try:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analyze color mood
        color_mood = analyze_color_mood(hsv)
        
        # Analyze lighting
        lighting_mood = analyze_lighting_mood(frame)
        
        # Analyze composition
        composition_mood = analyze_composition_mood(frame)
        
        # Return all moods without multiplication
        return {
            'color_mood': color_mood,
            'lighting_mood': lighting_mood,
            'composition_mood': composition_mood,
            'overall_mood': None  # Calculate this differently if needed
        }
        
    except Exception as e:
        logger.warning(f"Error in scene emotion analysis: {str(e)}")
        return None

def combine_emotion_signals(facial_emotions, body_emotions, scene_emotions, emotion_changes):
    """Combine different emotional signals into cohesive profile"""
    combined = {}
    
    # Weight different signals
    weights = {
        'facial': 0.5,
        'body': 0.3,
        'scene': 0.2
    }
    
    # Process facial emotions
    for emotion, score in facial_emotions.items():
        combined[emotion] = score * weights['facial']
    
    # Add body language
    for body_data in body_emotions:
        if body_data:
            for emotion, score in body_data.items():
                if emotion in combined:
                    combined[emotion] += score * weights['body']
                else:
                    combined[emotion] = score * weights['body']
    
    # Add scene context
    for scene_data in scene_emotions:
        if scene_data:
            scene_score = calculate_scene_emotion_score(scene_data)
            for emotion, score in scene_score.items():
                if emotion in combined:
                    combined[emotion] += score * weights['scene']
                else:
                    combined[emotion] = score * weights['scene']
    
    return combined

def generate_emotion_description(emotion_profile):
    """Generate detailed emotion description"""
    if not emotion_profile:
        return "Unable to detect clear emotions"
        
    description_parts = []
    
    # Get primary and secondary emotions
    sorted_emotions = sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True)
    primary_emotion = sorted_emotions[0]
    
    # Determine intensity
    if primary_emotion[1] > 0.7:
        intensity = "intensely"
    elif primary_emotion[1] > 0.5:
        intensity = "clearly"
    else:
        intensity = "somewhat"
    
    # Build description
    description_parts.append(f"{intensity} {primary_emotion[0]}")
    
    # Add secondary emotions
    if len(sorted_emotions) > 1:
        secondary = sorted_emotions[1]
        if secondary[1] > 0.3:
            description_parts.append(f"with {secondary[0]} undertones")
    
    # Add emotional progression if significant changes
    emotion_progression = analyze_emotion_progression(emotion_profile)
    if emotion_progression:
        description_parts.append(emotion_progression)
    
    return ", ".join(description_parts)


def analyze_scene_and_location(frames, audio_features):
   """Enhanced scene and location analysis"""
   try:
       if not frames:
           logger.warning("No frames provided for scene analysis")
           return get_default_scene_analysis()

       def debug_dict_update(name, original, update_with):
           """Debug dictionary updates"""
           logger.debug(f"Updating {name}")
           logger.debug(f"Original: {original}")
           logger.debug(f"Updating with: {update_with}")
           
       # Initialize with default scene data
       scene_data = get_default_scene_analysis()  # Start with defaults
           
       # Track scene elements across frames
       locations = []
       lighting_states = []
       compositions = []
       visual_elements = []
       prev_frame = None
       
       # Process frames with progress logging
       logger.info(f"Starting scene analysis on {len(frames)} frames")
       for frame_idx, frame in enumerate(frames):
           try:
               # Validate frame
               if not validate_frame(frame):
                   logger.debug(f"Frame {frame_idx} failed validation")
                   continue
                   
               # Location detection
               location = detect_music_video_location(frame)
               if location:
                   locations.append(location)
                   
               # Lighting analysis
               lighting = analyze_lighting_environment(frame)
               if lighting:
                   lighting_states.append(lighting)
                   
               # Composition analysis
               composition = analyze_scene_composition(frame)
               if composition:
                   compositions.append(composition)
                   
               # Visual elements detection
               elements = detect_setting_elements(frame)
               if elements:
                   visual_elements.extend(elements)
                   
               # Detect transitions
               if prev_frame is not None:
                   transition = detect_scene_transition(prev_frame, frame)
                   if transition:
                       scene_data['transitions'].append({
                           'frame': frame_idx,
                           'type': transition
                       })
                       
               prev_frame = frame.copy()
               
               # Log progress every 10 frames
               if frame_idx % 10 == 0:
                   logger.info(f"Processed {frame_idx}/{len(frames)} frames")
                   
           except Exception as e:
               logger.warning(f"Error processing frame {frame_idx}: {str(e)}\n{traceback.format_exc()}")
               continue
               
       # Process collected data with error handling
       try:
           if locations:
               location_counter = Counter(locations)
               scene_data['technical_data'].update({
                   'primary_location': location_counter.most_common(1)[0][0]
               })
               logger.info(f"Detected primary location: {scene_data['technical_data']['primary_location']}")
       except Exception as e:
           logger.error(f"Error processing locations: {str(e)}")
           
       try:
           if lighting_states:
               lighting_summary = summarize_lighting_states(lighting_states)
               debug_dict_update('lighting', scene_data['lighting'], lighting_summary)
               scene_data['lighting'].clear()
               scene_data['lighting'].update(lighting_summary)
               scene_data['technical_data']['lighting_summary'] = scene_data['lighting'].get('lighting_type', "unknown")
               logger.info(f"Analyzed lighting patterns: {scene_data['lighting']['lighting_type']}")
       except Exception as e:
           logger.error(f"Error processing lighting states: {str(e)}")
           
       try:
           if compositions:
               composition_metrics = {
                   'symmetry_score': np.mean([c.get('symmetry', 0) for c in compositions]),
                   'rule_of_thirds_score': np.mean([c.get('rule_of_thirds', 0) for c in compositions]),
                   'balance_score': np.mean([c.get('balance', 0) for c in compositions]),
                   'depth_score': np.mean([c.get('depth_score', 0) for c in compositions])
               }
               debug_dict_update('composition', scene_data['composition'], composition_metrics)
               scene_data['composition'].clear()
               scene_data['composition'].update(composition_metrics)
               logger.info("Calculated composition scores")
       except Exception as e:
           logger.error(f"Error processing compositions: {str(e)}")
           
       try:
           if visual_elements:
               unique_elements = list(set(visual_elements))
               scene_data['setting'].update({
                   'visual_elements': unique_elements
               })
               scene_data['technical_data']['setting_type'] = determine_setting_type(unique_elements)
               logger.info(f"Detected {len(unique_elements)} unique visual elements")
       except Exception as e:
           logger.error(f"Error processing visual elements: {str(e)}")
           
       # Update scene changes count
       scene_data['technical_data']['scene_changes'] = len(scene_data['transitions'])
       
       # Generate final description
       try:
           scene_data['description'] = generate_scene_description(
               scene_data['technical_data'],
               scene_data['lighting'],
               scene_data['setting']
           )
       except Exception as e:
           logger.error(f"Error generating scene description: {str(e)}")
           scene_data['description'] = "Error generating scene description"
           
       logger.info("Scene analysis completed successfully")
       return scene_data
       
   except Exception as e:
       logger.error(f"Error in scene analysis: {str(e)}\n{traceback.format_exc()}")
       return get_default_scene_analysis()

def combine_moods(moods, weights):
    """Safely combine moods with weights"""
    combined = {}
    
    for mood_type, weight in weights.items():
        if mood_type in moods and isinstance(moods[mood_type], dict):
            mood_values = moods[mood_type].get('moods', {})
            for mood, value in mood_values.items():
                if isinstance(value, (int, float)):
                    if mood not in combined:
                        combined[mood] = 0.0
                    combined[mood] += float(value) * weight
    
    return combined

def detect_music_video_location(frame):
    """Detect location with improved array handling"""
    try:
        if not isinstance(frame, np.ndarray):
            return "unknown"

        # Location confidence scores
        location_scores = {
            'studio': 0.0,
            'outdoor': 0.0,
            'venue': 0.0,
            'home': 0.0
        }

        # Studio detection
        try:
            studio_conf = detect_studio_setting(frame)
            location_scores['studio'] = float(studio_conf)
        except Exception:
            pass

        # Outdoor detection
        try:
            outdoor_conf = detect_outdoor_setting(frame)
            location_scores['outdoor'] = float(outdoor_conf)
        except Exception:
            pass

        # Venue detection
        try:
            venue_conf = detect_venue_setting(frame)
            location_scores['venue'] = float(venue_conf)
        except Exception:
            pass

        # Home detection
        try:
            home_conf = detect_home_setting(frame)
            location_scores['home'] = float(home_conf)
        except Exception:
            pass

        # Get location with highest confidence
        best_location = max(location_scores.items(), key=lambda x: x[1])
        
        # Only return location if confidence is high enough
        if best_location[1] > 0.3:
            return best_location[0]
        return "unknown"

    except Exception as e:
        logger.debug(f"Location detection error: {str(e)}")
        return "unknown"


def analyze_detailed_lighting(frame):
    """Enhanced lighting analysis for music video contexts"""
    lighting_data = {
        'intensity': None,
        'type': None,
        'effects': [],
        'mood': None,
        'technical_quality': None
    }
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Analyze lighting intensity
    brightness = np.mean(lab[:,:,0])
    if brightness > 180:
        lighting_data['intensity'] = "very bright"
    elif brightness > 140:
        lighting_data['intensity'] = "well-lit"
    elif brightness > 100:
        lighting_data['intensity'] = "moderately lit"
    else:
        lighting_data['intensity'] = "dimly lit"
    
    # Detect lighting type
    lighting_data['type'] = detect_lighting_type(frame)
    
    # Analyze lighting effects
    lighting_data['effects'] = detect_lighting_effects(frame)
    
    # Analyze lighting mood
    lighting_data['mood'] = analyze_lighting_mood(frame)
    
    # Assess technical quality
    lighting_data['technical_quality'] = assess_lighting_quality(frame)
    
    return lighting_data

def generate_scene_description(technical_data, lighting, setting):
    """Generate detailed scene description"""
    try:
        description_parts = []
        
        # Add location if known
        if technical_data['primary_location'] != "unknown":
            description_parts.append(technical_data['primary_location'])
            
        # Add lighting characteristics
        if lighting['lighting_type'] != "unknown":
            light_desc = f"{lighting['lighting_type']}"
            if lighting['color_temperature'] != "unknown":
                light_desc += f" {lighting['color_temperature']}"
            description_parts.append(light_desc)
            
        # Add setting type
        if technical_data['setting_type'] != "unknown":
            description_parts.append(technical_data['setting_type'])
            
        # Add scene changes if significant
        if technical_data['scene_changes'] > 0:
            description_parts.append(f"with {technical_data['scene_changes']} scene changes")
            
        return " | ".join(description_parts) if description_parts else "Basic scene"
        
    except Exception as e:
        logger.debug(f"Error generating scene description: {str(e)}")
        return "Basic scene"

def determine_engagement_strategies(engagement_features):
    """Determine engagement strategies from features"""
    try:
        strategies = []
        
        # Visual strategies
        if 'visual' in engagement_features:
            visual = engagement_features['visual']
            if isinstance(visual, dict):
                if visual.get('text_overlay'):
                    strategies.append('text_overlay')
                if visual.get('effects'):
                    strategies.append('visual_effects')
                if visual.get('hooks'):
                    strategies.append('visual_hooks')
                
        # Audio strategies
        if 'audio' in engagement_features:
            audio = engagement_features['audio']
            if isinstance(audio, dict):
                if audio.get('call_response'):
                    strategies.append('call_and_response')
                if audio.get('repetition'):
                    strategies.append('repetition')
                if audio.get('hooks'):
                    strategies.append('musical_hooks')
                
        # Performance strategies
        if 'performance' in engagement_features:
            perf = engagement_features['performance']
            if isinstance(perf, dict):
                if perf.get('direct_address'):
                    strategies.append('direct_address')
                if perf.get('gestures'):
                    strategies.append('gestures')
                
        return strategies
        
    except Exception as e:
        logger.error(f"Error determining engagement strategies: {str(e)}")
        return []


def detect_peaks(x):
    """Helper function to detect peaks in a signal"""
    peaks = []
    for i in range(1, len(x)-1):
        if x[i-1] < x[i] and x[i] > x[i+1]:
            peaks.append(i)
    return peaks

def detect_performance_strategies(frames):
    """Detect performance-based engagement strategies"""
    try:
        strategies = {
            'strategies': [],
            'consistency': 0.0,
            'details': {}
        }
        
        direct_address_frames = 0
        gesture_frames = 0
        emotion_scores = []
        total_frames = len(frames)
        
        for frame in frames:
            metrics = analyze_performance_metrics(frame)
            if metrics:
                # Check direct address
                if analyze_direct_address_enhanced(metrics).get('direct_address', False):
                    direct_address_frames += 1
                    
                # Check gestures
                if detect_performance_gestures(frame):
                    gesture_frames += 1
                    
                # Track emotional expression
                if metrics.get('emotions'):
                    emotion_scores.append(max(metrics['emotions'].values()))
                    
        # Calculate strategy presence
        if total_frames > 0:
            direct_address_ratio = direct_address_frames / total_frames
            gesture_ratio = gesture_frames / total_frames
            
            if direct_address_ratio > 0.3:
                strategies['strategies'].append('direct_address')
            if gesture_ratio > 0.25:
                strategies['strategies'].append('gestural_engagement')
                
            if emotion_scores:
                emotion_range = max(emotion_scores) - min(emotion_scores)
                if emotion_range > 30:
                    strategies['strategies'].append('emotional_variety')
                    
            strategies['consistency'] = (direct_address_ratio + gesture_ratio) / 2
            strategies['details'] = {
                'direct_address_frequency': float(direct_address_ratio),
                'gesture_frequency': float(gesture_ratio),
                'emotional_range': float(emotion_range) if emotion_scores else 0.0
            }
            
        return strategies
        
    except Exception as e:
        logger.debug(f"Error detecting performance strategies: {str(e)}")
        return {'strategies': [], 'consistency': 0.0, 'details': {}}


def detect_content_strategies(frames):
    """Detect content-based engagement strategies"""
    try:
        strategies = {
            'strategies': [],
            'consistency': 0.0,
            'details': {}
        }
        
        # Track content patterns
        scene_changes = 0
        pattern_interrupts = 0
        storytelling_elements = []
        prev_frame = None
        total_frames = len(frames)
        
        for frame in frames:
            # Detect scene transitions
            if prev_frame is not None:
                if detect_scene_transition(prev_frame, frame):
                    scene_changes += 1
                    
            # Detect pattern interrupts
            if detect_pattern_interrupt(frame):
                pattern_interrupts += 1
                
            # Detect storytelling elements
            elements = detect_storytelling_elements(frame)
            if elements:
                storytelling_elements.extend(elements)
                
            prev_frame = frame.copy()
            
        # Analyze content strategies
        if total_frames > 0:
            if scene_changes > total_frames * 0.1:
                strategies['strategies'].append('dynamic_pacing')
            if pattern_interrupts > total_frames * 0.05:
                strategies['strategies'].append('pattern_interrupts')
                
            if storytelling_elements:
                strategies['strategies'].append('narrative_elements')
                
            strategies['consistency'] = len(strategies['strategies']) / 3
            strategies['details'] = {
                'scene_change_frequency': float(scene_changes / total_frames),
                'pattern_interrupt_frequency': float(pattern_interrupts / total_frames),
                'storytelling_elements': list(set(storytelling_elements))
            }
            
        return strategies
        
    except Exception as e:
        logger.debug(f"Error detecting content strategies: {str(e)}")
        return {'strategies': [], 'consistency': 0.0, 'details': {}}


def detect_editing_patterns(video_path):
    """Analyze video editing patterns and techniques"""
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    scene_changes = []
    transition_types = []
    effect_sequences = []
    
    prev_frame = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if prev_frame is not None:
            # Detect scene changes
            diff = cv2.absdiff(prev_frame, frame)
            change_score = np.mean(diff)
            
            if change_score > 30:  # Threshold for scene change
                transition_type = classify_transition(prev_frame, frame)
                scene_changes.append(frame_count)
                transition_types.append(transition_type)
            
            # Detect effects
            effects = detect_frame_effects(frame)
            if effects:
                effect_sequences.append((frame_count, effects))
        
        prev_frame = frame.copy()
        frame_count += 1
    
    cap.release()
    
    # Analyze editing patterns
    edit_rhythm = analyze_edit_rhythm(scene_changes, frame_rate)
    effect_patterns = analyze_effect_sequences(effect_sequences, frame_rate)
    
    return {
        'transitions': len(scene_changes),
        'transition_types': summarize_transitions(transition_types),
        'edit_rhythm': edit_rhythm,
        'effects': effect_patterns
    }

def analyze_content_strategies(frames):
    """Analyze content-based engagement strategies"""
    content_elements = {
        'text_elements': {},
        'trends': [],
        'storytelling': [],
        'calls_to_action': []
    }
    
    frame_count = 0
    
    for frame in frames:
        try:
            # Detect and analyze text
            text_regions = detect_text_enhanced(frame)
            if text_regions:
                # Analyze text content and style
                text_analysis = analyze_text_style_and_content(text_regions, frame)
                if text_analysis:
                    content_type = text_analysis['type']
                    content_elements['text_elements'].setdefault(content_type, {
                        'content': [],
                        'style': set(),
                        'timing': []
                    })
                    content_elements['text_elements'][content_type]['content'].append(text_analysis['content'])
                    content_elements['text_elements'][content_type]['style'].update(text_analysis['style'])
                    content_elements['text_elements'][content_type]['timing'].append(frame_count)
            
            # Detect trending elements
            trends = detect_trending_elements(frame)
            if trends:
                content_elements['trends'].extend(trends)
            
            # Analyze storytelling elements
            story_elements = detect_storytelling_elements(frame)
            if story_elements:
                content_elements['storytelling'].extend(story_elements)
                
            frame_count += 1
            
        except Exception as e:
            logger.warning(f"Error analyzing frame {frame_count}: {str(e)}")
            continue
    
    # Process and summarize content strategies
    content_elements['trends'] = list(set(content_elements['trends']))
    content_elements['storytelling'] = summarize_storytelling(content_elements['storytelling'])
    
    return content_elements

def analyze_advanced_editing(frames):
    """Analyze advanced editing patterns and techniques"""
    editing_patterns = {
        'transitions': [],
        'effects': [],
        'rhythm': [],
        'patterns': [],
        'syncs': []
    }
    
    prev_frame = None
    frame_changes = []
    frame_count = 0
    
    for frame in frames:
        try:
            if prev_frame is not None:
                # Detect transition type
                transition = detect_transition_type(prev_frame, frame)
                if transition:
                    editing_patterns['transitions'].append(transition)
                
                # Analyze editing rhythm
                change_magnitude = calculate_frame_difference(prev_frame, frame)
                frame_changes.append(change_magnitude)
                
                # Detect effects
                effects = detect_frame_effects_advanced(frame)
                if effects:
                    editing_patterns['effects'].extend(effects)
            
            prev_frame = frame.copy()
            frame_count += 1
            
        except Exception as e:
            logger.warning(f"Error analyzing frame {frame_count}: {str(e)}")
            continue
    
    # Calculate effective frame rate (assuming 30fps as default)
    effective_fps = 30
    
    # Analyze editing rhythm
    editing_patterns['rhythm'] = analyze_editing_rhythm(frame_changes, effective_fps)
    
    # Detect common patterns
    editing_patterns['patterns'] = detect_editing_patterns(
        editing_patterns['transitions'],
        editing_patterns['effects'],
        editing_patterns['rhythm']
    )
    
    return editing_patterns

def analyze_music_video_activity(frames, audio_features=None):
    """Activity analysis with improved error handling"""
    try:
        if not frames:
            return get_default_activity_analysis()
            
        performance_types = []
        locations = []
        props_used = set()
        
        for frame in frames:
            try:
                # Process frame safely
                frame = process_frame_safely(frame)
                if frame is None:
                    continue
                    
                # Detect performance type
                try:
                    perf_type = detect_performance_type(frame)
                    if perf_type and perf_type != "unknown":
                        performance_types.append(perf_type)
                except Exception as e:
                    logger.debug(f"Error detecting performance type: {str(e)}")
                    
                # Detect location
                try:
                    location = detect_detailed_location(frame)
                    if location:
                        locations.append(location)
                except Exception as e:
                    logger.debug(f"Error detecting location: {str(e)}")
                    
                # Detect props
                try:
                    props = detect_music_props(frame)
                    if props:
                        props_used.update(props)
                except Exception as e:
                    logger.debug(f"Error detecting props: {str(e)}")
                    
            except Exception as e:
                logger.debug(f"Error processing frame: {str(e)}")
                continue
                
        # Process collected data
        primary_activity = determine_primary_activity(performance_types) if performance_types else "unknown"
        primary_location = max(set(locations), key=locations.count) if locations else "unknown"
        key_props = list(props_used)
        
        # Generate description
        activity_description = generate_activity_description(
            primary_activity,
            primary_location,
            key_props
        )
        
        return {
            'activity': activity_description,
            'performance_type': primary_activity,
            'props': key_props,
            'location': primary_location
        }
        
    except Exception as e:
        logger.error(f"Error in activity analysis: {str(e)}")
        return get_default_activity_analysis()

def analyze_audio_visual_sync(frames, audio_path):
    """Analyze synchronization between video and audio with improved detection"""
    try:
        sync_data = {
            'sync_score': 0.0,
            'description': "No sync data available",
            'details': {
                'strong_sync_points': 0,
                'weak_sync_points': 0,
                'sync_patterns': [],
                'timing_quality': 0.0,
                'movement_alignment': 0.0
            }
        }

        if not frames or not audio_path:
            return sync_data

        # Load and process audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Analyze visual motion
        motion_scores = []
        frame_times = []
        prev_frame = None
        
        for i, frame in enumerate(frames):
            if prev_frame is not None:
                # Calculate frame difference
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate motion magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion = np.mean(magnitude)
                
                motion_scores.append(motion)
                frame_times.append(i / 30.0)  # Assuming 30fps
                
            prev_frame = frame.copy()

        if motion_scores and len(onset_times) > 0:
            # Find motion peaks
            motion_peaks = detect_motion_peaks(motion_scores)
            peak_times = [frame_times[i] for i in motion_peaks]
            
            # Match peaks with onsets
            sync_points = find_sync_points(onset_times, peak_times)
            
            if sync_points:
                # Analyze sync quality
                sync_quality = analyze_sync_quality(sync_points)
                movement_alignment = analyze_movement_alignment(motion_scores, onset_times, frame_times)
                
                # Update sync data
                sync_data['sync_score'] = (sync_quality['overall_score'] + movement_alignment) / 2
                sync_data['details']['strong_sync_points'] = sync_quality['strong_points']
                sync_data['details']['weak_sync_points'] = sync_quality['weak_points']
                sync_data['details']['sync_patterns'] = detect_sync_patterns(sync_points)
                sync_data['details']['timing_quality'] = sync_quality['timing_quality']
                sync_data['details']['movement_alignment'] = movement_alignment
                
                # Generate description
                sync_data['description'] = generate_sync_description(sync_data)

        return sync_data

    except Exception as e:
        logger.error(f"Error in sync analysis: {str(e)}")
        return {
            'sync_score': 0.0,
            'description': "Error in sync analysis",
            'details': {
                'strong_sync_points': 0,
                'weak_sync_points': 0,
                'sync_patterns': [],
                'timing_quality': 0.0,
                'movement_alignment': 0.0
            }
        }

def detect_motion_peaks(motion_scores):
    """Detect peaks in motion scores"""
    try:
        peaks = []
        window_size = 3
        
        for i in range(window_size, len(motion_scores) - window_size):
            window = motion_scores[i-window_size:i+window_size+1]
            if motion_scores[i] == max(window) and motion_scores[i] > np.mean(motion_scores):
                peaks.append(i)
                
        return peaks
        
    except Exception as e:
        logger.debug(f"Error detecting motion peaks: {str(e)}")
        return []

def find_sync_points(onset_times, peak_times):
    """Find points where audio and visual events align"""
    try:
        sync_points = []
        max_offset = 0.1  # 100ms tolerance
        
        for onset in onset_times:
            # Find closest peak
            if peak_times:
                closest_peak = min(peak_times, key=lambda x: abs(x - onset))
                time_diff = abs(closest_peak - onset)
                
                if time_diff < max_offset:
                    strength = 1.0 - (time_diff / max_offset)
                    sync_points.append({
                        'time': onset,
                        'offset': time_diff,
                        'strength': strength
                    })
                    
        return sync_points
        
    except Exception as e:
        logger.debug(f"Error finding sync points: {str(e)}")
        return []

def analyze_sync_quality(sync_points):
    """Analyze quality of synchronization"""
    try:
        if not sync_points:
            return {
                'overall_score': 0.0,
                'strong_points': 0,
                'weak_points': 0,
                'timing_quality': 0.0
            }
            
        # Categorize sync points
        strong_points = sum(1 for point in sync_points if point['strength'] > 0.7)
        weak_points = len(sync_points) - strong_points
        
        # Calculate timing quality
        timing_offsets = [point['offset'] for point in sync_points]
        timing_quality = 1.0 - (np.mean(timing_offsets) / 0.1)  # Normalize to 0-1
        
        # Calculate overall score
        overall_score = (
            (strong_points * 0.7 + weak_points * 0.3) / len(sync_points) +
            timing_quality
        ) / 2
        
        return {
            'overall_score': float(overall_score),
            'strong_points': strong_points,
            'weak_points': weak_points,
            'timing_quality': float(timing_quality)
        }
        
    except Exception as e:
        logger.debug(f"Error analyzing sync quality: {str(e)}")
        return {
            'overall_score': 0.0,
            'strong_points': 0,
            'weak_points': 0,
            'timing_quality': 0.0
        }

def analyze_movement_alignment(motion_scores, onset_times, frame_times):
    """Analyze how well movement aligns with audio rhythm"""
    try:
        if not motion_scores or not onset_times.size or not frame_times:
            return 0.0
            
        # Create motion signal interpolated to match audio timing
        motion_interp = np.interp(onset_times, frame_times, motion_scores)
        
        # Calculate correlation between motion and onsets
        correlation = np.corrcoef(motion_interp, np.ones_like(onset_times))[0,1]
        
        return float(max(0.0, correlation))
        
    except Exception as e:
        logger.debug(f"Error analyzing movement alignment: {str(e)}")
        return 0.0

def detect_sync_patterns(sync_points):
    """Detect patterns in synchronization"""
    try:
        if not sync_points:
            return []
            
        patterns = []
        strengths = [point['strength'] for point in sync_points]
        
        # Check for consistent timing
        if np.std(strengths) < 0.2:
            patterns.append('consistent_timing')
            
        # Check for alternating pattern
        if len(strengths) > 3:
            diffs = np.diff(strengths)
            if np.all(diffs[::2] > 0) and np.all(diffs[1::2] < 0):
                patterns.append('alternating_sync')
                
        # Check for increasing/decreasing quality
        if len(strengths) > 2:
            if all(a <= b for a, b in zip(strengths, strengths[1:])):
                patterns.append('improving_sync')
            elif all(a >= b for a, b in zip(strengths, strengths[1:])):
                patterns.append('degrading_sync')
                
        return patterns
        
    except Exception as e:
        logger.debug(f"Error detecting sync patterns: {str(e)}")
        return []

def generate_sync_description(sync_data):
    """Generate human-readable sync description"""
    try:
        if sync_data['sync_score'] == 0:
            return "No sync detected"
            
        quality_levels = {
            0.8: "Excellent",
            0.6: "Good",
            0.4: "Moderate",
            0.0: "Poor"
        }
        
        # Determine quality level
        sync_quality = "Poor"
        for threshold, description in quality_levels.items():
            if sync_data['sync_score'] >= threshold:
                sync_quality = description
                break
                
        # Add pattern information
        if sync_data['details']['sync_patterns']:
            patterns = sync_data['details']['sync_patterns']
            if 'consistent_timing' in patterns:
                sync_quality += " consistent"
            elif 'improving_sync' in patterns:
                sync_quality += " improving"
            elif 'degrading_sync' in patterns:
                sync_quality += " degrading"
                
        return f"{sync_quality} sync"
        
    except Exception as e:
        logger.debug(f"Error generating sync description: {str(e)}")
        return "Unknown sync quality"

def process_frame_safely(frame):
    """Process frame optimized for vertical TikTok videos with complete validation"""
    try:
        if not isinstance(frame, np.ndarray):
            logger.debug("Frame is not a numpy array")
            return None
            
        original_shape = frame.shape
        height, width = original_shape[:2]
        original_aspect = height / width
        
        logger.info(f"Original frame shape: {original_shape}, Aspect ratio: {original_aspect:.2f}")
        
        # Ensure frame has 3 dimensions and correct color channels
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.debug(f"Invalid frame dimensions: {frame.shape}")
            return None
            
        # Target dimensions for TikTok videos
        target_width = 576
        target_height = 1024
        
        # Calculate new dimensions maintaining aspect ratio
        if original_aspect >= 1.3:  # Vertical video
            if original_aspect > (16/9):  # Taller than 9:16
                new_height = target_height
                new_width = int(target_height / original_aspect)
            else:
                new_width = target_width
                new_height = int(target_width * original_aspect)
        else:  # Horizontal or square video
            new_width = target_width
            new_height = int(target_width * original_aspect)
            
        # Ensure minimum dimensions
        if new_width < 480 or new_height < 480:
            new_width = max(480, new_width)
            new_height = max(480, new_height)
            
        try:
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized frame from {original_shape} to {frame.shape}")
        except Exception as e:
            logger.error(f"Resize operation failed: {str(e)}")
            return None
            
        # Ensure correct data type
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        # Final validation check
        if not validate_frame(frame):
            logger.debug("Frame failed final validation")
            return None
            
        return frame
        
    except Exception as e:
        logger.error(f"Error in process_frame_safely: {str(e)}")
        return None

def analyze_engagement_strategies(frames, audio_features):
   """Analyze comprehensive engagement strategies"""
   try:
       if not frames:
           return get_default_engagement_analysis()
           
       engagement_data = get_default_engagement_analysis()  # Start with default
       
       # Get strategy data
       visual_strat = detect_visual_strategies(frames)
       audio_strat = detect_audio_strategies(audio_features) if audio_features else {}
       perf_strat = detect_performance_strategies(frames)
       content_strat = detect_content_strategies(frames)
       
       # Update instead of assign
       engagement_data['visual_strategies'].update(visual_strat)
       engagement_data['audio_strategies'].update(audio_strat)
       engagement_data['performance_strategies'].update(perf_strat)
       engagement_data['content_strategies'].update(content_strat)
       
       # Reset metrics before calculating
       engagement_data['engagement_metrics'] = {
           'consistency': 0.0,
           'variety': 0.0,
           'effectiveness': 0.0
       }
       
       # Aggregate strategies safely
       all_strategies = []
       
       # Visual strategies
       if engagement_data['visual_strategies'].get('strategies'):
           all_strategies.extend(engagement_data['visual_strategies'].get('strategies', []))
           
       # Audio strategies  
       if engagement_data['audio_strategies'].get('strategies'):
           all_strategies.extend(engagement_data['audio_strategies'].get('strategies', []))
           
       # Performance strategies
       if engagement_data['performance_strategies'].get('strategies'):
           all_strategies.extend(engagement_data['performance_strategies'].get('strategies', []))
           
       # Content strategies
       if engagement_data['content_strategies'].get('strategies'):
           all_strategies.extend(engagement_data['content_strategies'].get('strategies', []))
           
       # Calculate consistency using accumulator
       consistency_sum = 0.0
       consistency_count = 0
       
       for strat_type in ['visual_strategies', 'audio_strategies', 'performance_strategies', 'content_strategies']:
           if engagement_data[strat_type].get('strategies'):
               consistency_sum += engagement_data[strat_type].get('consistency', 0)
               consistency_count += 1
       
       # Calculate overall metrics
       if consistency_count > 0:
           engagement_data['engagement_metrics']['consistency'] = consistency_sum / consistency_count
           
       strategy_types = len(set(all_strategies))
       total_strategies = len(all_strategies)
       
       if total_strategies > 0:
           engagement_data['engagement_metrics']['variety'] = strategy_types / max(8, total_strategies)
           engagement_data['engagement_metrics']['effectiveness'] = (
               engagement_data['engagement_metrics']['consistency'] * 0.6 +
               engagement_data['engagement_metrics']['variety'] * 0.4
           )
           
       # Determine primary focus and store strategies
       engagement_data['primary_focus'] = determine_primary_focus(engagement_data)
       engagement_data['strategies'] = list(set(all_strategies))
       
       logger.info("Engagement analysis completed successfully")
       logger.debug(f"Final engagement metrics: {engagement_data['engagement_metrics']}")
       
       return engagement_data
       
   except Exception as e:
       logger.error(f"Error in engagement analysis: {str(e)}\n{traceback.format_exc()}")
       return get_default_engagement_analysis()
def detect_visual_strategies(frames):
    """Detect visual engagement strategies"""
    try:
        strategies = {
            'strategies': [],
            'consistency': 0.0,
            'details': {}
        }
        
        frames_with_text = 0
        frames_with_effects = 0
        frames_with_hooks = 0
        total_frames = len(frames)
        
        for frame in frames:
            # Text overlay detection
            text_regions = detect_text_enhanced(frame)
            if text_regions:
                frames_with_text += 1
                if 'text_overlay' not in strategies['strategies']:
                    strategies['strategies'].append('text_overlay')
                    
            # Visual effects detection
            effects = detect_frame_effects(frame)
            if effects:
                frames_with_effects += 1
                strategies['strategies'].extend([f"effect_{effect}" for effect in effects])
                
            # Visual hooks detection
            hooks = detect_visual_hooks(frame)
            if hooks:
                frames_with_hooks += 1
                strategies['strategies'].extend(hooks)
                
        # Calculate consistency scores
        if total_frames > 0:
            text_consistency = frames_with_text / total_frames
            effects_consistency = frames_with_effects / total_frames
            hooks_consistency = frames_with_hooks / total_frames
            
            strategies['consistency'] = (text_consistency + effects_consistency + hooks_consistency) / 3
            strategies['details'] = {
                'text_frequency': float(text_consistency),
                'effects_frequency': float(effects_consistency),
                'hooks_frequency': float(hooks_consistency)
            }
            
        return strategies
        
    except Exception as e:
        logger.debug(f"Error detecting visual strategies: {str(e)}")
        return {'strategies': [], 'consistency': 0.0, 'details': {}}

def detect_audio_strategies(audio_features):
    """Detect audio engagement strategies"""
    try:
        strategies = {
            'strategies': [],
            'consistency': 0.0,
            'details': {}
        }
        
        if not audio_features:
            return strategies

        # Check for call and response patterns
        if audio_features.get('call_response'):
            strategies['strategies'].append('call_and_response')
            
        # Check for repetition
        if audio_features.get('repetition'):
            strategies['strategies'].append('repetitive_elements')
            
        # Check for hooks
        hooks = audio_features.get('hooks', [])
        if hooks:
            strategies['strategies'].append('musical_hooks')
            strategies['details']['hook_count'] = len(hooks)
            
        # Check for dynamic range
        if audio_features.get('dynamic_range', 0) > 40:
            strategies['strategies'].append('dynamic_contrast')
            
        # Calculate consistency based on presence of strategies
        strategies['consistency'] = len(strategies['strategies']) / 4  # Normalize by possible strategies
        
        return strategies
        
    except Exception as e:
        logger.debug(f"Error detecting audio strategies: {str(e)}")
        return {'strategies': [], 'consistency': 0.0, 'details': {}}


def process_video_frames(video_path, process_func, max_frames=None, sample_rate=1):
    """Process video frames with a given function"""
    try:
        # Changed to get_video_frames_fixed
        frames = get_video_frames_fixed(video_path, max_frames, sample_rate)
        if not frames:
            return []
            
        results = []
        for frame in frames:
            try:
                # Process the frame after confirming size
                if frame.shape[0] < 480 or frame.shape[1] < 480:
                    logger.warning(f"Small frame detected in process_video_frames: {frame.shape}")
                    frame = process_frame_safely(frame)
                    if frame is None:
                        continue
                
                result = process_func(frame)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error processing frame: {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return []

def analyze_video(video_path, analysis_functions):
    """Run multiple analysis functions on a video"""
    try:
        # Changed to get_video_frames_fixed
        frames = get_video_frames_fixed(video_path)
        if not frames:
            return {}
            
        # Log frame sizes at this point
        logger.info(f"analyze_video received {len(frames)} frames")
        if frames:
            logger.info(f"First frame size in analyze_video: {frames[0].shape}")
            
        results = {}
        for name, func in analysis_functions.items():
            try:
                results[name] = func(frames)
            except Exception as e:
                logger.warning(f"Error in {name} analysis: {str(e)}")
                results[name] = None
                
        return results
        
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        return {}

def process_frames_sequence(frames, process_func):
    """
    Process a sequence of frames with enhanced error handling and validation.
    
    Args:
        frames (list): List of frames
        process_func (callable): Processing function to apply
        
    Returns:
        list: List of processed results
    """
    if not frames:
        return []
        
    results = []
    for i, frame in enumerate(frames):
        try:
            # Process and validate frame first
            processed_frame = process_frame_safely(frame)
            if processed_frame is None:
                continue
                
            if not validate_frame(processed_frame):
                logger.debug(f"Frame {i} failed validation")
                continue
                
            # Apply the processing function
            result = process_func(processed_frame)
            if result is not None:
                results.append(result)
                
        except Exception as e:
            logger.warning(f"Error processing frame {i}: {str(e)}")
            continue
            
    return results

def process_single_video(video_file, audio_file, metadata):
    try:
        post_id = video_file.split('/')[-1].split('.')[0]
        logger.info(f"\n{'='*50}\nStarting processing for video {post_id}\n{'='*50}")

        # Download files
        logger.info("Downloading video and audio files...")
        local_video = download_file_from_s3('upnextsocialanalysis', video_file)
        local_audio = download_file_from_s3('upnextsocialanalysis', audio_file)

        if not local_video or not local_audio:
            logger.error(f"Failed to download files for {post_id}")
            return None

        logger.info("Successfully downloaded video and audio files")

        # Extract frames with logging
        logger.info("Extracting video frames...")
        frames = get_video_frames_fixed(local_video, max_frames=30, sample_rate=10)
        if frames:
            logger.info(f"Successfully extracted {len(frames)} frames")
            logger.info(f"Frame shape: {frames[0].shape if frames else 'No frames'}")
            logger.info(f"Frame data type: {frames[0].dtype if frames else 'No frames'}")
        else:
            logger.error("No frames extracted from video")
            return None

        try:
            # Audio Feature Extraction
            logger.info("\n---Audio Analysis Start---")
            try:
                logger.info("Loading audio file...")
                y, sr = librosa.load(local_audio, sr=22050)
                logger.info(f"Audio loaded - Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
                
                logger.info("Extracting audio features...")
                audio_features = extract_audio_features(local_audio)
                if audio_features:
                    logger.info("Audio Features Extracted:")
                    logger.info(f"Rhythm features: {audio_features.get('rhythm', {})}")
                    logger.info(f"Spectral features: {audio_features.get('spectral', {})}")
                    logger.info(f"Production features: {audio_features.get('production', {})}")
                else:
                    logger.warning("No audio features extracted")
            except Exception as e:
                logger.error(f"Error in audio feature extraction: {str(e)}")
                audio_features = None
            logger.info("---Audio Analysis End---\n")

            # Initialize results dictionary
            results = {}

            # Genre Detection
            logger.info("\n---Genre Detection Start---")
            try:
                logger.info("Detecting genre...")
                results['genre'] = classify_genre(audio_features if audio_features else {'y': y, 'sr': sr})
                logger.info(f"Detected genre: {results['genre']}")
            except Exception as e:
                logger.error(f"Error in genre detection: {str(e)}")
                results['genre'] = "Unknown Genre"
            logger.info("---Genre Detection End---\n")

            # Camera Analysis
            logger.info("\n---Camera Analysis Start---")
            try:
                logger.info("Analyzing camera movement...")
                results['camera'] = camera_analysis(frames)
                logger.info(f"Camera movement analysis: {results['camera']}")
            except Exception as e:
                logger.error(f"Error in camera analysis: {str(e)}")
                results['camera'] = "static"
            logger.info("---Camera Analysis End---\n")

            # Scene Analysis
            logger.info("\n---Scene Analysis Start---")
            try:
                logger.info("Analyzing scene and location...")
                scene_results = analyze_scene_and_location(frames, audio_features)
                logger.info("Scene Analysis Results:")
                logger.info(f"Technical data: {scene_results.get('technical_data', {})}")
                logger.info(f"Lighting: {scene_results.get('lighting', {})}")
                logger.info(f"Composition: {scene_results.get('composition', {})}")
                logger.info(f"Setting: {scene_results.get('setting', {})}")
                results['scene'] = scene_results
            except Exception as e:
                logger.error(f"Error in scene analysis: {str(e)}")
                results['scene'] = get_default_scene_analysis()
            logger.info("---Scene Analysis End---\n")

            # Activity Analysis
            logger.info("\n---Activity Analysis Start---")
            try:
                logger.info("Analyzing activity...")
                activity_results = analyze_music_video_activity(frames, audio_features)
                logger.info("Activity Analysis Results:")
                logger.info(f"Activity description: {activity_results.get('activity', '')}")
                logger.info(f"Performance type: {activity_results.get('performance_type', '')}")
                logger.info(f"Props detected: {activity_results.get('props', [])}")
                logger.info(f"Location: {activity_results.get('location', '')}")
                results['activity'] = activity_results
            except Exception as e:
                logger.error(f"Error in activity analysis: {str(e)}")
                results['activity'] = get_default_activity_analysis()
            logger.info("---Activity Analysis End---\n")

            # Performance Analysis
            logger.info("\n---Performance Analysis Start---")
            try:
                logger.info("Analyzing performance engagement...")
                performance_results = analyze_performance_engagement(frames)
                logger.info("Performance Analysis Results:")
                logger.info(f"Direct address: {performance_results.get('direct_address', False)}")
                logger.info(f"Gestures: {performance_results.get('gestures', False)}")
                logger.info(f"Emotional range: {performance_results.get('emotional_range', 0.0)}")
                logger.info(f"Techniques: {performance_results.get('techniques', [])}")
                results['performance'] = performance_results
            except Exception as e:
                logger.error(f"Error in performance analysis: {str(e)}")
                results['performance'] = get_default_performance_analysis()
            logger.info("---Performance Analysis End---\n")

            # Emotion Analysis
            logger.info("\n---Emotion Analysis Start---")
            try:
                logger.info("Analyzing emotions...")
                emotion_results = emotion_analysis_safe(frames[:ANALYSIS_FRAME_LIMIT])
                logger.info("Emotion Analysis Results:")
                logger.info(f"Emotion scores: {emotion_results[0] if isinstance(emotion_results, tuple) else {}}")
                logger.info(f"Description: {emotion_results[1] if isinstance(emotion_results, tuple) else 'NA'}")
                results['emotions'] = emotion_results
            except Exception as e:
                logger.error(f"Error in emotion analysis: {str(e)}")
                results['emotions'] = ({}, "NA")
            logger.info("---Emotion Analysis End---\n")

            # Engagement Analysis
            logger.info("\n---Engagement Analysis Start---")
            try:
                logger.info("Analyzing engagement strategies...")
                engagement_results = analyze_engagement_strategies(frames[:ANALYSIS_FRAME_LIMIT], audio_features)
                logger.info("Engagement Analysis Results:")
                logger.info(f"Strategies: {engagement_results.get('strategies', [])}")
                logger.info(f"Primary focus: {engagement_results.get('primary_focus', '')}")
                logger.info(f"Effectiveness score: {engagement_results.get('engagement_metrics', {}).get('effectiveness', 0.0)}")
                logger.info(f"Visual engagement: {engagement_results.get('visual_engagement', {})}")
                logger.info(f"Performance engagement: {engagement_results.get('performance_engagement', {})}")
                results['engagement'] = engagement_results
            except Exception as e:
                logger.error(f"Error in engagement analysis: {str(e)}")
                results['engagement'] = get_default_engagement_analysis()
            logger.info("---Engagement Analysis End---\n")

            # Sync Analysis
            logger.info("\n---Sync Analysis Start---")
            try:
                logger.info("Analyzing audio-visual sync...")
                sync_results = analyze_audio_visual_sync(frames, local_audio)
                logger.info("Sync Analysis Results:")
                logger.info(f"Sync score: {sync_results.get('sync_score', 0.0)}")
                logger.info(f"Quality: {sync_results.get('quality', '')}")
                logger.info(f"Timing details: {sync_results.get('timing_details', {})}")
                results['sync'] = sync_results
            except Exception as e:
                logger.error(f"Error in sync analysis: {str(e)}")
                results['sync'] = get_default_sync_analysis()
            logger.info("---Sync Analysis End---\n")

            # Compile Results
            logger.info("\n---Results Compilation Start---")
            try:
                logger.info(f"Compiling final results for {post_id}")
                logger.info("Pre-compilation results structure:")
                for key, value in results.items():
                    logger.info(f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        logger.info(f"{key} keys: {value.keys()}")
                
                analysis_results = compile_analysis_results(post_id, metadata, results, audio_features)
                
                if analysis_results:
                    logger.info("Results compiled successfully")
                    logger.info("Compiled results structure:")
                    for key, value in analysis_results.items():
                        logger.info(f"{key}: {type(value)}")
                else:
                    logger.error("Failed to compile results")
                    
            except Exception as e:
                logger.error(f"Error in results compilation: {str(e)}")
                return None
            logger.info("---Results Compilation End---\n")

            return analysis_results

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error processing video {post_id}: {str(e)}")
        return None
    finally:
        # Cleanup
        try:
            if 'local_video' in locals() and os.path.exists(local_video):
                os.remove(local_video)
            if 'local_audio' in locals() and os.path.exists(local_audio):
                os.remove(local_audio)
            gc.collect()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")


def compile_analysis_results(post_id, metadata, results, audio_features):
   """Compile comprehensive analysis results with enhanced metrics"""
   try:
       if not results:
           logger.error("No results to compile")
           return None
           
       logger.info(f"Compiling results for post {post_id}")

       compiled_results = {
           'post_id': str(post_id),
           'timestamp': str(metadata.iloc[0]['timePosted']) if not metadata.empty else None,
           'user': str(metadata.iloc[0]['username']) if not metadata.empty else None,
           
           # Enhanced Audio Analysis
           'genre': results.get('genre', "Unknown Genre"),
           'audio_features': {
               'rhythm': {
                   'tempo': audio_features.get('rhythm', {}).get('tempo', 0),
                   'beat_strength': audio_features.get('rhythm', {}).get('beat_strength', 0),
                   'beat_regularity': audio_features.get('rhythm', {}).get('beat_regularity', 0),
                   'rhythm_patterns': audio_features.get('rhythm', {}).get('rhythm_patterns', {}),
                   'additional_metrics': {
                       'beat_count': audio_features.get('rhythm', {}).get('additional_metrics', {}).get('beat_count', 0),
                       'average_bpm': audio_features.get('rhythm', {}).get('additional_metrics', {}).get('average_bpm', 0),
                       'bpm_stability': audio_features.get('rhythm', {}).get('additional_metrics', {}).get('bpm_stability', 0),
                       'groove_consistency': audio_features.get('rhythm', {}).get('additional_metrics', {}).get('groove_consistency', 0),
                       'tempo_stability': audio_features.get('rhythm', {}).get('additional_metrics', {}).get('tempo_stability', 0)
                   }
               },
               'spectral': {
                   'spectral_centroid': audio_features.get('spectral', {}).get('spectral_centroid', 0),
                   'spectral_bandwidth': audio_features.get('spectral', {}).get('spectral_bandwidth', 0),
                   'spectral_rolloff': audio_features.get('spectral', {}).get('spectral_rolloff', 0),
                   'mfccs': audio_features.get('spectral', {}).get('mfccs', [])
               },
               'production': {
                   'eq_profile': analyze_eq_profile_fixed(y, sr, n_fft=2048, hop_length=512),
                   'bass_character': analyze_bass_character_safely(audio_features),
                   'dynamic_range': audio_features.get('production', {}).get('dynamic_range', 0),
                   'compression': audio_features.get('production', {}).get('compression', 0),
                   'reverb': audio_features.get('production', {}).get('reverb', 0),
                   'distortion': audio_features.get('production', {}).get('distortion', 0)
               }
           },

           'emotions': analyze_emotions_safely(results.get('emotions', ({}, "NA"))),

           # Enhanced Scene Analysis
           'scene': {
               'location': results.get('scene', {}).get('technical_data', {}).get('primary_location', "unknown"),
               'lighting': analyze_lighting_safely(results.get('scene', {})),
               'composition': analyze_composition_safely(results.get('scene', {})),
               'setting_type': results.get('scene', {}).get('technical_data', {}).get('setting_type', "unknown"),
               'visual_elements': results.get('scene', {}).get('setting', {}).get('visual_elements', []),
               'scene_changes': results.get('scene', {}).get('technical_data', {}).get('scene_changes', 0)
           },

           # Enhanced Performance Analysis
           'performance': analyze_performance_safely(results.get('performance', {})),

           # Enhanced Engagement Analysis
           'engagement': analyze_engagement_safely(results.get('engagement', {})),

           # Enhanced Sync Analysis
           'sync': analyze_sync_safely(results.get('sync', {}))
       }

       # Validate compiled results
       for key, value in compiled_results.items():
           if value is None:
               logger.warning(f"Missing value for {key}")

       logger.info("Results compilation completed successfully")
       return compiled_results

   except Exception as e:
       logger.error(f"Error compiling analysis results: {str(e)}\n{traceback.format_exc()}")
       return None

# Helper functions for safe analysis
def analyze_eq_profile_safely(audio_features):
   """Safely analyze EQ profile"""
   try:
       eq = audio_features.get('production', {}).get('eq_profile', {})
       return {
           'sub_bass': eq.get('sub_bass', {}).get('energy', 0),
           'bass': eq.get('bass', {}).get('energy', 0),
           'low_mids': eq.get('low_mids', {}).get('energy', 0),
           'mids': eq.get('mids', {}).get('energy', 0),
           'high_mids': eq.get('high_mids', {}).get('energy', 0),
           'highs': eq.get('highs', {}).get('energy', 0)
       }
   except Exception as e:
       logger.warning(f"Error analyzing EQ profile: {str(e)}")
       return {}

def analyze_bass_character_safely(audio_features):
   """Safely analyze bass character"""
   try:
       bass = audio_features.get('production', {}).get('bass_character', {})
       return {
           'intensity': bass.get('intensity', 0),
           'variation': bass.get('variation', 0),
           'character': bass.get('character', 'unknown')
       }
   except Exception as e:
       logger.warning(f"Error analyzing bass character: {str(e)}")
       return {}

def analyze_emotions_safely(emotions):
   """Safely analyze emotions"""
   try:
       return {
           'scores': emotions[0] if isinstance(emotions, tuple) else {},
           'description': emotions[1] if isinstance(emotions, tuple) else "NA",
           'emotional_range': 0.0,
           'emotional_consistency': 0.0,
           'intensity_metrics': {
               'peak_intensity': 0.0,
               'average_intensity': 0.0,
               'intensity_variation': 0.0
           }
       }
   except Exception as e:
       logger.warning(f"Error analyzing emotions: {str(e)}")
       return {}

def analyze_lighting_safely(scene_data):
   """Safely analyze lighting data"""
   try:
       lighting = scene_data.get('lighting', {})
       return {
           'summary': scene_data.get('technical_data', {}).get('lighting_summary', "unknown"),
           'brightness_level': lighting.get('brightness_level', 0.0),
           'contrast_level': lighting.get('contrast_level', 0.0),
           'lighting_type': lighting.get('lighting_type', "unknown"),
           'color_temperature': lighting.get('color_temperature', "unknown"),
           'direction': lighting.get('direction', 0),
           'uniformity': lighting.get('uniformity', 0.0),
           'effects': lighting.get('effects', [])
       }
   except Exception as e:
       logger.warning(f"Error analyzing lighting: {str(e)}")
       return {}

def analyze_composition_safely(scene_data):
   """Safely analyze composition data"""
   try:
       composition = scene_data.get('composition', {})
       return {
           'symmetry_score': composition.get('symmetry_score', 0.0),
           'rule_of_thirds_score': composition.get('rule_of_thirds_score', 0.0),
           'balance_score': composition.get('balance_score', 0.0),
           'depth_score': composition.get('depth_score', 0.0),
           'framing_score': composition.get('framing_score', 0.0)
       }
   except Exception as e:
       logger.warning(f"Error analyzing composition: {str(e)}")
       return {}

def analyze_performance_safely(performance_data):
   """Safely analyze performance data"""
   try:
       return {
           'direct_address': performance_data.get('direct_address', False),
           'gestures': performance_data.get('gestures', False),
           'emotional_range': performance_data.get('emotional_range', 0.0),
           'techniques': performance_data.get('techniques', []),
           'confidence_metrics': {
               'confidence_score': performance_data.get('confidence_score', 0.0),
               'face_size_avg': performance_data.get('face_size_avg', 0.0),
               'position_stability': performance_data.get('position_stability', 0.0)
           },
           'energy_score': performance_data.get('energy_score', 0.0)
       }
   except Exception as e:
       logger.warning(f"Error analyzing performance: {str(e)}")
       return {}

def analyze_engagement_safely(engagement_data):
   """Safely analyze engagement data"""
   try:
       return {
           'strategies': engagement_data.get('strategies', []),
           'primary_focus': engagement_data.get('primary_focus', "unknown"),
           'effectiveness_score': engagement_data.get('engagement_metrics', {}).get('effectiveness', 0.0),
           'consistency': engagement_data.get('engagement_metrics', {}).get('consistency', 0.0),
           'variety': engagement_data.get('engagement_metrics', {}).get('variety', 0.0)
       }
   except Exception as e:
       logger.warning(f"Error analyzing engagement: {str(e)}")
       return {}

def analyze_sync_safely(sync_data):
   """Safely analyze sync data"""
   try:
       return {
           'sync_score': sync_data.get('sync_score', 0.0),
           'quality': sync_data.get('quality', "NA"),
           'timing_details': {
               'timing_quality': sync_data.get('timing_details', {}).get('timing_quality', 0.0),
               'movement_alignment': sync_data.get('timing_details', {}).get('movement_alignment', 0.0),
               'strong_sync_points': sync_data.get('timing_details', {}).get('strong_sync_points', 0),
               'weak_sync_points': sync_data.get('timing_details', {}).get('weak_sync_points', 0)
           }
       }
   except Exception as e:
       logger.warning(f"Error analyzing sync: {str(e)}")
       return {}

def format_analysis_results(results):
    """Format analysis results into CSV-compatible list with comprehensive metrics"""
    try:
        if not results:
            return [''] * len(CSV_HEADERS)
            
        return [
            # Basic Information
            str(results.get('post_id', '')),
            str(results.get('timestamp', '')),
            str(results.get('user', '')),
            
            # Audio Analysis - Rhythm
            str(results.get('genre', '')),
            str(results.get('audio_features', {}).get('rhythm', {}).get('tempo', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('beat_strength', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('beat_regularity', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('rhythm_patterns', {}).get('trap', False)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('rhythm_patterns', {}).get('drill', False)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('rhythm_patterns', {}).get('boom_bap', False)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('rhythm_patterns', {}).get('dance', False)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('additional_metrics', {}).get('beat_count', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('additional_metrics', {}).get('average_bpm', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('additional_metrics', {}).get('bpm_stability', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('additional_metrics', {}).get('groove_consistency', 0)),
            str(results.get('audio_features', {}).get('rhythm', {}).get('additional_metrics', {}).get('tempo_stability', 0)),
            
            # Audio Analysis - Spectral
            str(results.get('audio_features', {}).get('spectral', {}).get('spectral_centroid', 0)),
            str(results.get('audio_features', {}).get('spectral', {}).get('spectral_bandwidth', 0)),
            str(results.get('audio_features', {}).get('spectral', {}).get('spectral_rolloff', 0)),
            
            # Audio Analysis - Production
            str(results.get('audio_features', {}).get('production', {}).get('dynamic_range', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('compression', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('reverb', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('distortion', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('eq_profile', {}).get('sub_bass', {}).get('energy', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('eq_profile', {}).get('bass', {}).get('energy', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('eq_profile', {}).get('low_mids', {}).get('energy', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('eq_profile', {}).get('mids', {}).get('energy', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('eq_profile', {}).get('high_mids', {}).get('energy', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('eq_profile', {}).get('highs', {}).get('energy', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('bass_character', {}).get('intensity', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('bass_character', {}).get('variation', 0)),
            str(results.get('audio_features', {}).get('production', {}).get('bass_character', {}).get('character', '')),
            
            # Emotional Analysis
            str(results.get('emotions', {}).get('dominant_emotion', '')),
            str(results.get('emotions', {}).get('description', '')),
            str(results.get('emotions', {}).get('emotional_range', 0)),
            str(results.get('emotions', {}).get('emotional_consistency', 0)),
            str(results.get('emotions', {}).get('intensity_metrics', {}).get('peak_intensity', 0)),
            str(results.get('emotions', {}).get('intensity_metrics', {}).get('average_intensity', 0)),
            str(results.get('emotions', {}).get('intensity_metrics', {}).get('intensity_variation', 0)),
            
            # Scene Analysis - Location and Setting
            str(results.get('scene', {}).get('location', '')),
            str(results.get('scene', {}).get('setting_type', '')),
            str(results.get('scene', {}).get('scene_changes', 0)),
            '; '.join(str(elem) for elem in results.get('scene', {}).get('visual_elements', [])),
            
            # Scene Analysis - Lighting
            str(results.get('scene', {}).get('lighting', {}).get('summary', '')),
            str(results.get('scene', {}).get('lighting', {}).get('brightness_level', 0)),
            str(results.get('scene', {}).get('lighting', {}).get('contrast_level', 0)),
            str(results.get('scene', {}).get('lighting', {}).get('lighting_type', '')),
            str(results.get('scene', {}).get('lighting', {}).get('color_temperature', '')),
            str(results.get('scene', {}).get('lighting', {}).get('direction', 0)),
            str(results.get('scene', {}).get('lighting', {}).get('uniformity', 0)),
            '; '.join(str(effect) for effect in results.get('scene', {}).get('lighting', {}).get('effects', [])),
            
            # Scene Analysis - Composition
            str(results.get('scene', {}).get('composition', {}).get('symmetry_score', 0)),
            str(results.get('scene', {}).get('composition', {}).get('rule_of_thirds_score', 0)),
            str(results.get('scene', {}).get('composition', {}).get('balance_score', 0)),
            str(results.get('scene', {}).get('composition', {}).get('depth_score', 0)),
            str(results.get('scene', {}).get('composition', {}).get('framing_score', 0)),
            str(results.get('scene', {}).get('composition', {}).get('focal_point', {}).get('position', '')),
            
            # Performance Analysis - Basic
            str(results.get('performance', {}).get('direct_address', False)),
            str(results.get('performance', {}).get('gestures', False)),
            str(results.get('performance', {}).get('emotional_range', 0)),
            '; '.join(str(tech) for tech in results.get('performance', {}).get('techniques', [])),
            
            # Performance Analysis - Detailed
            str(results.get('performance', {}).get('confidence_metrics', {}).get('confidence_score', 0)),
            str(results.get('performance', {}).get('confidence_metrics', {}).get('face_size_avg', 0)),
            str(results.get('performance', {}).get('confidence_metrics', {}).get('position_stability', 0)),
            str(results.get('performance', {}).get('gesture_frequency', 0)),
            str(results.get('performance', {}).get('gesture_intensity', 0)),
            '; '.join(str(type) for type in results.get('performance', {}).get('gesture_types', [])),
            str(results.get('performance', {}).get('energy_score', 0)),
            
            # Engagement Analysis - Strategies
            '; '.join(str(strat) for strat in results.get('engagement', {}).get('strategies', [])),
            str(results.get('engagement', {}).get('primary_focus', '')),
            str(results.get('engagement', {}).get('effectiveness_score', 0)),
            
            # Engagement Analysis - Visual
            str(results.get('engagement', {}).get('visual_engagement', {}).get('text_persistence', 0)),
            str(results.get('engagement', {}).get('visual_engagement', {}).get('effect_frequency', 0)),
            str(results.get('engagement', {}).get('visual_engagement', {}).get('hooks_frequency', 0)),
            str(results.get('engagement', {}).get('visual_engagement', {}).get('visual_variety', 0)),
            
            # Engagement Analysis - Performance
            str(results.get('engagement', {}).get('performance_engagement', {}).get('direct_address_frequency', 0)),
            str(results.get('engagement', {}).get('performance_engagement', {}).get('emotional_range', 0)),
            str(results.get('engagement', {}).get('performance_engagement', {}).get('interaction_level', 0)),
            str(results.get('engagement', {}).get('consistency', 0)),
            str(results.get('engagement', {}).get('variety', 0)),
            
            # Sync Analysis
            str(results.get('sync', {}).get('sync_score', 0)),
            str(results.get('sync', {}).get('quality', '')),
            str(results.get('sync', {}).get('timing_details', {}).get('timing_quality', 0)),
            str(results.get('sync', {}).get('timing_details', {}).get('movement_alignment', 0)),
            str(results.get('sync', {}).get('timing_details', {}).get('strong_sync_points', 0)),
            str(results.get('sync', {}).get('timing_details', {}).get('weak_sync_points', 0)),
            '; '.join(str(pattern) for pattern in results.get('sync', {}).get('timing_details', {}).get('sync_patterns', []))
        ]

    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        return [''] * len(CSV_HEADERS)

def process_all_files():
    """Process all videos with minimal required functionality"""
    print("Starting process...")
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed")
        return
        
    # Download metadata
    metadata_df = download_metadata_from_s3('upnextsocialanalysis', 'csvdata/CleanedTikTokData.csv')
    if metadata_df is None:
        print("No metadata found")
        return
        
    try:
        # Get video files
        result = s3.list_objects_v2(Bucket='upnextsocialanalysis', Prefix='videos/')
        video_files = [item['Key'] for item in result.get('Contents', [])]
        print(f"Found {len(video_files)} video files")
        
        # Process each video
        for i, video_file in enumerate(video_files):
            try:
                print(f"\nProcessing video {i+1}/{len(video_files)}: {video_file}")
                
                # Get corresponding audio file
                audio_file = video_file.replace('videos/', 'audios/').replace('.mp4', '.mp3')
                
                # Get metadata
                video_id = video_file.split('/')[-1].split('.')[0]
                metadata = metadata_df[metadata_df['post_id'] == int(video_id)]
                
                if metadata.empty:
                    print(f"No metadata found for {video_file}")
                    continue
                    
                # Process video
                result = process_single_video(video_file, audio_file, metadata)
                
                if result:
                    try:
                        formatted_result = format_analysis_results(result)
                        write_results_to_csv([formatted_result], f"/tmp/results.csv")
                        print(f"Processed video {i+1}")
                    except Exception as e:
                        print(f"Error formatting/writing results for video {i+1}: {str(e)}")
                        continue
                    
            except Exception as e:
                print(f"Error processing video {i+1}: {str(e)}")
                continue
                
            # Clear memory periodically
            if i % 10 == 0:
                gc.collect()
                
    except Exception as e:
        print(f"Processing error: {str(e)}")
    finally:
        # Cleanup
        try:
            if os.path.exists('/tmp/results.csv'):
                s3.upload_file('/tmp/results.csv', 'upnextsocialanalysis', 'csvdata/results.csv')
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

def write_results_to_csv(results, csv_file="/tmp/results.csv"):
    """Write enhanced results to CSV with comprehensive metrics"""
    try:
        # Read existing data if file exists
        existing_data = []
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                df = df[~df['PostID'].isin(['post_id', 'None', 'nan'])]
                existing_data = df.values.tolist()
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {str(e)}")

        # Combine existing data with new results
        all_data = existing_data + results

        # Create DataFrame without index
        df = pd.DataFrame(all_data, columns=CSV_HEADERS)
        
        # Clean up data
        df = df[~df['PostID'].isin(['post_id', 'None', 'nan'])]
        df = df.replace('nan', '')
        df = df.replace('None', '')
        df = df.replace('unknown', '')
        
        # Write to CSV
        df.to_csv(csv_file, index=False)

        # Upload to S3
        try:
            s3.upload_file(csv_file, 'upnextsocialanalysis', 'csvdata/results.csv')
            logger.info(f"Results successfully written to {csv_file} and uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading results to S3: {str(e)}")

    except Exception as e:
        logger.error(f"Error writing results to CSV: {str(e)}")


if __name__ == "__main__":
    process_all_files()
