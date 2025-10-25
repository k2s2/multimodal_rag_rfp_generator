import cv2
from PIL import Image
import numpy as np
import easyocr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer


# New optimised code
_blip_model = None
_blip_processor = None
_easyocr_reader = None
_device = None

def get_easyocr_reader():
    """Get cached EasyOCR reader"""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'])
    return _easyocr_reader

def get_blip_model():
    """Get cached BLIP model components"""
    global _blip_model, _blip_processor, _device
    if _blip_model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        # local_path = "./models/blip-large"
        # tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
        # _blip_processor = BlipProcessor.from_pretrained(local_path, tokenizer=tokenizer)
        # _blip_model = BlipForConditionalGeneration.from_pretrained(local_path).to(_device)
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", tokenizer=tokenizer)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return model, processor, _device

def extract_image_text_easyocr(file):
    """Extract text from image using EasyOCR"""
    pil_image = Image.open(file)
    img_array = np.array(pil_image)
    reader = get_easyocr_reader()
    results = reader.readtext(img_array)
    return '\n'.join([result[1] for result in results])

def extract_image_description(image_path):
    """Generate a visual description of an image using BLIP model"""
    try:
        model, processor, device = get_blip_model()
        image = load_image(image_path)
        if image is None:
            return ""
        return generate_detailed_caption(image, processor, model, device)
    except Exception:
        return ""

def load_image(image_path):
    """Load and preprocess image for analysis"""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception:
        try:
            image_cv = cv2.imread(str(image_path))
            if image_cv is None:
                return None
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        except Exception:
            return None

def generate_detailed_caption(image, processor, model, device, 
                             max_length=150, num_beams=8, 
                             temperature=0.7, repetition_penalty=1.2):
    """Generate detailed caption for image using BLIP model with enhanced parameters"""
    try:
        inputs = processor(image, return_tensors="pt", padding=True).to(device)
        generation_config = {
            'max_length': max_length,
            'min_length': 25,
            'num_beams': num_beams,
            'early_stopping': True,
            'temperature': temperature,
            'do_sample': True,
            'repetition_penalty': repetition_penalty,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        caption = processor.decode(outputs[0], skip_special_tokens=True).strip()
        if caption:
            # Remove common prefixes
            prefixes = ["a photo of", "an image of", "a picture of"]
            caption_lower = caption.lower()
            for prefix in prefixes:
                if caption_lower.startswith(prefix):
                    caption = caption[len(prefix):].strip()
                    break
            # Capitalize and add period
            if caption:
                caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()
                if not caption.endswith(('.', '!', '?')):
                    caption += '.'
        return caption if caption else "No detailed description could be generated"
    except Exception:
        return ""

# def extract_image_text_easyocr(file):
#     """Extract text from image using EasyOCR"""
#     # Read image and convert to numpy array
#     pil_image = Image.open(file)
#     img_array = np.array(pil_image)
#     # Extract text
#     reader = easyocr.Reader(['en'])
#     results = reader.readtext(img_array)
#     return '\n'.join([result[1] for result in results])


# def extract_image_description(image_path):
#     """Generate a visual description of an image using BLIP model"""
#     try:
#         # Initialize BLIP model
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         local_path = "./models/blip-large"
#         tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
#         processor = BlipProcessor.from_pretrained(local_path, tokenizer=tokenizer)
#         model = BlipForConditionalGeneration.from_pretrained(local_path).to(device)
#         # Load and preprocess the image
#         image = load_image(image_path)
#         if image is None:
#             print("Error: Could not load image")
#             return ""
#         # Generate description
#         description = generate_detailed_caption(image, processor, model, device)
#         return description
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         return ""

# def load_image(image_path):
#     """Load and preprocess image for analysis"""
#     try:
#         # Try loading with PIL first
#         try:
#             image = Image.open(image_path)
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
#             return image
#         except Exception:
#             # Fallback to OpenCV
#             image_cv = cv2.imread(str(image_path))
#             if image_cv is None:
#                 return None
#             # Convert BGR to RGB and then to PIL
#             image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
#             return Image.fromarray(image_rgb)
#     except Exception as e:
#         print(f"Error loading image: {str(e)}")
#         return None


# def generate_detailed_caption(image, processor, model, device, 
#                              max_length=150, num_beams=8, 
#                              temperature=0.7, repetition_penalty=1.2):
#     """Generate detailed caption for image using BLIP model with enhanced parameters"""
#     try:
#         # Ensure image is in RGB format
#         if hasattr(image, 'convert'):
#             image = image.convert('RGB')
#         # Process image
#         inputs = processor(image, return_tensors="pt", padding=True).to(device)
#         # Enhanced generation parameters for detailed captions
#         generation_config = {
#             'max_length': max_length,
#             'min_length': 25,
#             'num_beams': num_beams,
#             'early_stopping': True,
#             'temperature': temperature,
#             'do_sample': True,
#             'repetition_penalty': repetition_penalty,
#             'length_penalty': 1.0,
#             'no_repeat_ngram_size': 3,  # Avoid repetitive phrases
#         }
#         # Generate caption
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **generation_config)
#         # Decode and clean caption
#         caption = processor.decode(outputs[0], skip_special_tokens=True)
#         # Post-process for better quality
#         caption = caption.strip()
#         if caption:
#             # Remove common prefixes
#             prefixes = ["a photo of", "an image of", "a picture of"]
#             caption_lower = caption.lower()
#             for prefix in prefixes:
#                 if caption_lower.startswith(prefix):
#                     caption = caption[len(prefix):].strip()
#                     break
#             # Capitalize first letter and ensure proper ending
#             if caption:
#                 caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()
#                 if not caption.endswith(('.', '!', '?')):
#                     caption += '.'
#         return caption if caption else "No detailed description could be generated"
#     except Exception as e:
#         print(f"Error generating detailed description: {str(e)}")
#         return ""
