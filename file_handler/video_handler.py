import cv2
import whisper
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr


# New optimised code
_whisper_model = None
_blip_model = None
_blip_processor = None
_easyocr_reader = None
_device = None

def get_models():
    """Get cached models"""
    global _whisper_model, _blip_model, _blip_processor, _easyocr_reader, _device
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    if _blip_model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        # local_path = "./models/blip-large"
        # _blip_processor = BlipProcessor.from_pretrained(local_path)
        # _blip_model = BlipForConditionalGeneration.from_pretrained(local_path).to(_device)
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", tokenizer=tokenizer)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'])
    return _whisper_model, model, processor, _easyocr_reader, _device

def extract_video_content(video_path):
    """Extract content from video using a comprehensive pipeline"""
    try:
        metadata = get_video_metadata(video_path)
        audio_text = extract_audio_from_video(video_path)
        visual_text = extract_visual_content_from_video(video_path, metadata)
        ocr_text = extract_text_from_video_frames(video_path, metadata)
        combined_content = f"""Video File: {Path(video_path).name}
            Duration: {metadata['duration']:.2f} seconds
            Resolution: {metadata['width']}x{metadata['height']}
            FPS: {metadata['fps']:.2f}

            AUDIO TRANSCRIPTION:
            {audio_text}

            VISUAL CONTENT ANALYSIS:
            {visual_text}

            TEXT EXTRACTED FROM VIDEO (OCR):
            {ocr_text}"""
        return combined_content.strip()
    except Exception:
        return ""

def extract_audio_from_video(video_path):
    """Extract and transcribe audio from video"""
    try:
        model, _, _, _, _ = get_models()
        result = model.transcribe(video_path)
        return result["text"]
    except Exception:
        return ""

def extract_visual_content_from_video(video_path, metadata=None):
    """Extract visual content from video frames using multiple approaches"""
    try:
        _, model, processor, _, device = get_models()
        if metadata is None:
            metadata = get_video_metadata(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = metadata['fps']
        duration = metadata['duration']
        # Calculate frame interval
        interval_seconds = min(30, duration / 10)
        frame_interval = max(1, int(fps * interval_seconds))
        visual_descriptions = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                description = frame_captioning(frame, timestamp, processor, model, device)
                if description:
                    visual_descriptions.append(description)
            frame_count += 1
        cap.release()
        return "\n".join(visual_descriptions)
    except Exception:
        return ""

def frame_captioning(frame, timestamp, processor, model, device):
    """Using BLIP to caption frames to get description"""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        inputs = processor(pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return f"At {timestamp:.2f}s: {caption}".strip()
    except Exception:
        return ""

def extract_text_from_video_frames(video_path, metadata=None):
    """Extract text from video frames using OCR"""
    try:
        _, _, _, reader, _ = get_models()
        if metadata is None:
            metadata = get_video_metadata(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = metadata['fps']
        ocr_interval = max(1, int(fps * 5))
        extracted_texts = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % ocr_interval == 0:
                timestamp = frame_count / fps
                try:
                    results = reader.readtext(frame)
                    frame_texts = [text.strip() for (bbox, text, confidence) in results if confidence > 0.5]
                    if frame_texts:
                        extracted_texts.append(f"Text at {timestamp:.1f}s: {' | '.join(frame_texts)}")
                except Exception:
                    pass
            frame_count += 1
        cap.release()
        return "\n".join(extracted_texts) if extracted_texts else "No readable text found in video frames"
    except Exception:
        return ""

def get_video_metadata(video_path):
    """Extract basic video metadata"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': frame_count
        }
    except Exception:
        return {
            'duration': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'total_frames': 0
        }


# def extract_video_content(video_path):
#     """Extract content from video using a comprehensive pipeline
#     Returns: Combined text content from video"""
#     try:
#         print(f"Processing video: {video_path}")
#         audio_text=extract_audio_from_video(video_path)
#         visual_text=extract_visual_content_from_video(video_path)
#         ocr_text=extract_text_from_video_frames(video_path)
#         metadata=get_video_metadata(video_path)
#         combined_content=f"""
#             Video File: {Path(video_path).name}
#             Duration: {metadata['duration']:.2f} seconds
#             Resolution: {metadata['width']}x{metadata['height']}
#             FPS: {metadata['fps']:.2f}
#             AUDIO TRANSCRIPTION:\n{audio_text}
#             VISUAL CONTENT ANALYSIS:\n{visual_text}
#             TEXT EXTRACTED FROM VIDEO (OCR):\n{ocr_text}"""
#         return combined_content.strip()
#     except Exception as e:
#         print(f"Error processing video: {str(e)}")
#         return ""

# def extract_audio_from_video(video_path):
#     """Extract and transcribe audio from video"""
#     try:
#         model=whisper.load_model("base")
#         result=model.transcribe(video_path)
#         return result["text"]
#     except Exception as e:
#         print(f"Error extracting audio: {str(e)}")
#         return ""

# def extract_visual_content_from_video(video_path):
#     """Extract visual content from video frames using multiple approaches"""
#     try:
#         # Initialize models for image captioning
#         device="cuda" if torch.cuda.is_available() else "cpu"
#         local_path = "./models/blip-large"
#         processor = BlipProcessor.from_pretrained(local_path)
#         model = BlipForConditionalGeneration.from_pretrained(local_path).to(device)
#         # Open video file
#         cap=cv2.VideoCapture(video_path)
#         # Get video properties
#         fps=cap.get(cv2.CAP_PROP_FPS)
#         total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration=total_frames/fps
#         # Extract frames at intervals
#         interval_seconds=min(30,duration/10)
#         frame_interval = max(1, int(fps * interval_seconds))
#         visual_descriptions=[]
#         frame_count=0
#         print(f"  Analyzing video frames every {interval_seconds:.1f} seconds...")
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # Process frame every interval
#             if frame_count%frame_interval==0:
#                 timestamp=frame_count/fps
#                 description=frame_captioning(frame, timestamp, processor, model, device)
#                 visual_descriptions.append(description)
#                 print(f"    Processed frame at {timestamp:.1f}s")
#             frame_count+=1
#         cap.release()
#         return "\n".join(visual_descriptions)
#     except Exception as e:
#         print(f"Error extracting visual content: {str(e)}")
#         return ""

# def frame_captioning(frame, timestamp, processor, model, device):
#     """Using BLIP to caption frames to get description"""
#     try:
#         # Convert BGR to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(frame_rgb)
#         # Generate image caption using BLIP
#         inputs = processor(pil_image, return_tensors="pt").to(device)
#         with torch.no_grad():
#             out = model.generate(**inputs, max_length=50)
#         caption = processor.decode(out[0], skip_special_tokens=True)
#         description = f"At {timestamp:.2f}s: {caption}"
#         return description.strip()
#     except Exception as e:
#         print(f"Error analyzing frame at {timestamp:.2f}s: {str(e)}")
#         return ""


# def extract_text_from_video_frames(video_path):
#     """Extract text from video frames using OCR"""
#     try:
#         reader = easyocr.Reader(['en'])
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         ocr_interval = max(1, int(fps * 5))
#         extracted_texts = []
#         frame_count = 0
#         print(f"  Extracting text from video frames...")
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_count % ocr_interval == 0:
#                 timestamp = frame_count / fps
#                 # Use EasyOCR to extract text
#                 try:
#                     results = reader.readtext(frame)
#                     frame_texts = []
#                     for (bbox, text, confidence) in results:
#                         if confidence > 0.5:  # Only keep high-confidence text
#                             frame_texts.append(text.strip())
#                     if frame_texts:
#                         extracted_texts.append(f"Text at {timestamp:.1f}s: {' | '.join(frame_texts)}")
#                 except Exception as ocr_error:
#                     print(f"    OCR error at {timestamp:.1f}s: {str(ocr_error)}")
#             frame_count += 1
#         cap.release()
#         if extracted_texts:
#             return "\n".join(extracted_texts)
#         else:
#             return "No readable text found in video frames"
#     except Exception as e:
#         print(f"Error extracting text from video: {str(e)}")
#         return ""

# def get_video_metadata(video_path):
#     """Extract basic video metadata"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         duration = frame_count / fps if fps > 0 else 0
#         cap.release()
#         return {
#             'duration': duration,
#             'fps': fps,
#             'width': width,
#             'height': height,
#             'total_frames': frame_count
#         }
#     except Exception as e:
#         print(f"Error extracting video metadata: {str(e)}")
#         return {
#             'duration': 0,
#             'fps': 0,
#             'width': 0,
#             'height': 0,
#             'total_frames': 0
#         }
