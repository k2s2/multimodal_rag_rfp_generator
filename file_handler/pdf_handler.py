import fitz
import easyocr
from PIL import Image
import io
import os
import numpy as np
from file_handler.image_handler import extract_image_description

# New optimised code
def extract_pdf_text(pdf_path):
    """Extract text from PDF using EasyOCR for images, with fallback to image description"""
    reader = easyocr.Reader(['en'])
    doc = fitz.open(pdf_path)
    all_text = []    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if page_text.strip():
            all_text.append(f"Page {page_num + 1}:\n{page_text}")
        # Extract and OCR images
        for img_index, img in enumerate(page.get_images()):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha>=4:
                    pix = None
                    continue
                # Convert directly to numpy array
                img_data = pix.pil_tobytes(format="PNG")
                pil_image = Image.open(io.BytesIO(img_data))
                img_array = np.array(pil_image)
                # Use EasyOCR
                results = reader.readtext(img_array)
                ocr_text = '\n'.join([result[1] for result in results])
                if ocr_text.strip():
                    all_text.append(f"Page {page_num + 1} Image {img_index + 1} (OCR):\n{ocr_text}")
                else:
                    # Fallback to image description using in-memory image
                    try:
                        temp_path = f"temp_img_{page_num}_{img_index}.png"
                        pil_image.save(temp_path)
                        description = extract_image_description(temp_path)
                        os.remove(temp_path)
                        if description and description.strip():
                            all_text.append(f"Page {page_num + 1} Image {img_index + 1} (Description):\n{description}")
                    except Exception:
                        pass
                pix = None
            except Exception:
                continue
    doc.close()
    return '\n\n'.join(all_text)


# def extract_pdf_text(pdf_path):
#     """Extract text from PDF using EasyOCR for images, with fallback to image description"""
#     reader = easyocr.Reader(['en'])
#     doc = fitz.open(pdf_path)
#     all_text = []
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         page_text = page.get_text()
#         if page_text.strip():
#             all_text.append(f"Page {page_num + 1}:\n{page_text}")
#         # Extract and OCR images
#         image_list = page.get_images()
#         for img_index, img in enumerate(image_list):
#             try:
#                 xref = img[0]
#                 pix = fitz.Pixmap(doc, xref)
#                 if pix.n - pix.alpha < 4:
#                     # Convert to numpy array for EasyOCR
#                     img_data = pix.tobytes("png")
#                     pil_image = Image.open(io.BytesIO(img_data))
#                     img_array = np.array(pil_image)
#                     # Use EasyOCR first
#                     results = reader.readtext(img_array)
#                     ocr_text = '\n'.join([result[1] for result in results]) if results else ""
#                     if ocr_text and ocr_text.strip():
#                         all_text.append(f"Page {page_num + 1} Image {img_index + 1} (OCR):\n{ocr_text}")
#                     else:
#                         # If no text is found we image description
#                         try:
#                             # Save temporary image for description function
#                             temp_img_path = f"temp_img_{page_num}_{img_index}.png"
#                             pil_image.save(temp_img_path)
#                             # Get visual description
#                             description = extract_image_description(temp_img_path)
#                             if description and description.strip():
#                                 all_text.append(f"Page {page_num + 1} Image {img_index + 1} (Description):\n{description}")
#                             # Clean up temp file
#                             if os.path.exists(temp_img_path):
#                                 os.remove(temp_img_path)
#                         except Exception as desc_error:
#                             print(f"Error getting image description: {desc_error}")
#                 pix = None
#             except Exception as e:
#                 print(f"Error processing image: {e}")
#     doc.close()
#     return '\n\n'.join(all_text)