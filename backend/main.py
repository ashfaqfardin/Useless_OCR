from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
from PIL import Image
import io
from typing import List, Optional, Tuple
import base64
import requests
import json
from pydantic import BaseModel, ValidationError

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove global reader, create per-request for language flexibility

@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    languages: Optional[str] = Form(None)  # comma-separated string, e.g. 'en,bn'
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    # Parse languages
    if languages:
        lang_list = [lang.strip() for lang in languages.split(',') if lang.strip()]
    else:
        lang_list = ['en']
    reader = easyocr.Reader(lang_list, gpu=False)
    results = reader.readtext(image_np)
    # Format: [ [bbox, text, confidence], ... ]
    response = []
    for bbox, text, conf in results:
        # Convert bbox coordinates to native Python types
        bbox_py = [[float(x), float(y)] for x, y in bbox]
        response.append({
            "bbox": bbox_py,
            "text": text,
            "confidence": float(conf)
        })
    return JSONResponse(content={"results": response})

@app.post("/groq-ocr")
async def groq_ocr_image(
    file: UploadFile = File(...),
    groq_api_key: str = Form(...),
    model: str = Form("meta-llama/llama-4-maverick-17b-128e-instruct"),
    prompt: Optional[str] = Form(None)
):
    """
    Extract text from image using Groq API with vision models.
    Uses multimodal message format for Llama 4 Maverick 17B 128E.
    """
    # Default prompt for text extraction
    default_prompt = "Please extract all the text from this image. Return the text exactly as it appears, preserving line breaks and formatting. If there are multiple text regions, separate them with newlines. Do not add any commentary or interpretation."
    user_prompt = prompt if prompt else default_prompt
    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        # Use multimodal format for Llama 4 Maverick
        if model == "meta-llama/llama-4-maverick-17b-128e-instruct":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]
        else:
            # Fallback to text-only for other models
            messages = [
                {"role": "user", "content": user_prompt}
            ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            error_msg = f"Groq API error: {response.status_code} - {response.text}"
            return JSONResponse(content={"error": error_msg}, status_code=400)
        result = response.json()
        extracted_text = result["choices"][0]["message"]["content"].strip()
        response_data = {
            "results": [
                {
                    "bbox": [[0, 0], [100, 0], [100, 100], [0, 100]],  # Placeholder
                    "text": extracted_text,
                    "confidence": 0.95  # Placeholder confidence
                }
            ],
            "model_used": model,
            "extraction_method": "groq_vision"
        }
        return JSONResponse(content=response_data)
    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": f"Network error: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Processing error: {str(e)}"}, status_code=500)

@app.get("/groq-models")
async def get_groq_models():
    """Get list of available Groq models that support vision"""
    models = [
        {
            "id": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "name": "Llama 4 Maverick 17B 128E (Vision)",
            "description": "Multimodal (text + image) - Use for OCR and image tasks",
            "max_tokens": 8192
        },
        {
            "id": "llama3-8b-8192",
            "name": "Llama 3 8B",
            "description": "Fast and efficient 8B parameter model (text only)",
            "max_tokens": 8192
        },
        {
            "id": "llama3-70b-8192", 
            "name": "Llama 3 70B",
            "description": "High quality 70B parameter model (text only)",
            "max_tokens": 8192
        },
        {
            "id": "mixtral-8x7b-32768",
            "name": "Mixtral 8x7B",
            "description": "High performance mixture of experts model (text only)",
            "max_tokens": 32768
        },
        {
            "id": "gemma2-9b-it",
            "name": "Gemma 2 9B",
            "description": "Google's efficient 9B parameter model (text only)",
            "max_tokens": 8192
        }
    ]
    return JSONResponse(content={"models": models}) 

@app.post("/hybrid-ocr")
async def hybrid_ocr_image(
    file: UploadFile = File(...),
    groq_api_key: str = Form(...),
    model: str = Form("meta-llama/llama-4-maverick-17b-128e-instruct"),
    languages: Optional[str] = Form("en"),  # Default to English
    prompt: Optional[str] = Form(None)
):
    """
    Hybrid OCR: Use Groq for text extraction and EasyOCR for bounding boxes.
    Combines the accuracy of Groq's vision model with EasyOCR's bounding box detection.
    """
    # Default prompt for text extraction
    default_prompt = "Please extract all the text from this image. Return the text exactly as it appears, preserving line breaks and formatting. If there are multiple text regions, separate them with newlines. Do not add any commentary or interpretation."
    user_prompt = prompt if prompt else default_prompt
    
    try:
        # Read image for both Groq and EasyOCR
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Step 1: Use Groq to extract text
        if model == "meta-llama/llama-4-maverick-17b-128e-instruct":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt}
            ]
        
        groq_payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        groq_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=groq_payload,
            timeout=60
        )
        
        if groq_response.status_code != 200:
            error_msg = f"Groq API error: {groq_response.status_code} - {groq_response.text}"
            return JSONResponse(content={"error": error_msg}, status_code=400)
        
        groq_result = groq_response.json()
        extracted_text = groq_result["choices"][0]["message"]["content"].strip()
        
        # Step2Use EasyOCR to get bounding boxes
        lang_list = [lang.strip() for lang in languages.split(',') if lang.strip()] if languages else ['en']
        reader = easyocr.Reader(lang_list, gpu=False)
        easyocr_results = reader.readtext(image_np)
        
        # Step 3tch Groq text with EasyOCR bounding boxes
        matched_results = []
        groq_text_lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        # If no lines from Groq, split by sentences or words
        if not groq_text_lines:
            groq_text_lines = [extracted_text]
        
        # For each line of text from Groq, find the best matching bounding box from EasyOCR
        for groq_line in groq_text_lines:
            best_match = None
            best_score = 0
            
            for bbox, text, conf in easyocr_results:
                # More flexible text similarity matching
                similarity = calculate_text_similarity(groq_line.lower(), text.lower())
                if similarity > best_score and similarity > 0.1:  # Lower threshold for matching
                    best_score = similarity
                    best_match = {
                       "bbox": [[float(x), float(y)] for x, y in bbox],
                        "text": groq_line,  # Use Groq's text (more accurate)
                   "confidence": float(conf),
                       "similarity_score": similarity
                    }
            
            if best_match:
                matched_results.append(best_match)
            else:
                # If no match found, create a placeholder bounding box
                matched_results.append({
                    "bbox": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "text": groq_line,
                    "confidence": 0.8,
                    "similarity_score": 0.0,
                    "note": "No bounding box found - using placeholder"
                })
        
        # If no matches found, return Groq text with placeholder bounding boxes
        if not matched_results:
            matched_results = [{
                "bbox": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "text": extracted_text,
                "confidence": 0.9,
                "similarity_score": 0.0,
                "note": "Using Groq text with placeholder bounding box"
            }]
        
        response_data = {
            "results": matched_results,
            "model_used": model,
            "extraction_method": "hybrid_groq_easyocr",
            "groq_text": extracted_text,
            "easyocr_detections": len(easyocr_results),
            "debug_info": {
                "groq_lines": len(groq_text_lines),
                "easyocr_boxes": len(easyocr_results),
                "matched_boxes": len([r for r in matched_results if r.get("similarity_score", 0) > 0])
            }
        }
        
        return JSONResponse(content=response_data)
        
    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": f"Network error: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Processing error: {str(e)}"}, status_code=500)

class GroqPromptResult(BaseModel):
    # You can adjust this schema to fit your expected output
    text: str
    # Add more fields if you expect structured output (e.g., fields, numbers, etc.)

@app.post("/groq-prompt")
async def groq_prompt_image(
    file: UploadFile = File(...),
    groq_api_key: str = Form(...),
    model: str = Form("meta-llama/llama-4-maverick-17b-128e-instruct"),
    prompt: str = Form(...)
):
    """
    Send an image and prompt to Groq vision model, validate/format response with Pydantic.
    No OCR is used; just pure LLM vision prompt.
    """
    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            error_msg = f"Groq API error: {response.status_code} - {response.text}"
            return JSONResponse(content={"error": error_msg}, status_code=400)
        result = response.json()
        model_content = result["choices"][0]["message"]["content"].strip()
        # Try to parse/validate with Pydantic
        try:
            validated = GroqPromptResult(text=model_content)
            return JSONResponse(content={"result": validated.dict(), "raw": model_content})
        except ValidationError as ve:
            return JSONResponse(content={"error": "Validation failed", "details": ve.errors(), "raw": model_content}, status_code=422)
    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": f"Network error: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Processing error: {str(e)}"}, status_code=500)

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    confidence: float = 0.9

class GroqBoundingBoxResult(BaseModel):
    bounding_boxes: List[BoundingBox]

@app.post("/groq-bounding-boxes")
async def groq_bounding_boxes(
    file: UploadFile = File(...),
    groq_api_key: str = Form(...),
    model: str = Form("meta-llama/llama-4-maverick-17b-128e-instruct"),
    prompt: Optional[str] = Form(None)
):
    """
    Extract bounding boxes from image using Groq vision model.
    Sends a prompt asking for bounding box coordinates in JSON format.
    """
    # Default prompt for bounding box extraction
    default_prompt = "Please analyze this image and extract all text regions with their bounding box coordinates. Return the result as a JSON object with the following format:\n\n{\n  \"bounding_boxes\": [\n    {\n      \"x1\": 10.5,\n      \"y1\": 203,\n      \"x2\": 150.2,\n      \"y2\": 457,\n      \"text\": \"extracted text here\",\n      \"confidence\": 0.95\n    }\n  ]\n}\n\nWhere:\n- x1, y1: corner coordinates\n- x2, y2: bottom-right corner coordinates\n- text: the text content in that region\n- confidence: confidence score (0.0 to 1.0)\n\nPlease be precise with the coordinates and extract all visible text regions."
    user_prompt = prompt if prompt else default_prompt
    
    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            error_msg = f"Groq API error: {response.status_code} - {response.text}"
            return JSONResponse(content={"error": error_msg}, status_code=400)
        
        result = response.json()
        model_content = result["choices"][0]["message"]["content"].strip()
        
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', model_content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                # Parse and validate with Pydantic
                parsed_json = json.loads(json_str)
                validated = GroqBoundingBoxResult(**parsed_json)
                
                # Convert to the format expected by the frontend
                frontend_results = []
                for box in validated.bounding_boxes:
                    # Convert from (x1,y1,x2y2) to [[x1,y1],x2,y1], [x2,y2], [x1,y2]] format
                    bbox_coords = [
                        [box.x1, box.y1],
                        [box.x2, box.y1], 
                        [box.x2, box.y2],
                        [box.x1, box.y2]
                    ]
                    frontend_results.append({
                        "bbox": bbox_coords,
                        "text": box.text,
                        "confidence": box.confidence
                    })
                
                return JSONResponse(content={
                    "results": frontend_results,
                    "model_used": model,
                    "extraction_method": "groq_vision_bounding_boxes",
                    "raw_response": model_content
                })
                
            except (json.JSONDecodeError, ValidationError) as e:
                return JSONResponse(content={
                    "error": "Failed to parse bounding boxes from model response",
                    "details": str(e),
                    "raw_response": model_content
                }, status_code=422)
            else:
                return JSONResponse(content={
                    "error": "No JSON found in model response",
                    "raw_response": model_content
                }, status_code=422)
            
    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": f"Network error: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Processing error: {str(e)}"}, status_code=500)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    Uses multiple similarity metrics for better matching.
    """
    if not text1 or not text2:
        return 0.0
    
    # Remove common punctuation and whitespace
    import re
    text1_clean = re.sub(r'\W+', ' ', text1.lower()).strip()
    text2_clean = re.sub(r'\W+', ' ', text2.lower()).strip()
    
    # Split into words
    words1 = set(text1_clean.split())
    words2 = set(text2_clean.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard = intersection / union if union > 0 else 0.0
    
    # Calculate substring similarity
    substring_score = 0.0
    if len(text1_clean) > 3 and len(text2_clean) > 3:
        if text1_clean in text2_clean or text2_clean in text1_clean:
            substring_score = 0.8
    
    # Calculate character-level similarity
    char_similarity = 0.0
    if len(text1_clean) > 0 and len(text2_clean) > 0:
        common_chars = len(set(text1_clean) & set(text2_clean))
        total_chars = len(set(text1_clean) | set(text2_clean))
        char_similarity = common_chars / total_chars if total_chars > 0 else 0.0
    
    # Return the maximum of all similarity scores
    return max(jaccard, substring_score, char_similarity) 