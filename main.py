import asyncio
import cv2
import easyocr
import re
import numpy as np
import time
import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VIN Scanner API",
    description="Fast VIN detection from images using OCR - Hosted on Render.com",
    version="1.0.0"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# VIN format rule (no I, O, Q allowed)
VIN_REGEX = r'\b(?!.*[IOQ])[A-HJ-NPR-Z0-9]{17}\b'

# Character corrections for OCR misreads
corrections = {
    'S': '5', 'Z': '2', 'O': '0', 'B': '8', 'G': '6', 'I': '1'
}

# Global EasyOCR reader
reader = None

class VINResponse(BaseModel):
    vin: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: float
    detected_texts: list = []
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    memory_usage: str
    platform: str

def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f}MB"
    except:
        return "unknown"

def initialize_ocr():
    """Initialize EasyOCR reader"""
    global reader
    try:
        logger.info("🔄 Initializing EasyOCR reader for Render.com...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False, download_enabled=True)
        logger.info("✅ EasyOCR reader initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize EasyOCR: {e}")
        reader = None
        return False

def is_valid_vin(text: str) -> Optional[str]:
    """Check if text matches VIN format"""
    text = text.upper()
    match = re.search(VIN_REGEX, text)
    return match.group() if match else None

def correct_vin(text: str) -> str:
    """Apply character corrections to potential VIN"""
    corrected = ''
    for char in text.upper():
        corrected += corrections.get(char, char)
    return corrected

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction
        blur = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(blur)
        
        # Apply threshold
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize for better OCR
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        
        return resized
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        raise

async def process_image_async(image_data: bytes) -> VINResponse:
    """Process image asynchronously to detect VIN"""
    start_time = time.time()
    
    try:
        # Check if OCR is available
        if reader is None:
            raise ValueError("OCR reader not initialized. Please try again in a few moments.")
        
        # File size limit for Render.com free tier
        if len(image_data) > 5 * 1024 * 1024:  # 5MB limit
            raise ValueError("Image too large. Please use images under 5MB.")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image. Please check image format.")
        
        # Resize if too large (memory optimization for Render.com)
        height, width = image.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920/width, 1080/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"📏 Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run OCR with timeout (Render.com has request timeout)
        try:
            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: reader.readtext(processed_image)),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            raise ValueError("OCR processing timeout. Try a clearer, smaller image.")
        
        detected_texts = []
        best_vin = None
        best_confidence = 0.0
        
        # Process OCR results
        for bbox, text, confidence in results:
            detected_texts.append({"text": text, "confidence": round(confidence, 3)})
            
            # Clean text
            original = text.replace(" ", "").replace("-", "").upper()
            
            # Check if it's already a valid VIN
            vin = is_valid_vin(original)
            if vin and confidence > best_confidence:
                best_vin = vin
                best_confidence = confidence
                logger.info(f"✅ Found valid VIN: {vin} (confidence: {confidence:.3f})")
                continue
            
            # If text is 17 characters, try correction
            if len(original) == 17:
                corrected = correct_vin(original)
                vin = is_valid_vin(corrected)
                if vin and confidence > best_confidence:
                    best_vin = vin
                    best_confidence = confidence
                    logger.info(f"✅ Found corrected VIN: {vin} from {original} (confidence: {confidence:.3f})")
        
        processing_time = time.time() - start_time
        
        return VINResponse(
            vin=best_vin,
            confidence=round(best_confidence, 3) if best_vin else None,
            processing_time=round(processing_time, 2),
            detected_texts=detected_texts[:10]  # Limit for response size
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error processing image: {e}")
        return VINResponse(
            processing_time=round(processing_time, 2),
            error=str(e)
        )

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting VIN Scanner API on Render.com...")
    
    # Initialize OCR in background to avoid startup timeout
    async def init_ocr_background():
        await asyncio.sleep(1)  # Small delay
        await asyncio.to_thread(initialize_ocr)
    
    asyncio.create_task(init_ocr_background())

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if reader is not None else "initializing",
        timestamp=time.time(),
        memory_usage=get_memory_usage(),
        platform="Render.com"
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚗 VIN Scanner API",
        "version": "1.0.0",
        "platform": "Render.com Free Tier",
        "status": "ready" if reader is not None else "initializing OCR...",
        "memory": get_memory_usage(),
        "endpoints": {
            "health": "/health",
            "scan_vin": "/read-vin",
            "documentation": "/docs",
            "openapi": "/openapi.json"
        },
        "limits": {
            "max_file_size": "5MB",
            "timeout": "25 seconds",
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
        }
    }

@app.post("/read-vin", response_model=VINResponse)
async def read_vin(file: UploadFile = File(...)):
    """
    Extract VIN from uploaded image
    
    Upload an image containing a VIN number and get the detected VIN with confidence score.
    
    - **file**: Image file (JPEG, PNG, BMP, TIFF) - Max 5MB
    
    Returns:
    - **vin**: Detected VIN number (17 characters)
    - **confidence**: OCR confidence score (0-1)
    - **processing_time**: Time taken to process
    - **detected_texts**: All texts found in image
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, BMP, TIFF)"
            )
        
        # Read file data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        logger.info(f"📸 Processing image: {file.filename}, size: {len(image_data)} bytes")
        
        # Process image
        result = await process_image_async(image_data)
        
        # Return appropriate status code
        if result.error:
            return JSONResponse(
                status_code=422,
                content=result.dict()
            )
        
        if result.vin:
            logger.info(f"🎉 Successfully detected VIN: {result.vin}")
        else:
            logger.info(f"⚠️ No VIN found in image, detected {len(result.detected_texts)} texts")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error in read_vin: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"🚨 Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# For Render.com deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )