# backend/main.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import google.generativeai as genai
import ee
import geopandas as gpd
import geojson
import requests
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.colors as cl
from pydantic import BaseModel
import logging
from shapely.geometry import mapping
import datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Gemini AI
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
    generation_config=generation_config,
)

# Initialize PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
num_ftrs = pytorch_model.fc.in_features
pytorch_model.fc = torch.nn.Linear(num_ftrs, 10)

# Load PyTorch model weights
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "resnet-50-eurosat-model.pth")
logger.info(f"Loading PyTorch model from: {model_path}")

if not os.path.exists(model_path):
    logger.error(f"Model file not found at: {model_path}")
    raise FileNotFoundError(f"Model file not found at: {model_path}")

try:
    pytorch_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    pytorch_model.eval()
    logger.info("PyTorch model loaded successfully")
except Exception as e:
    logger.error(f"Error loading PyTorch model: {str(e)}")
    raise


# Initialize Earth Engine
try:
    # For service account authentication
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ee_credentials_path = os.path.join(BASE_DIR, "lulc-project-v288399work-026c21cf58ad.json")
    logger.info(f"Initializing Earth Engine with credentials from: {ee_credentials_path}")

    credentials = ee.ServiceAccountCredentials(
        'lulc-service-acc-v288388work@lulc-project-v288399work.iam.gserviceaccount.com',
        ee_credentials_path
    )
    ee.Initialize(credentials)
    logger.info("Earth Engine initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Earth Engine: {str(e)}", exc_info=True)
    raise


# LULC classes and colors
classes = [  # Updated class list
    'Urban', 'Agriculture', 'Forest', 'Water', 'Barren', 'Grassland', 'Shrubland', 'Wetlands', 'Ice/Snow', 'Clouds' 
]
colors = {  # Updated color map
    'Urban': 'gray', 'Agriculture': 'lightgreen', 'Forest': 'forestgreen', 'Water': 'blue', 'Barren': 'brown', 
    'Grassland': 'yellowgreen', 'Shrubland': 'olive', 'Wetlands': 'cyan', 'Ice/Snow': 'white', 'Clouds': 'lightgray', 'Unknown': '#000000'  # Added Unknown class with black color
}



# Image transformations
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




async def get_roi_from_prompt(prompt):  # Modified for google-generativeai

    chat_session = gemini_model.start_chat()  # Changed from model to gemini_model
    response = chat_session.send_message(
        f"""You are a helpful AI assistant specializing in Geographic Information Systems (GIS). 
        You can extract location information from user prompts related to land use and land cover classification.

        User Prompt: {prompt}

        Your Task:  Extract the region of interest.

        Return ONLY the name of the ROI.  If no specific region is found, return "India".
        """ # Your detailed prompt here
    )
    extracted_roi = response.text.strip()  # Get the text and remove whitespace

    return extracted_roi





async def get_geoboundary(iso_code: str, adm_level: str, roi: str):
    """
    Get the boundary for a region using geoBoundaries API
    
    Parameters:
        iso_code: str - Country code (e.g., 'IND')
        adm_level: str - Administrative level ('ADM0', 'ADM1', 'ADM2')
        roi: str - Region of interest name
    """
    logger.info(f"Fetching boundary for {roi} ({adm_level}) in {iso_code}")
    
    try:
        # For country-level requests (ADM0)
        if adm_level == "ADM0":
            url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso_code}/ADM0"
            r = requests.get(url)
            r.raise_for_status()
            download_path = r.json()["gjDownloadURL"]
            geoboundary_data = requests.get(download_path).json()
            region = gpd.GeoDataFrame.from_features(geoboundary_data["features"])
            logger.info(f"Retrieved country boundary for {iso_code}")
            return region
        
        # For state/district level requests
        url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso_code}/{adm_level}"
        r = requests.get(url)
        r.raise_for_status()
        
        download_path = r.json()["gjDownloadURL"]
        geoboundary_data = requests.get(download_path).json()
        geoboundary = gpd.GeoDataFrame.from_features(geoboundary_data["features"])
        
        # Search for the region (case-insensitive)
        region = geoboundary[
            geoboundary['shapeName'].str.lower().str.contains(roi.lower()) |
            geoboundary['shapeID'].str.lower().str.contains(roi.lower())
        ]
        
        if region.empty:
            # Try searching in alternate names if available
            if 'shapeGroup' in geoboundary.columns:
                region = geoboundary[
                    geoboundary['shapeGroup'].str.lower().str.contains(roi.lower())
                ]
            
            if region.empty:
                logger.error(f"Region '{roi}' not found in {iso_code} at {adm_level}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Region '{roi}' not found in {iso_code} at {adm_level}. Please check the spelling or try a different administrative level."
                )
        
        logger.info(f"Found region boundary for {roi}")
        return region.copy()  # Return a copy to avoid SettingWithCopyWarning
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from geoBoundaries API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing geoBoundaries API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_geoboundary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing boundary data: {str(e)}"
        )



async def generate_and_classify_tiles(region, image):
    tiles = [] 
    # Generate tiles within the region.  Adapt from your existing tile generation logic.
    bounds = region.total_bounds
    tile_size = 64  # Or whatever size you use


    for x in range(int(bounds[0]), int(bounds[2]), tile_size):
        for y in range(int(bounds[1]), int(bounds[3]), tile_size):
             bbox = box(x, y, x + tile_size, y + tile_size)
             tiles.append(bbox)


    predictions = []
    for tile in tiles:

        try:
            out_image, out_transform = mask(image, [tile], crop=True, all_touched=True)   #Use all_touched=True
            out_image = np.transpose(out_image, (1, 2, 0))

            # Convert masked image to PIL Image
            pil_image = Image.fromarray((out_image * 255).astype(np.uint8))
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():  # Disable gradient calculation during inference
               output = pytorch_model(input_tensor)  # Changed from model to pytorch_model
               _, pred = torch.max(output, 1)
               label = classes[pred.item()]  # Convert to class name
               predictions.append(label)



        except Exception as e:
            print(f"Error processing tile: {e}")  # Log the error
            predictions.append("Unknown")  # Or handle the error in another appropriate way


    return tiles, predictions


async def analyze_results(predictions):
    class_counts = {}
    for pred in predictions:
        class_counts[pred] = class_counts.get(pred, 0) + 1


    total_tiles = len(predictions)
    class_percentages = {cls: (count / total_tiles) * 100 for cls, count in class_counts.items()}


    # Gemini Summary generation
    chat_session = gemini_model.start_chat()  # Changed from model to gemini_model
    response = chat_session.send_message(
        f"""You are a helpful AI assistant that summarizes land use and land cover classification results.

        Classification Percentages: {class_percentages}

        Your Task: Provide a concise overall summary, demographic implications, and socio-lifestyle impacts.
        """
    )

    summary = response.text.strip()  # Get the summarized text
    return class_percentages, summary


class ClassificationRequest(BaseModel):
    iso_code: str
    adm_level: str
    roi: str

class ChatRequest(BaseModel):
    prompt: str
    analysis_data: dict

@app.post("/classify/")
async def classify_land_cover(request: ClassificationRequest):
    try:
        logger.info(f"Received classification request for {request.roi} in {request.iso_code} at {request.adm_level}")
        
        # Get region boundary
        region = await get_geoboundary(request.iso_code, request.adm_level, request.roi)
        logger.info(f"Got geoboundary for region: {region.shape}")
        
        # Earth Engine processing
        try:
            # Convert GeoDataFrame to proper GeoJSON format
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {}
                    }
                    for geom in region.geometry
                ]
            }
            
            region_ee = ee.FeatureCollection(geojson_data)
            logger.info("Created Earth Engine FeatureCollection")
            
            image = generate_image(region_ee)
            logger.info("Generated Earth Engine image")
            
            # Process image in chunks if it's too large
            if isinstance(image, ee.ImageCollection):
                logger.info("Processing large image in chunks")
                chunks = image.toList(image.size())
                final_image = ee.Image(chunks.get(0))
                for i in range(1, chunks.size().getInfo()):
                    chunk = ee.Image(chunks.get(i))
                    final_image = final_image.addBands(chunk)
                image = final_image
            
            # Generate and classify tiles
            tiles, predictions = await generate_and_classify_tiles(region, image)
            logger.info(f"Generated and classified {len(tiles)} tiles")
            
            # Analyze results
            class_percentages, summary = await analyze_results(predictions)
            logger.info("Completed analysis")
            
            # Calculate center coordinates for the map
            center_lat = region.geometry.centroid.y.mean()
            center_lon = region.geometry.centroid.x.mean()
            
            # Prepare response data
            response_data = {
                "roi": request.roi,
                "tiles": [mapping(tile) for tile in tiles],
                "predictions": predictions,
                "class_percentages": class_percentages,
                "summary": summary,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "colors": colors  # Use the complete colors dictionary instead of the hardcoded one
            }
            
            return response_data
            
        except ee.EEException as e:
            logger.error(f"Earth Engine error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Earth Engine processing failed: {str(e)}"
            )
        except ValueError as e:
            logger.error(f"Value error in processing: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input or processing error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in classify_land_cover: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in classify_land_cover: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )

@app.post("/chat/")
async def chat_about_analysis(request: ChatRequest):
    """Handle follow-up questions about the analysis"""
    try:
        chat_session = gemini_model.start_chat()
        response = chat_session.send_message(
            f"""You are analyzing land use and land cover data for a region.
            The analysis data shows: {request.analysis_data}
            
            User question: {request.prompt}
            
            Provide a helpful response based on the analysis data."""
        )
        return {"response": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_image(region_ee):
    """Generate Earth Engine image for the region"""
    try:
        # Get current date and 3 months ago date
        now = ee.Date(datetime.datetime.now())
        three_months_ago = now.advance(-3, 'month')
        
        # Get Sentinel-2 image collection
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region_ee) \
            .filterDate(three_months_ago, now) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .first()
        
        if s2 is None:
            raise HTTPException(
                status_code=500,
                detail="No Sentinel-2 imagery found for this region in the last 3 months"
            )
        
        # Select the bands we need (RGB)
        rgb_image = s2.select(['B4', 'B3', 'B2'])
        
        # Calculate bounds and dimensions
        bounds = region_ee.geometry().bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        # Calculate width and height in meters
        width = abs(coords[2][0] - coords[0][0])
        height = abs(coords[2][1] - coords[0][1])
        
        # Calculate appropriate scale to keep dimensions under limits
        max_pixels = 32768
        scale = max(width / max_pixels, height / max_pixels)
        scale = max(10, scale)  # Ensure minimum 10m resolution
        
        logger.info(f"Using scale of {scale} meters per pixel")
        
        # Clip to region and normalize
        clipped = rgb_image.clip(region_ee)
        normalized = clipped.divide(10000)  # Scale factor for Sentinel-2
        
        # Split large regions into tiles if necessary
        if width * height > 50331648:  # Max request size
            logger.info("Large region detected, processing in tiles")
            geometry = region_ee.geometry()
            tiles = geometry.cut(width / scale / 2, height / scale / 2)
            
            # Process each tile
            def process_tile(tile):
                return normalized.clip(tile)
            
            tiled_image = ee.ImageCollection(tiles.map(process_tile)).mosaic()
            return tiled_image
        else:
            return normalized
            
    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing satellite imagery: {str(e)}"
        )








