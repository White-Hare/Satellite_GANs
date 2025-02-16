import ee
import random
import os

# Initialize the Earth Engine module.
def init_ee():
    ee.Authenticate()
    ee.Initialize(project='satellitegans')

def get_image_url(lon_min, lat_min, lon_max, lat_max):
    # Define a region of interest with the square bounding box
    roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    # Use Sentinel-2 collection instead
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))   # Stricter cloud filtering
        .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    # Get size of collection
    count = collection.size().getInfo()
    if count == 0:
        return None
        
    image = collection.first()
    
    # Reproject the image to a common coordinate system (e.g., EPSG:4326)
    reprojected_image = image.reproject(crs='EPSG:4326', scale=10)
    
    # Select the RGB bands for Sentinel-2 (different band names)
    formated_image = reprojected_image.select(['B4', 'B3', 'B2'])

    # Get the URL for the image with visualization parameters
    url = formated_image.getThumbURL({
        'region': roi,
        'format': 'png',
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    })
    
    return url

def get_random_satellite_images(num_locations=5):
    # Fixed size for the bounding box (in degrees)
    BOX_SIZE = 0.1
    
    result = []
    for _ in range(num_locations):
        # Generate random coordinates for the bounding box
        lon_min = random.uniform(-180, 180)
        lat_min = random.uniform(-90, 90)
        lon_max = lon_min + BOX_SIZE
        lat_max = lat_min + BOX_SIZE
        
        url = get_image_url(lon_min, lat_min, lon_max, lat_max)
        
        if url is not None:
            result.append(url)
    
    return result