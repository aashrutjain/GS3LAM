import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import json
import os
import io
import PIL.Image
from plyfile import PlyData, PlyElement
import numpy.lib.recfunctions as rfn

# Import the modern SDK
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = "AIzaSyAtg_2fuoc1AA2UfUJuFPNrk0CYkIzeM78" # Your key

# Initialize the modern client
client = genai.Client(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# UPDATE THESE TO YOUR NEW REPLICA DATA!
PLY_PATH = "logs/Replica/room0_seed1/260422-13:33:16/gsplat.ply"
PARAMS_PATH = "logs/Replica/room0_seed1/260422-13:33:16/params.npz"
CLASSIFIER_PATH = "logs/Replica/room0_seed1/260422-13:33:16/classifier.pth"
TUM_RGB_DIR = "data/Replica/room0/results/" # Replica stores its frames here
OUTPUT_PLY_PATH = "logs/Replica/room0_seed1/260422-13:33:16/safety_gsplat.ply"

# ==========================================
# 1. LOAD MAP & CLASSIFIER
# ==========================================

import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData

class SemanticDecoder(nn.Module):
    def __init__(self, in_channels=16, out_channels=256):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        return self.linear(x)

def load_and_classify_splats():
    print("Loading 3D Gaussians and Classifier...")
    plydata = PlyData.read(PLY_PATH)
    
    # 1. Extract XYZ coordinates from the PLY structure (so we can save it back later)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    points_3d = np.vstack((x, y, z)).T
    
    # 2. Setup the PyTorch Classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = SemanticDecoder().to(device)
    
    state_dict = torch.load(CLASSIFIER_PATH, map_location=device)
    classifier.load_state_dict(state_dict, strict=False)
    classifier.eval()
    
    # 3. Extract the 16 Semantic Features from params.npz
    print("Loading semantic features from params.npz...")
    params = np.load(PARAMS_PATH)
    semantic_features = params['obj_dc']
    
    # Flatten the extra spherical harmonic dimension if it exists (e.g., N, 1, 16 -> N, 16)
    if len(semantic_features.shape) == 3:
        semantic_features = semantic_features.squeeze(1)
        
    # 4. Push through the model to get the true Class IDs
    print("Decoding latent semantic vectors into discrete Class IDs...")
    semantic_tensor = torch.tensor(semantic_features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = classifier(semantic_tensor)
        class_ids = torch.argmax(logits, dim=1).cpu().numpy()
        
    unique_classes_found = np.unique(class_ids)
    print(f"Successfully decoded! Found {len(unique_classes_found)} unique object classes in the room.")
    
    return points_3d, class_ids, plydata

# ==========================================
# 2. FIND HERO FRAME & MASK (REPLICA DATASET VERSION)
# ==========================================
def extract_canonical_view(target_class_id, points_3d, class_ids):
    print(f"Finding Hero Frame for Object Class: {target_class_id}")
   
    params = np.load(PARAMS_PATH)
    w2c_matrices = params['w2c'].reshape(-1, 4, 4)
    keyframe_indices = params['keyframe_time_indices']
   
    # 1. Handle the Intrinsics Matrix properly
    intrinsics = params['intrinsics']
    if intrinsics.shape == (4,): # If they saved it as [fx, fy, cx, cy]
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        K = intrinsics
       
    # 2. Parse the Replica dataset images
    # Replica simply uses sequentially numbered frames, no text file needed
    image_paths = sorted(glob.glob(os.path.join(TUM_RGB_DIR, "*.jpg")))
   
    if not image_paths:
        raise FileNotFoundError(f"Could not find any .jpg images in {TUM_RGB_DIR}. Check your path!")

    obj_points_3d = points_3d[class_ids == target_class_id]
   
    best_frame_idx = -1
    max_pixel_footprint = 0
    best_2d_points = None
   
    # Find the frame where the object takes up the most pixels
    for frame_idx, w2c in enumerate(w2c_matrices):
        points_c = (w2c[:3, :3] @ obj_points_3d.T).T + w2c[:3, 3]
       
        valid_mask = points_c[:, 2] > 0.1
        points_c = points_c[valid_mask]
       
        if len(points_c) == 0: continue
       
        points_2d = (K @ points_c.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
       
        min_x, max_x = np.min(points_2d[:,0]), np.max(points_2d[:,0])
        min_y, max_y = np.min(points_2d[:,1]), np.max(points_2d[:,1])
        area = (max_x - min_x) * (max_y - min_y)
       
        if area > max_pixel_footprint:
            max_pixel_footprint = area
            best_frame_idx = frame_idx
            best_2d_points = points_2d
           
    # 3. Direct Indexing for Replica
    actual_image_idx = keyframe_indices[best_frame_idx]
    hero_img_path = image_paths[actual_image_idx]
   
    img = cv2.imread(hero_img_path)
   
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(best_2d_points.astype(np.float32))
    cv2.fillConvexPoly(mask, np.int32(hull), 255)
   
    cropped_img = cv2.bitwise_and(img, img, mask=mask)
   
    _, buffer = cv2.imencode('.jpg', cropped_img)
    return base64.b64encode(buffer).decode('utf-8')


# ==========================================
# 3. GEMINI 2.5 SAFETY AUDIT (MODERN SDK)
# ==========================================
def query_vlm_safety(base64_image):
    print("Querying Gemini 2.5 Flash for Safety Score...")
   
    # Convert the base64 string back into a PIL Image for Gemini
    img_bytes = base64.b64decode(base64_image)
    img = PIL.Image.open(io.BytesIO(img_bytes))
   
    # The physical reasoning prompt
    prompt = """
    You are a physical safety auditor for a 3kg wheeled robot (TurtleBot4).
    Analyze the physical materials, structure, and stability of the isolated object in this image.
    Output a strictly formatted JSON dictionary with a single key 'safety_score', holding a float from 0.0 (lethal hazard/fragile/easily tipped/cables) to 1.0 (completely safe to drive on/flat solid ground).
    """
   
    # Call Gemini using the modern 2026 syntax
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2 # Keep it deterministic and analytical
        )
    )
   
    # Parse the guaranteed JSON response
    result = json.loads(response.text)
    return float(result['safety_score'])

# ==========================================
# 4. BROADCAST & SAVE
# ==========================================
def broadcast_scores_and_save(plydata, class_ids, safety_dict):
    print("Broadcasting scores to 3D map...")
    num_points = len(class_ids)
    safety_array = np.zeros(num_points, dtype=np.float32)
   
    # Paint the map with the VLM scores
    for obj_id, score in safety_dict.items():
        safety_array[class_ids == obj_id] = score
       
    # Append the new 'safety' parameter to the PLY structure
    new_data = rfn.append_fields(
        plydata.elements[0].data, 
        'safety', 
        safety_array, 
        dtypes=np.float32, 
        usemask=False
    )
   
    # Save the physics-ready 3D map
    new_element = PlyElement.describe(new_data, 'vertex')
    PlyData([new_element], text=False).write(OUTPUT_PLY_PATH)
    print(f"Success! Map saved to {OUTPUT_PLY_PATH}")

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    points_3d, class_ids, plydata = load_and_classify_splats()
    
    # Count how many points belong to each class
    unique_objects, counts = np.unique(class_ids, return_counts=True)
    
    # Sort by frequency and isolate the Top 5 largest objects in the room
    top_indices = np.argsort(-counts)[:5]
    top_objects = unique_objects[top_indices]
    
    print(f"\nFiltering noisy data... Auditing the Top {len(top_objects)} largest objects.")
    
    safety_dictionary = {}
    
    for obj_id in top_objects:
        try:
            b64_img = extract_canonical_view(obj_id, points_3d, class_ids)
            score = query_vlm_safety(b64_img)
            safety_dictionary[obj_id] = score
            print(f"-> Object {obj_id} Safety Score: {score}")
        except Exception as e:
            print(f"-> Skipping Object {obj_id} due to projection error: {e}")
            safety_dictionary[obj_id] = 1.0 # Default to safe if it fails
        
    broadcast_scores_and_save(plydata, class_ids, safety_dictionary)