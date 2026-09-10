import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import json
import os
import io
import glob
import PIL.Image
from plyfile import PlyData, PlyElement
import numpy.lib.recfunctions as rfn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData

from src.Decoder import SemanticDecoder as TrainedSemanticDecoder
from src.utils.gaussian_utils import build_rotation

# Modern SDK
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Congigs
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# Pinned GA model (not the experimental 'gemini-flash-latest' rolling alias) so the
# model reported in the write-up is reproducible on re-runs. One constant, referenced
# at the single call site below.
GEMINI_MODEL = "gemini-3.5-flash"

PLY_PATH = "logs/Replica/room0_seed1/260422-13:33:16/gsplat.ply"
PARAMS_PATH = "logs/Replica/room0_seed1/260422-13:33:16/params.npz"
CLASSIFIER_PATH = "logs/Replica/room0_seed1/260422-13:33:16/classifier.pth"
TUM_RGB_DIR = "data/Replica/room0/results/"
OUTPUT_PLY_PATH = "logs/Replica/room0_seed1/260422-13:33:16/safety_gsplat.ply"

# Must match configs/Replica/room0.py's semantic.num_objects/num_classes — these are
# the exact args src/GS3LAM.py:103 used to build the classifier that was trained and
# saved as classifier.pth.
SEMANTIC_IN_CHANNELS = 16
SEMANTIC_OUT_CHANNELS = 256

def load_and_classify_splats():
    print("Loading 3D Gaussians and Classifier...")
    plydata = PlyData.read(PLY_PATH)
    
    # Extract XYZ coordinates from the PLY structure
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    points_3d = np.vstack((x, y, z)).T
    
    # Setup the PyTorch Classifier — reuse the actual trained architecture (src/Decoder.py)
    # instead of reimplementing it, so the state_dict keys can never silently drift again.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = TrainedSemanticDecoder(SEMANTIC_IN_CHANNELS, SEMANTIC_OUT_CHANNELS).to(device)

    state_dict = torch.load(CLASSIFIER_PATH, map_location=device)
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()

    # Extract the 16 Semantic Features from params.npz
    print("Loading semantic features from params.npz...")
    params = np.load(PARAMS_PATH)
    semantic_features = params['obj_dc']

    # Flatten the extra spherical harmonic dimension if it exists (e.g., N, 1, 16 -> N, 16)
    if len(semantic_features.shape) == 3:
        semantic_features = semantic_features.squeeze(1)

    # Push through the model to get the true Class IDs
    print("Decoding latent semantic vectors into discrete Class IDs...")
    semantic_tensor = torch.tensor(semantic_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        # The trained decoder is a 1x1 conv over per-pixel (C,H,W) feature maps in normal
        # use (src/Evaluater.py, src/Loss.py). A 1x1 conv has no spatial mixing, so it's
        # mathematically a per-vector linear map — reshape (N,16) -> (N,16,1,1) to reuse it
        # correctly per splat, then squeeze the logits back to (N,256).
        conv_input = semantic_tensor.view(-1, SEMANTIC_IN_CHANNELS, 1, 1)
        logits = classifier(conv_input).view(-1, SEMANTIC_OUT_CHANNELS)
        class_ids = torch.argmax(logits, dim=1).cpu().numpy()
        
    unique_classes_found = np.unique(class_ids)
    print(f"Successfully decoded! Found {len(unique_classes_found)} unique object classes in the room.")
    
    return points_3d, class_ids, plydata

class HeroFrameSelectionError(RuntimeError):
    """A hero-frame selection failure, as distinct from a VLM failure.

    Kept as its own exception type so the top-level loop can report a pose/geometry
    failure separately from a Gemini failure. Both used to collapse into one generic
    "projection/VLM error" line, which is precisely what let the single-candidate-pose
    bug below go unnoticed -- an unusable pose set and a dead API key produced the
    same log line and the same 0.0 score.
    """


def load_per_frame_w2c(params):
    """Reconstruct the per-frame world-to-camera matrices from GS3LAM's saved params.

    `params['w2c']` is NOT the camera path, despite its name: src/GS3LAM.py:481
    assigns it `first_frame_w2c`, a single (4,4) matrix, and it is the only line in
    that file that touches the key. Because the dataset is constructed with
    `relative_pose=True` (src/GS3LAM.py:63 -> src/datasets/basedataset.py:156,228,
    "setting first pose in a sequence to identity"), that single matrix is the
    identity. Reading it as a trajectory -- `params['w2c'].reshape(-1, 4, 4)` --
    silently yields shape (1,4,4), so the hero-frame loop had exactly one candidate
    pose and selected nothing.

    The real per-frame poses are `cam_unnorm_rots` (1,4,num_frames) and `cam_trans`
    (1,3,num_frames), both world-to-camera relative to frame 0. They are genuinely
    persisted: src/utils/logger.py:14-21 (`params2cpu`) converts every key without a
    whitelist, and src/GaussianManager.py:54 explicitly excludes both from per-splat
    pruning, so they survive the run at full length. These are the *estimated* poses
    (GS3LAM's own tracking output), which is what Stage 2 should consume;
    `gt_w2c_all_frames` is ground truth and would make Stage 2 depend on data a real
    robot does not have.

    The quaternion -> rotation conversion reuses src/utils/gaussian_utils.build_rotation,
    exactly as the in-repo call sites do (src/GS3LAM.py:415-419 is the reference,
    mirrored at src/Mapper.py:217-221), rather than reimplementing the (w,x,y,z)
    convention here. Reimplementing a convention that already exists in the repo is
    the same mistake that produced the earlier SemanticDecoder state_dict bug.

    Runs on CPU or GPU: build_rotation follows its input's device as of 2026-09-10.
    """
    if 'cam_unnorm_rots' not in params or 'cam_trans' not in params:
        raise HeroFrameSelectionError(
            f"{PARAMS_PATH} has no 'cam_unnorm_rots'/'cam_trans', so the camera path "
            "cannot be rebuilt. 'w2c' is not a usable substitute -- it is a single "
            "first-frame matrix, not a trajectory (src/GS3LAM.py:481)."
        )

    cam_unnorm_rots = np.asarray(params['cam_unnorm_rots'])  # (1, 4, num_frames)
    cam_trans = np.asarray(params['cam_trans'])              # (1, 3, num_frames)
    num_frames = cam_unnorm_rots.shape[-1]

    # build_rotation() follows its input's device as of the 2026-09-10 fix (it used to
    # hardcode device='cuda'), so this path is no longer GPU-only. Pick the device the
    # same way the rest of this script does.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q = torch.tensor(cam_unnorm_rots, dtype=torch.float32, device=device)[0].permute(1, 0)
    t = torch.tensor(cam_trans, dtype=torch.float32, device=device)[0].permute(1, 0)

    with torch.no_grad():
        # F.normalize then build_rotation, then R into the top-left block and t into the
        # last column: the same four steps as src/GS3LAM.py:415-419, batched over frames.
        rot = build_rotation(F.normalize(q, dim=-1))          # (num_frames, 3, 3)
        w2c = torch.eye(4, device=rot.device).repeat(num_frames, 1, 1)
        w2c[:, :3, :3] = rot
        w2c[:, :3, 3] = t

    return w2c.detach().cpu().numpy()


def load_native_resolution_K(params, native_width, native_height):
    """Return the saved intrinsics rescaled to the resolution of the on-disk RGB frames.

    The K in params.npz is at the *downsampled training* resolution, not the native
    one: src/datasets/basedataset.py:295 applies
    `datautils.scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)`
    before the dataset hands the frame over, and src/GS3LAM.py:480 saves that already
    scaled matrix. This script, however, loads the ORIGINAL frame*.jpg files at their
    native size, so K has to be scaled back up -- otherwise the projected convex hull
    lands in the wrong region of the image and the VLM is shown the wrong pixels.

    Upscaling K is deliberate in preference to downscaling the images: the VLM then
    reasons about a full-resolution crop, which is the input quality the safety score
    is supposed to reflect.

    The ratio is derived from the data, never hardcoded. `params['org_width']` and
    `params['org_height']` are -- despite the "org" name -- the *desired*, i.e.
    downsampled, dimensions: src/GS3LAM.py:482-483 assigns them straight from
    dataset_config["desired_image_width"/"desired_image_height"]. Dividing the actual
    on-disk frame size by those recovers the exact inverse of the downsample that was
    applied. For the committed Replica config that is exactly 2.0 x 2.0
    (configs/camera/replica.yaml 1200x680 against configs/Replica/room0.py 600x340),
    but nothing here depends on that value.

    Only fx, fy, cx, cy are touched, mirroring datautils.scale_intrinsics:112-115.
    """
    intrinsics = np.asarray(params['intrinsics'])
    if intrinsics.shape == (4,): # If they saved it as [fx, fy, cx, cy]
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        K = intrinsics[:3, :3].astype(np.float64).copy()

    if 'org_width' not in params or 'org_height' not in params:
        raise HeroFrameSelectionError(
            f"{PARAMS_PATH} has no 'org_width'/'org_height', so the training resolution "
            "is unknown and the saved intrinsics cannot be rescaled to the native "
            "frames. Refusing to guess a scale factor (src/GS3LAM.py:482-483)."
        )

    trained_width = int(params['org_width'])
    trained_height = int(params['org_height'])
    if trained_width <= 0 or trained_height <= 0:
        raise HeroFrameSelectionError(
            f"Nonsensical training resolution in {PARAMS_PATH}: "
            f"{trained_width}x{trained_height}."
        )

    scale_x = native_width / trained_width
    scale_y = native_height / trained_height

    K[0, 0] *= scale_x  # fx
    K[1, 1] *= scale_y  # fy
    K[0, 2] *= scale_x  # cx
    K[1, 2] *= scale_y  # cy

    return K, scale_x, scale_y


# Hero Frame and Mask
def extract_canonical_view(target_class_id, points_3d, class_ids):
    print(f"Finding Hero Frame for Object Class: {target_class_id}")

    params = np.load(PARAMS_PATH)
    w2c_matrices = load_per_frame_w2c(params)
    num_frames = w2c_matrices.shape[0]

    # Parse the Replica dataset images
    image_paths = sorted(glob.glob(os.path.join(TUM_RGB_DIR, "*.jpg")))

    if not image_paths:
        raise FileNotFoundError(f"Could not find any .jpg images in {TUM_RGB_DIR}. Check your path!")

    # frame_idx below is a dataset time_idx, and with the committed Replica config
    # (start=0, stride=1 -- configs/Replica/room0.py:50-52) time_idx indexes frame*.jpg
    # 1:1. Check that rather than assume it: a strided or offset run would silently pair
    # every pose with the wrong image, which is not a failure that shows up in the crop.
    if len(image_paths) < num_frames:
        raise HeroFrameSelectionError(
            f"{len(image_paths)} frame*.jpg files in {TUM_RGB_DIR} but {num_frames} camera "
            "poses in params.npz. Pose index maps to image index 1:1 only when the dataset "
            "was built with start=0 and stride=1; either it was not, or frames are missing."
        )

    # Intrinsics must match the resolution of the images actually being loaded, not the
    # resolution GS3LAM trained at. PIL reads the header only here -- no full decode.
    native_width, native_height = PIL.Image.open(image_paths[0]).size
    K, scale_x, scale_y = load_native_resolution_K(params, native_width, native_height)
    print(f"   Intrinsics rescaled to native {native_width}x{native_height} "
          f"(x{scale_x:.4g}, y{scale_y:.4g})")

    obj_points_3d = points_3d[class_ids == target_class_id]

    if len(obj_points_3d) == 0:
        raise HeroFrameSelectionError(
            f"No splats carry class id {target_class_id}; there is nothing to project."
        )

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
           
    if best_frame_idx == -1:
        # Previously this fell straight through to keyframe_indices[best_frame_idx],
        # which numpy resolves as keyframe_indices[-1] -- the LAST keyframe -- and then
        # died on best_2d_points being None, landing in the generic except block as an
        # indistinguishable "projection/VLM error". Raise a typed, specific error instead.
        raise HeroFrameSelectionError(
            f"Hero-frame selection failed for class {target_class_id}: none of the "
            f"{num_frames} camera poses produced a non-degenerate projection of its "
            f"{len(obj_points_3d)} splats (all points either behind the camera, z <= 0.1, "
            "or projecting to zero bounding-box area). This is a pose/geometry failure, "
            "not a VLM failure."
        )

    # best_frame_idx is a dataset time_idx, which indexes frame*.jpg directly under
    # start=0/stride=1 (checked above). It is NOT a position in keyframe_time_indices:
    # that indirection belonged to the old single-pose loop and would now select the
    # wrong image entirely.
    hero_img_path = image_paths[best_frame_idx]
    print(f"   Hero frame: time_idx {best_frame_idx} -> {os.path.basename(hero_img_path)} "
          f"(footprint {max_pixel_footprint:.0f} px^2)")

    img = cv2.imread(hero_img_path)
   
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(best_2d_points.astype(np.float32))
    cv2.fillConvexPoly(mask, np.int32(hull), 255)
   
    cropped_img = cv2.bitwise_and(img, img, mask=mask)
   
    _, buffer = cv2.imencode('.jpg', cropped_img)
    return base64.b64encode(buffer).decode('utf-8')


# Gemini Flash Safety Call
def query_vlm_safety(base64_image):
    print(f"Querying {GEMINI_MODEL} for Safety Score...")
   
    # Convert the base64 string back into a PIL Image for Gemini
    img_bytes = base64.b64decode(base64_image)
    img = PIL.Image.open(io.BytesIO(img_bytes))
   
    # The physical reasoning prompt
    prompt = """
    You are a physical safety auditor for a 3kg wheeled robot (TurtleBot4).
    Analyze the physical materials, structure, and stability of the isolated object in this image.
    Output a strictly formatted JSON dictionary with a single key 'safety_score', holding a float from 0.0 (lethal hazard/fragile/easily tipped/cables) to 1.0 (completely safe to drive on/flat solid ground).
    """
   
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    )
   
    # Parse the guaranteed JSON response
    result = json.loads(response.text)
    return float(result['safety_score'])

# Broadcastin and saving
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

# Execution
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
        except HeroFrameSelectionError as e:
            # Reported separately from a VLM failure on purpose: these two have very
            # different causes and very different fixes, and collapsing them into one
            # log line is what hid the single-candidate-pose bug (see PROGRESS.md).
            print(f"-> Skipping Object {obj_id}: HERO-FRAME SELECTION FAILED (not a VLM error): {e}")
            safety_dictionary[obj_id] = 0.0
        except Exception as e:
            print(f"-> Skipping Object {obj_id} due to VLM/other error: {e}")
            # Conservative fail-safe: a failed hero-frame projection or VLM call means we
            # have NO valid safety judgment for this object, so default to 0.0 (treat as a
            # hazard), matching the 0.0 default already applied to unqueried splats in
            # broadcast_scores_and_save(). This was previously 1.0 ("completely safe"), an
            # inverted fail-safe that told the CBF to drive over an object we knew nothing
            # about. Changed 2026-07-20 — safety-score-scale decision, flagged per CLAUDE.md
            # and recorded in PROGRESS.md.
            safety_dictionary[obj_id] = 0.0
        
    broadcast_scores_and_save(plydata, class_ids, safety_dictionary)
