#!/usr/bin/env python3
# Enhanced inference with AWS Rekognition face recognition, activity tracking, and cost optimization
import os, io, json, time, base64
import numpy as np
import cv2
from ultralytics import YOLO
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import traceback
from datetime import datetime

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logger = logging.getLogger("inference")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

LOG_FILE = os.environ.get('LOG_FILE')
if LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {LOG_FILE}")

logger.info("=" * 60)
logger.info("Enhanced Inference Engine Starting")
logger.info("=" * 60)

# =====================================================
# AWS & DATABASE CONFIGURATION
# =====================================================
REKOGNITION_ENABLED = os.environ.get('REKOGNITION_ENABLED', '1') == '1'
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
COLLECTION_ID = os.environ.get('REKOGNITION_COLLECTION_ID', 'EmployeeCollection')
FACE_MATCH_THRESHOLD = float(os.environ.get('FACE_MATCH_THRESHOLD', '80.0'))

# Rekognition optimization
REKOGNITION_INTERVAL_FRAMES = int(os.environ.get('REKOGNITION_INTERVAL_FRAMES', '1800'))  # 60 sec at 30fps
TRACKING_IOU_THRESHOLD = float(os.environ.get('TRACKING_IOU_THRESHOLD', '0.3'))

# Database config
DB_HOST = os.environ.get('DB_HOST', '54.225.63.242')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'visco')
DB_USER = os.environ.get('DB_USER', 'visco_cctv')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'Visco@0408')

# Global clients
rekognition_client = None
db_connection = None
employee_cache = {}

# Global tracking and timing state
person_tracker = {
    'last_recognition_frame': 0,
    'tracked_persons': {}  # {track_id: {employee_info, bbox, last_frame}}
}

employee_timers = {}  # {employee_id: {working_seconds, idle_seconds, first_timestamp, last_timestamp, last_frame, current_activity}}

current_frame_number = 0
video_fps = 30.0  # Default, will be updated from input if available


# =====================================================
# ACTIVITY CLASSIFICATION (from second file)
# =====================================================
def classify_activity(keypoints, confidence_threshold=0.3):
    """
    Improved worker activity classification based on pose keypoints.
    
    Args:
        keypoints: List of keypoints in format [[x, y, conf], ...]
        confidence_threshold: Minimum confidence for keypoint validity
    
    Returns:
        str: Classification label ("WORKING", "IDLE", or "UNCERTAIN")
    """
    # Convert to numpy array and extract x, y coordinates
    kpts = np.array([[kp[0], kp[1]] for kp in keypoints])
    
    # Check if keypoints array is valid
    if kpts.shape[0] < 17:
        return "UNCERTAIN"
    
    # Extract keypoints with indices
    head = kpts[0]          # Nose
    L_sh, R_sh = kpts[5], kpts[6]      # Shoulders
    L_elb, R_elb = kpts[7], kpts[8]    # Elbows
    L_wr, R_wr = kpts[9], kpts[10]     # Wrists
    L_hp, R_hp = kpts[11], kpts[12]    # Hips
    L_kn, R_kn = kpts[13], kpts[14]    # Knees
    
    # Check if critical keypoints are detected (non-zero)
    critical_points = [head, L_sh, R_sh, L_hp, R_hp, L_wr, R_wr]
    if any(np.all(pt == 0) for pt in critical_points):
        return "UNCERTAIN"
    
    # Calculate midpoints
    hip = (L_hp + R_hp) / 2
    knee = (L_kn + R_kn) / 2
    shoulder = (L_sh + R_sh) / 2
    wrist_mid = (L_wr + R_wr) / 2
    elbow_mid = (L_elb + R_elb) / 2
    
    # Calculate body dimensions
    body_height = abs(shoulder[1] - hip[1]) + 1e-6  # Avoid division by zero
    torso_height = abs(shoulder[1] - hip[1]) + 1e-6
    
    # ========== POSTURE DETECTION ==========
    
    # 1. SITTING DETECTION (improved)
    hip_shoulder_dist = hip[1] - shoulder[1]
    hip_knee_dist = knee[1] - hip[1] if np.any(knee != 0) else 0
    
    # Normalized ratios
    torso_ratio = hip_shoulder_dist / body_height
    leg_bend_ratio = hip_knee_dist / body_height if hip_knee_dist > 0 else 0
    
    # More robust sitting detection
    is_sitting = (
        torso_ratio > 0.4 and  # Torso is reasonably upright
        leg_bend_ratio < 0.5   # Legs are bent (knees close to hips)
    )
    
    # 2. HANDS ACTIVITY DETECTION
    # Check if hands are raised (above elbows or in working position)
    hands_raised = (
        wrist_mid[1] < elbow_mid[1] - 20 or  # Hands above elbows
        wrist_mid[1] < shoulder[1]            # Hands at shoulder level or higher
    )
    
    # 3. HANDS IN FRONT (working on something)
    # Calculate horizontal distance of hands from body center
    body_center_x = shoulder[0]
    hands_forward = abs(wrist_mid[0] - body_center_x) > 50  # Hands extended forward
    
    # 4. HEAD ORIENTATION (looking down at work)
    head_down = head[1] > shoulder[1] - 50  # Head is lowered
    
    # 5. ARMS BENT (actively working)
    L_arm_bent = False
    R_arm_bent = False
    
    if np.any(L_elb != 0) and np.any(L_wr != 0):
        L_forearm_len = np.linalg.norm(L_wr - L_elb)
        L_upper_arm_len = np.linalg.norm(L_elb - L_sh)
        if L_forearm_len > 0 and L_upper_arm_len > 0:
            # Check if arm is bent (not fully extended)
            L_arm_extension = np.linalg.norm(L_wr - L_sh) / (L_forearm_len + L_upper_arm_len)
            L_arm_bent = L_arm_extension < 0.9
    
    if np.any(R_elb != 0) and np.any(R_wr != 0):
        R_forearm_len = np.linalg.norm(R_wr - R_elb)
        R_upper_arm_len = np.linalg.norm(R_elb - R_sh)
        if R_forearm_len > 0 and R_upper_arm_len > 0:
            R_arm_extension = np.linalg.norm(R_wr - R_sh) / (R_forearm_len + R_upper_arm_len)
            R_arm_bent = R_arm_extension < 0.9
    
    arms_bent = L_arm_bent or R_arm_bent
    
    # 6. HANDS NEAR FACE/CHEST (detailed work)
    hands_near_center = (
        abs(wrist_mid[1] - shoulder[1]) < body_height * 0.5 and
        abs(wrist_mid[0] - shoulder[0]) < body_height * 0.4
    )
    
    # ========== WORKING CLASSIFICATION ==========
    
    working_score = 0
    
    if is_sitting:
        working_score += 2  # Sitting is a strong indicator of desk work
        
        # Additional working indicators
        if hands_raised:
            working_score += 2
        if hands_forward:
            working_score += 1
        if head_down:
            working_score += 1
        if arms_bent:
            working_score += 1
        if hands_near_center:
            working_score += 1
    else:
        # Standing - less likely to be working at desk, but check anyway
        if hands_raised and arms_bent:
            working_score += 2
        if hands_forward:
            working_score += 1
    
    # ========== IDLE DETECTION ==========
    
    # Hands down and relaxed
    hands_down = wrist_mid[1] > hip[1] - 30
    
    # Arms fully extended (not actively working)
    arms_relaxed = not arms_bent
    
    # Hands at sides
    hands_at_sides = abs(wrist_mid[0] - shoulder[0]) < 30
    
    idle_score = 0
    
    if hands_down:
        idle_score += 2
    if arms_relaxed:
        idle_score += 1
    if hands_at_sides:
        idle_score += 1
    if not is_sitting:
        idle_score += 1  # Standing idle
    
    # ========== FINAL DECISION ==========
    
    # Decision thresholds
    WORKING_THRESHOLD = 3
    IDLE_THRESHOLD = 3
    
    if working_score >= WORKING_THRESHOLD:
        return "WORKING"
    elif idle_score >= IDLE_THRESHOLD:
        return "IDLE"
    else:
        # Default to previous behavior or use uncertainty
        return "WORKING" if is_sitting else "IDLE"


# =====================================================
# TIME TRACKING FUNCTIONS
# =====================================================
def _seconds_to_hhmmss(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _update_employee_timers(employee_id, activity, frame_number, fps, working_activity='WORKING', idle_activity='IDLE'):
    """
    Update time tracking for an employee

    Args:
        employee_id: Employee ID
        activity: Current activity (WORKING/IDLE/UNCERTAIN)
        frame_number: Current frame number
        fps: Video FPS
        working_activity: Activity considered as "working" (default: WORKING)
        idle_activity: Activity considered as "idle" (default: IDLE)
    """
    global employee_timers

    current_timestamp = frame_number / fps

    if employee_id not in employee_timers:
        # Initialize timer for new employee
        employee_timers[employee_id] = {
            'working_seconds': 0.0,
            'idle_seconds': 0.0,
            'first_timestamp': current_timestamp,
            'last_timestamp': current_timestamp,
            'last_frame': frame_number,
            'current_activity': activity
        }
        logger.debug(f"Initialized timer for employee {employee_id}")
        return

    timer = employee_timers[employee_id]

    # Calculate time elapsed since last frame
    frames_elapsed = frame_number - timer['last_frame']
    time_delta = frames_elapsed / fps

    # Update appropriate timer based on previous activity
    prev_activity = timer['current_activity']
    if prev_activity == working_activity:
        timer['working_seconds'] += time_delta
        logger.debug(f"Employee {employee_id}: Added {time_delta:.2f}s to working_time (total: {timer['working_seconds']:.2f}s)")
    elif prev_activity == idle_activity:
        timer['idle_seconds'] += time_delta
        logger.debug(f"Employee {employee_id}: Added {time_delta:.2f}s to idle_time (total: {timer['idle_seconds']:.2f}s)")

    # Update timestamps and activity
    timer['last_timestamp'] = current_timestamp
    timer['last_frame'] = frame_number
    timer['current_activity'] = activity


def _get_employee_times(employee_id):
    """Get formatted times for an employee"""
    if employee_id not in employee_timers:
        return {
            'working_time': '00:00:00',
            'idle_time': '00:00:00',
            'first_recognized_timestamp': '00:00:00',
            'last_recognized_timestamp': '00:00:00'
        }

    timer = employee_timers[employee_id]
    return {
        'working_time': _seconds_to_hhmmss(timer['working_seconds']),
        'idle_time': _seconds_to_hhmmss(timer['idle_seconds']),
        'first_recognized_timestamp': _seconds_to_hhmmss(timer['first_timestamp']),
        'last_recognized_timestamp': _seconds_to_hhmmss(timer['last_timestamp'])
    }


# =====================================================
# PERSON TRACKING FUNCTIONS
# =====================================================
def _calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


def _match_person_to_tracked(bbox, tracked_persons, threshold=0.3):
    """Match a detected person to tracked persons using IoU"""
    best_iou = 0.0
    best_track_id = None

    for track_id, tracked in tracked_persons.items():
        tracked_bbox = tracked['bbox']
        iou = _calculate_iou(bbox, tracked_bbox)

        if iou > best_iou and iou >= threshold:
            best_iou = iou
            best_track_id = track_id

    if best_track_id:
        logger.debug(f"Matched to track_id {best_track_id} with IoU {best_iou:.3f}")

    return best_track_id


def _should_run_rekognition(frame_number, interval):
    """Determine if current frame should run AWS Rekognition"""
    global person_tracker

    last_recog = person_tracker['last_recognition_frame']

    # Always run on first frame
    if last_recog == 0:
        return True

    # Run if interval has passed
    if frame_number - last_recog >= interval:
        return True

    return False


# =====================================================
# AWS & DATABASE INITIALIZATION
# =====================================================
def _init_rekognition():
    """Initialize AWS Rekognition client"""
    global rekognition_client

    logger.debug("Initializing AWS Rekognition client...")

    if not REKOGNITION_ENABLED:
        logger.info("⚠️  Face recognition is DISABLED (REKOGNITION_ENABLED=0)")
        return

    if rekognition_client is not None:
        logger.debug("Rekognition client already initialized")
        return

    try:
        logger.info(f"Connecting to AWS Rekognition in region: {AWS_REGION}")
        rekognition_client = boto3.client('rekognition', region_name=AWS_REGION)

        try:
            response = rekognition_client.describe_collection(CollectionId=COLLECTION_ID)
            face_count = response.get('FaceCount', 0)
            logger.info(f"✅ AWS Rekognition connected successfully")
            logger.info(f"   Collection: {COLLECTION_ID}")
            logger.info(f"   Face Count: {face_count}")
            logger.info(f"   Recognition Interval: Every {REKOGNITION_INTERVAL_FRAMES} frames")
        except rekognition_client.exceptions.ResourceNotFoundException:
            logger.warning(f"⚠️  Collection '{COLLECTION_ID}' does not exist")
        except Exception as e:
            logger.warning(f"⚠️  Could not describe collection: {e}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Rekognition client: {e}")
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        rekognition_client = None


def _init_database():
    """Initialize PostgreSQL connection and cache employee data"""
    global db_connection, employee_cache

    logger.debug("Initializing database connection...")

    if not REKOGNITION_ENABLED:
        logger.info("⚠️  Database connection skipped (face recognition disabled)")
        return

    if db_connection is not None:
        logger.debug("Database already connected")
        return

    try:
        logger.info(f"Connecting to PostgreSQL database...")
        logger.debug(f"   Host: {DB_HOST}:{DB_PORT}")

        db_connection = psycopg2.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logger.info("✅ PostgreSQL connected successfully")

        logger.info("Loading employees into cache...")
        cursor = db_connection.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("SELECT * FROM public.face_recognization")
            employees = cursor.fetchall()

            cached_count = 0
            missing_face_id = 0

            for emp in employees:
                face_id = emp.get('face_id')
                if face_id:
                    employee_cache[face_id] = {
                        'id': emp.get('id'),
                        'name': emp.get('name'),
                        'department': emp.get('department'),
                        'email': emp.get('email')
                    }
                    cached_count += 1
                else:
                    missing_face_id += 1

            cursor.close()

            logger.info(f"✅ Employee cache loaded: {cached_count} employees")
            if missing_face_id > 0:
                logger.warning(f"   ⚠️  Employees without face_id: {missing_face_id}")

        except psycopg2.Error as e:
            logger.error(f"❌ Database query failed: {e}")
            cursor.close()
            raise

    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        db_connection = None


# =====================================================
# IMAGE ENCODING/DECODING
# =====================================================
def _b64_to_np(b64_str: str):
    buf = base64.b64decode(b64_str, validate=True)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _np_to_b64(img: np.ndarray, quality=80) -> str:
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError('JPEG encode failed')
    return base64.b64encode(buf.tobytes()).decode('ascii')


# =====================================================
# FACE RECOGNITION FUNCTIONS
# =====================================================
def _extract_face_region(img, bbox, expand_ratio=0.2):
    """Extract face region from person's bounding box"""
    try:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]

        # Take upper 30% of person bbox for face
        face_height = int((y2 - y1) * 0.3)
        face_y1 = y1
        face_y2 = y1 + face_height

        # Add horizontal padding
        face_width = x2 - x1
        padding = int(face_width * expand_ratio)
        face_x1 = max(0, x1 - padding)
        face_x2 = min(w, x2 + padding)

        face_y1 = max(0, face_y1)
        face_y2 = min(h, face_y2)

        if face_x2 <= face_x1 or face_y2 <= face_y1:
            return None

        face_crop = img[face_y1:face_y2, face_x1:face_x2]

        if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
            logger.debug(f"Face crop too small: {face_crop.shape}")
            return None

        return face_crop

    except Exception as e:
        logger.error(f"❌ Face extraction failed: {e}")
        return None


def _recognize_face(face_img):
    """Recognize face using AWS Rekognition"""
    if not REKOGNITION_ENABLED or rekognition_client is None:
        return None

    try:
        _, img_encoded = cv2.imencode('.jpg', face_img)
        img_bytes = img_encoded.tobytes()

        response = rekognition_client.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': img_bytes},
            MaxFaces=1,
            FaceMatchThreshold=FACE_MATCH_THRESHOLD
        )

        if response.get('FaceMatches') and len(response['FaceMatches']) > 0:
            match = response['FaceMatches'][0]
            face_id = match['Face']['FaceId']
            similarity = match['Similarity']

            logger.info(f"✅ Face matched: {face_id[:16]}... (Similarity: {similarity:.2f}%)")

            if face_id in employee_cache:
                employee = employee_cache[face_id].copy()
                employee['similarity'] = round(similarity, 2)
                employee['face_id'] = face_id
                logger.info(f"   Employee: {employee.get('name')}")
                return employee
            else:
                logger.warning(f"⚠️  Face ID not in cache: {face_id[:16]}...")
                return None
        else:
            logger.info(f"🔴 No face match (visitor)")
            return None

    except rekognition_client.exceptions.InvalidParameterException:
        logger.debug(f"⚠️  No face detected in image")
        return None
    except Exception as e:
        logger.error(f"❌ Face recognition error: {e}")
        return None


# =====================================================
# ANNOTATION
# =====================================================
def _annotate_frame_with_recognition(img, dets, working_activity='WORKING', idle_activity='IDLE'):
    """Annotate frame with bounding boxes and labels"""
    out = img.copy()
    h, w = out.shape[:2]

    # Colors (BGR)
    color_deep_blue = (139, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    color_white = (255, 255, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness_box = 2
    thickness_text = 1

    # Counts
    working_count = 0
    idle_count = 0
    uncertain_count = 0
    recognized_count = 0
    visitor_count = 0

    for d in dets:
        try:
            x1, y1, x2, y2 = map(int, d.get('bbox_xyxy', [0, 0, 0, 0]))
        except Exception:
            continue

        activity = d.get('activity') or 'UNCERTAIN'
        employee_info = d.get('employee_info')
        is_recognized = employee_info is not None

        # Update counts
        if activity == working_activity:
            working_count += 1
        elif activity == idle_activity:
            idle_count += 1
        else:
            uncertain_count += 1

        if is_recognized:
            recognized_count += 1
        else:
            visitor_count += 1

        # Box color based on activity
        if activity == 'WORKING':
            box_color = color_green
        elif activity == 'IDLE':
            box_color = color_red
        else:
            box_color = color_blue

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness_box)

        # Prepare text
        text_lines = []
        if is_recognized:
            name = employee_info.get('name', 'Unknown')
            dept = employee_info.get('department', '')
            employee_id = employee_info.get('id')

            # Get time information
            times = _get_employee_times(employee_id)
            working_time = times['working_time']
            idle_time = times['idle_time']

            text_lines = [
                f"{name}",
                f"{dept} - {activity}",
                f"W:{working_time} I:{idle_time}"
            ]
        else:
            text_lines = [
                "VISITOR",
                f"{activity}"
            ]

        # Draw text with background
        pad = 4
        tx = x1 + pad
        ty_start = y1 + 15

        for i, text in enumerate(text_lines):
            ty = ty_start + (i * 20)
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness_text)

            rect_x1 = max(tx - pad, 0)
            rect_y1 = max(ty - text_h - pad, 0)
            rect_x2 = min(tx + text_w + pad, w - 1)
            rect_y2 = min(ty + baseline + pad, h - 1)

            cv2.rectangle(out, (rect_x1, rect_y1), (rect_x2, rect_y2), color_white, -1)
            cv2.putText(out, text, (tx, ty), font, font_scale, color_deep_blue, thickness_text, lineType=cv2.LINE_AA)

    # Top-left overlay
    overlay_lines = [
        f'Working: {working_count}   Idle: {idle_count}   Uncertain: {uncertain_count}',
        f'Recognized: {recognized_count}   Visitors: {visitor_count}'
    ]

    pad = 8
    bx1, by1 = 5, 5
    max_width = max([cv2.getTextSize(line, font, 0.7, 2)[0][0] for line in overlay_lines])

    bx2 = bx1 + max_width + pad
    by2 = by1 + len(overlay_lines) * 25 + pad

    cv2.rectangle(out, (bx1, by1), (bx2, by2), color_white, -1)

    for i, line in enumerate(overlay_lines):
        tx = bx1 + pad // 2
        ty = by1 + 20 + (i * 25)
        cv2.putText(out, line, (tx, ty), font, 0.7, color_deep_blue, 2, lineType=cv2.LINE_AA)

    return out


# =====================================================
# MODEL LOADING
# =====================================================
def model_fn(model_dir):
    """Load YOLO model and initialize AWS/DB connections"""
    logger.info("=" * 60)
    logger.info("MODEL LOADING")
    logger.info("=" * 60)

    weights = os.path.join(model_dir, 'best.pt')
    alt = os.environ.get('WEIGHTS_PATH')

    if alt and os.path.exists(alt):
        weights = alt

    if not os.path.exists(weights):
        raise FileNotFoundError(f'Weights not found: {weights}')

    logger.info(f"Loading YOLO model from: {weights}")
    model = YOLO(weights)
    logger.info(f"✅ YOLO model loaded successfully")

    _init_rekognition()
    _init_database()

    logger.info("=" * 60)
    logger.info("MODEL READY")
    logger.info("=" * 60)

    return model


# =====================================================
# INPUT PROCESSING
# =====================================================
def input_fn(input_data, content_type):
    """Parse input JSON"""
    if content_type != 'application/json':
        raise ValueError('Only application/json is supported')

    if isinstance(input_data, (bytes, bytearray)):
        body = json.loads(input_data.decode('utf-8'))
    elif isinstance(input_data, str):
        body = json.loads(input_data)
    else:
        body = input_data

    for k in ('org_id', 'cam_id', 'encoding'):
        if k not in body:
            raise ValueError(f'Missing required field: {k}')

    img = _b64_to_np(body['encoding'])
    body['_img'] = img

    # Optional: Get FPS if provided
    global video_fps
    if 'fps' in body:
        video_fps = float(body['fps'])

    return body


# =====================================================
# PREDICTION (MAIN LOGIC WITH ACTIVITY TRACKING)
# =====================================================
def predict_fn(data, model):
    """Run YOLO detection + optimized face recognition + activity tracking"""
    global current_frame_number, person_tracker, video_fps

    img = data['_img']
    cam_id = int(data['cam_id'])
    org_id = int(data['org_id'])

    # Get FPS from data if available
    if 'fps' in data:
        video_fps = float(data['fps'])

    # Increment frame counter
    current_frame_number += 1

    # Parameters
    imgsz = int(os.environ.get('IMG_SIZE', '640'))
    conf = float(os.environ.get('CONF_THRESHOLD', '0.5'))
    iou = float(os.environ.get('IOU_THRESHOLD', '0.45'))
    ret_preview = os.environ.get('RETURN_PREVIEW', '1') != '0'

    WORKING_ACTIVITY = os.environ.get('WORKING_ACTIVITY', 'WORKING')
    IDLE_ACTIVITY = os.environ.get('IDLE_ACTIVITY', 'IDLE')

    # Determine if we should run rekognition
    run_rekognition = _should_run_rekognition(current_frame_number, REKOGNITION_INTERVAL_FRAMES)

    if run_rekognition:
        logger.info(f"🔍 Frame {current_frame_number}: Running AWS Rekognition")
        person_tracker['last_recognition_frame'] = current_frame_number

    # YOLO detection
    results = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    r = results[0]
    dets = []
    boxes = r.boxes
    kpts = r.keypoints

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            b = boxes[i]
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            confv = float(b.conf[0]) if b.conf is not None else 0.0

            bbox = [x1, y1, x2, y2]

            # Extract keypoints
            kp_list = []
            if kpts is not None and len(kpts) > i:
                try:
                    xy = kpts.xy[i].cpu().numpy()
                    confs = None
                    if hasattr(kpts, 'conf') and kpts.conf is not None:
                        confs = kpts.conf[i].cpu().numpy()
                    for j in range(xy.shape[0]):
                        x, y = float(xy[j, 0]), float(xy[j, 1])
                        c = float(confs[j]) if confs is not None else 1.0
                        kp_list.append([x, y, c])
                except Exception:
                    kp_list = []

            # Classify activity using the new function
            activity = classify_activity(kp_list) if kp_list else 'UNCERTAIN'

            # PERSON TRACKING & RECOGNITION
            employee_info = None
            track_id = None

            if REKOGNITION_ENABLED:
                if run_rekognition:
                    # Run face recognition
                    face_crop = _extract_face_region(img, bbox)
                    if face_crop is not None:
                        employee_info = _recognize_face(face_crop)

                        if employee_info:
                            # Create or update tracking entry
                            track_id = f"track_{i}_{current_frame_number}"
                            person_tracker['tracked_persons'][track_id] = {
                                'employee_info': employee_info,
                                'bbox': bbox,
                                'last_frame': current_frame_number
                            }
                else:
                    # Try to match with tracked persons
                    matched_track_id = _match_person_to_tracked(
                        bbox,
                        person_tracker['tracked_persons'],
                        TRACKING_IOU_THRESHOLD
                    )

                    if matched_track_id:
                        # Reuse cached employee info
                        tracked = person_tracker['tracked_persons'][matched_track_id]
                        employee_info = tracked['employee_info']
                        track_id = matched_track_id

                        # Update bbox and last seen frame
                        tracked['bbox'] = bbox
                        tracked['last_frame'] = current_frame_number

                        logger.debug(f"Reused cached recognition for {employee_info.get('name')}")

            # UPDATE TIME TRACKING
            if employee_info:
                employee_id = employee_info.get('id')
                _update_employee_timers(
                    employee_id,
                    activity,
                    current_frame_number,
                    video_fps,
                    WORKING_ACTIVITY,
                    IDLE_ACTIVITY
                )

            # Build detection object
            det = {
                'person_id': i + 1,
                'confidence': round(confv, 4),
                'activity': activity,
                'employee_info': employee_info,
                'bbox_xyxy': bbox,
                'keypoints': kp_list,
                'track_id': track_id
            }

            dets.append(det)

    # Generate annotated frame
    ann = None
    if ret_preview:
        try:
            annotated_img = _annotate_frame_with_recognition(
                img, dets,
                working_activity=WORKING_ACTIVITY,
                idle_activity=IDLE_ACTIVITY
            )
            ann = _np_to_b64(annotated_img, quality=80)
        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            ann = None

    # Prepare output with TIME FIELDS
    output_dets = []
    for d in dets:
        emp_info = d.get('employee_info')

        detection = {
            'person_id': d['person_id'],
            'confidence': d['confidence'],
            'activity': d['activity'],
        }

        # Add employee info if recognized
        if emp_info:
            employee_id = emp_info.get('id')

            detection['recognized'] = True
            detection['employee_id'] = employee_id
            detection['employee_name'] = emp_info.get('name')
            detection['department'] = emp_info.get('department')
            detection['similarity'] = emp_info.get('similarity')

            # Add time tracking fields
            times = _get_employee_times(employee_id)
            detection['working_time'] = times['working_time']
            detection['idle_time'] = times['idle_time']
            detection['first_recognized_timestamp'] = times['first_recognized_timestamp']
            detection['last_recognized_timestamp'] = times['last_recognized_timestamp']
        else:
            detection['recognized'] = False
            detection['status'] = 'visitor'

        output_dets.append(detection)

    return {
        'org_id': org_id,
        'cam_id': cam_id,
        'status': 'ok',
        'rekognition_enabled': REKOGNITION_ENABLED,
        'frame_number': current_frame_number,
        'rekognition_called': run_rekognition,
        'detections_count': len(output_dets),
        'detections': output_dets,
        'annotated_frame': ann
    }


# =====================================================
# OUTPUT FORMATTING
# =====================================================
def output_fn(prediction, accept):
    """Format output as JSON"""
    if accept and 'application/json' not in accept:
        raise ValueError('Only application/json Accept is supported')
    return json.dumps(prediction, separators=(',', ':'))