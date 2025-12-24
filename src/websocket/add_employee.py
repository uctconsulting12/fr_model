import os
import sys
import boto3
import psycopg2
import datetime
import json
import logging

# =====================================================
# CONFIGURATION
# =====================================================
# AWS settings: Defines where we send the images for analysis
AWS_REGION = 'us-east-1'
COLLECTION_ID = 'EmployeeCollection'  # The 'Folder' in AWS Rekognition where faces are stored

# Database Credentials: Connectivity for PostgreSQL
DB_HOST = '54.225.63.242'
DB_PORT = '5432'
DB_NAME = 'visco'
DB_USER = 'visco_cctv'
DB_PASSWORD = 'Visco@0408'

# S3 Configuration: Where the physical image files are saved
S3_BUCKET = 'employ-recog'
S3_FOLDER = 'employee_data'

# Threshold for duplicate checks. 
# If a new photo matches an old photo by 85% or more, we consider it a duplicate.
SIMILARITY_THRESHOLD = 85.0

# Configure logging to output errors/info to the console
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='LOG: %(message)s')

def get_employee_face_ids(emp_id):
    """
    Helper Function:
    Goes to the database and gets a list of all 'face_ids' (fingerprints) 
    we have already saved for this specific employee.
    Used later to check if the new photo matches any of these.
    """
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cursor = conn.cursor()
        
        # Selects only the face_ids belonging to this specific Employee ID
        cursor.execute("SELECT face_id FROM public.face_recognization WHERE id = %s", (emp_id,))
        
        # Convert the list of tuples [(id1,), (id2,)] into a clean list [id1, id2]
        face_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        return face_ids
    except Exception as e:
        logging.error(f"Error fetching employee face IDs: {e}")
        return []

def check_duplicate_for_employee(rekognition, image_bytes, emp_id):
    """
    Logic Check:
    Before we register a new face, we ask AWS: "Does this person look like 
    anyone we already know?"
    
    If AWS says "Yes", we check if that match belongs to THIS employee.
    If it does, we reject the upload to prevent duplicate/redundant data.
    """
    try:
        # 1. Get the list of faces this employee already has
        employee_face_ids = get_employee_face_ids(emp_id)
        if not employee_face_ids:
            return False, None  # This is their first photo, so it can't be a duplicate.
        
        # 2. Send the image to AWS to find matches in the whole collection
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': image_bytes},
            MaxFaces=10, 
            FaceMatchThreshold=SIMILARITY_THRESHOLD
        )
        
        # 3. Check if any found matches are in this employee's existing list
        for match in response.get('FaceMatches', []):
            if match['Face']['FaceId'] in employee_face_ids:
                # Found a match! Return True (It is a duplicate)
                return True, {
                    'face_id': match['Face']['FaceId'],
                    'similarity': round(match['Similarity'], 2),
                    'employee_id': emp_id
                }
        return False, None
    except Exception:
        return False, None

def register_face(emp_id, image_path, name, department, org_id, user_id, email=""):
    """
    Main Workflow Function:
    1. Validates image.
    2. Checks for duplicates.
    3. Uploads to S3.
    4. Indexes face in AWS (with safety checks).
    5. Saves metadata to Database.
    """
    
    # Validation: Ensure file exists locally
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image file not found: {image_path}"}

    try:
        logging.info(f"Processing ID: {emp_id}, Name: {name}")

        # Step 0: Read the image file into memory (bytes)
        with open(image_path, 'rb') as image:
            img_bytes = image.read()

        # Initialize AWS connections
        rekognition = boto3.client('rekognition', region_name=AWS_REGION)
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Step 1: Duplicate Check (The Safety Net)
        # We check BEFORE uploading to S3 to save storage costs and keep DB clean.
        is_dup, match_data = check_duplicate_for_employee(rekognition, img_bytes, emp_id)
        if is_dup:
            return {"status": "error", "code": 409, "message": "Duplicate image detected.", "duplicate_info": match_data}

        # Step 2: Upload to S3 (Private Storage)
        # Generate a unique filename using timestamp to avoid overwriting files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _, ext = os.path.splitext(image_path)
        s3_key = f"{S3_FOLDER}/{emp_id}_{timestamp}{ext}".replace('\\', '/')
        
        # Upload the file. Note: No ACL is set, so it remains Private by default.
        s3_client.upload_file(image_path, S3_BUCKET, s3_key)
        
        # Construct the URL (used later for referencing the file)
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

        # Step 3: Index Face in AWS Rekognition
        # 'ExternalImageId' acts as a label for the face inside AWS
        external_image_id = f"{name.replace(' ', '_')}_{emp_id}_{timestamp}"
        
        # CRITICAL: We use MaxFaces=2.
        # This allows us to detect if there is MORE than 1 person in the photo.
        aws_response = rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'S3Object': {'Bucket': S3_BUCKET, 'Name': s3_key}},
            ExternalImageId=external_image_id,
            MaxFaces=2, 
            QualityFilter="AUTO", 
            DetectionAttributes=['ALL']
        )

        faces = aws_response['FaceRecords']
        count = len(faces)

        # Safety Check A: No face found
        if count == 0:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key) # Clean up S3
            return {"status": "error", "message": "No face detected in image."}

        # Safety Check B: Too many faces (Group photo rejection)
        if count > 1:
            # We must delete the faces AWS just added, otherwise we pollute the collection
            rekognition.delete_faces(CollectionId=COLLECTION_ID, FaceIds=[f['Face']['FaceId'] for f in faces])
            s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key) # Clean up S3
            return {"status": "error", "code": 400, "message": f"Image rejected! Contains {count} faces."}

        # If we get here, exactly 1 face was found. Success!
        face_id = faces[0]['Face']['FaceId']
        confidence = faces[0]['Face']['Confidence']

        # Step 4: Save to Database
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cursor = conn.cursor()

        # We insert the new record. 'cam_id' is omitted, so it defaults to NULL.
        insert_query = """
            INSERT INTO public.face_recognization 
            (id, face_id, name, department, email, image_path, org_id, user_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING face_id;
        """
        
        cursor.execute(insert_query, (
            emp_id, face_id, name, department, email, s3_url, org_id, user_id, datetime.datetime.now()
        ))
        
        conn.commit() # Save changes
        cursor.close()
        conn.close()

        return {
            "status": "success",
            "code": 200,
            "message": "Face registered.",
            "data": {"face_id": face_id, "s3_path": s3_url, "confidence": round(confidence, 2)}
        }

    except psycopg2.Error as e:
        return {"status": "error", "code": 500, "message": f"Database Error: {e}"}
    except Exception as e:
        return {"status": "error", "code": 500, "message": str(e)}

def main():
    # Input Handling: Check if arguments are provided
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No input provided."}))
        sys.exit(1)

    try:
        # Determine if input is a File Path or a raw JSON string
        if os.path.isfile(sys.argv[1]):
            with open(sys.argv[1], 'r') as f: data = json.load(f)
        else:
            data = json.loads(sys.argv[1])
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}))
        sys.exit(1)

    # Validation: Ensure all necessary fields are present
    required = ['id', 'image', 'name', 'department', 'org_id', 'user_id']
    if not all(field in data for field in required):
        print(json.dumps({"status": "error", "message": f"Missing fields. Needed: {required}"}))
        sys.exit(1)

    # Run the registration logic
    result = register_face(
        data['id'], data['image'], data['name'], data['department'],
        data['org_id'], data['user_id'], data.get('email', "")
    )

    # Print result as JSON (Standard Output)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()