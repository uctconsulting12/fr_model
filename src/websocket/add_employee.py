import os
import sys
import boto3
import psycopg2
import json
import datetime

# =====================================================
# CONFIGURATION
# =====================================================
# AWS Configuration
AWS_REGION = 'us-east-1'
COLLECTION_ID = 'EmployeeCollection'

# Database Credentials
DB_HOST = '54.225.63.242'
DB_PORT = '5432'
DB_NAME = 'visco'
DB_USER = 'visco_cctv'
DB_PASSWORD = 'Visco@0408'

# S3 Configuration
S3_BUCKET = 'employ-recog'     # Your actual bucket name
S3_FOLDER = 'employee_data'    # The specific folder inside the bucket

def register_face(employee_data):
    """
    Registers employee face with the following steps:
    1. Uploads image to S3 (employee_data/ID.jpg).
    2. Indexes face in Rekognition using S3 object.
    3. Saves employee data with S3 URL to Database.
    
    Parameters:
    employee_data (dict): Dictionary containing:
        - emp_id (int): Employee ID
        - image_path (str): Local path to image file
        - name (str): Full name
        - department (str): Department name
        - org_id (int): Organization ID
        - user_id (int): User ID
        - email (str, optional): Email address
    """
    
    # Extract parameters from JSON input
    emp_id = employee_data.get('emp_id')
    image_path = employee_data.get('image_path')
    name = employee_data.get('name')
    department = employee_data.get('department')
    org_id = employee_data.get('org_id')
    user_id = employee_data.get('user_id')
    email = employee_data.get('email', '')
    
    # Validate required fields
    required_fields = ['emp_id', 'image_path', 'name', 'department', 'org_id', 'user_id']
    missing_fields = [field for field in required_fields if not employee_data.get(field)]
    
    if missing_fields:
        print(f"❌ Error: Missing required fields: {', '.join(missing_fields)}")
        return False
    
    # Validate Local Image Path
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at: {image_path}")
        return False

    print("="*60)
    print(f"👤 REGISTERING: {name} (ID: {emp_id})")
    print(f"   Organization ID: {org_id}")
    print(f"   User ID: {user_id}")
    print(f"   Department: {department}")
    print("="*60)

    # ---------------------------------------------------------
    # STEP 1: Upload to S3 FIRST
    # ---------------------------------------------------------
    print("1️⃣  Uploading image to S3...")
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Get extension (e.g., .jpg, .png) from original file
        _, file_extension = os.path.splitext(image_path)
        
        # Construct S3 key: "employee_data/42.jpg"
        s3_key = f"{S3_FOLDER}/{emp_id}{file_extension}"
        
        print(f"   Uploading to S3: {s3_key}...")
        # Upload with public-read ACL for public access
        s3_client.upload_file(
            image_path, 
            S3_BUCKET, 
            s3_key,
            ExtraArgs={'ACL': 'public-read'}
        )
        
        # Construct the S3 URL (object URL format)
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        print(f"   ✅ Upload Success (Public Access): {s3_url}")

    except Exception as e:
        print(f"❌ S3 Upload Failed: {e}")
        return False

    # ---------------------------------------------------------
    # STEP 2: Index Face in Rekognition using S3 object
    # ---------------------------------------------------------
    print("\n2️⃣  Indexing face in Rekognition...")
    try:
        rekognition = boto3.client('rekognition', region_name=AWS_REGION)

        # Use S3 object instead of bytes
        aws_response = rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={
                'S3Object': {
                    'Bucket': S3_BUCKET,
                    'Name': s3_key
                }
            },
            ExternalImageId=name.replace(" ", "_"),
            MaxFaces=1,
            QualityFilter="AUTO",
            DetectionAttributes=['ALL']
        )

        if not aws_response['FaceRecords']:
            print("❌ AWS Error: No face detected in the image.")
            return False

        face_record = aws_response['FaceRecords'][0]
        face_id = face_record['Face']['FaceId']
        confidence = face_record['Face']['Confidence']
        print(f"   ✅ Face Indexed! FaceId: {face_id} (Conf: {confidence:.2f}%)")

    except Exception as e:
        print(f"❌ Rekognition Indexing Failed: {e}")
        return False

    # ---------------------------------------------------------
    # STEP 3: Save to Database with org_id and user_id
    # ---------------------------------------------------------
    print("\n3️⃣  Saving to Database...")
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO public.face_recognization 
            (id, face_id, name, department, email, image_path, org_id, user_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        now = datetime.datetime.now()

        # Execute Query with all parameters including org_id and user_id
        cursor.execute(insert_query, (
            emp_id, 
            face_id, 
            name, 
            department, 
            email, 
            s3_url,
            org_id,
            user_id,
            now
        ))
        
        db_id = cursor.fetchone()[0]
        conn.commit()
        
        cursor.close()
        print(f"✅ Database Success! Employee Record ID: {db_id} saved.")
        print("\n🎉 REGISTRATION COMPLETE")
        return True

    except Exception as e:
        print(f"❌ Database Failed: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def main():
    """
    Accepts JSON input from stdin or command line argument.
    
    Example JSON:
    {
        "emp_id": 42,
        "image_path": "/path/to/image.jpg",
        "name": "John Doe",
        "department": "Engineering",
        "org_id": 1,
        "user_id": 100,
        "email": "john@example.com"
    }
    
    Usage:
    1. From stdin:   echo '{"emp_id":42,...}' | python script.py
    2. From file:    python script.py input.json
    3. Direct JSON:  python script.py '{"emp_id":42,...}'
    """
    
    try:
        # Check if argument is provided
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            
            # Check if it's a file path
            if os.path.isfile(arg):
                print(f"📄 Reading from file: {arg}")
                with open(arg, 'r') as f:
                    employee_data = json.load(f)
            else:
                # Assume it's a JSON string
                print(f"📝 Parsing JSON from argument")
                employee_data = json.loads(arg)
        else:
            # Read from stdin
            print(f"📥 Reading JSON from stdin...")
            json_input = sys.stdin.read()
            employee_data = json.loads(json_input)
        
        # Register the face
        success = register_face(employee_data)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        print("\nExpected format:")
        print(json.dumps({
            "emp_id": 42,
            "image_path": "/path/to/image.jpg",
            "name": "John Doe",
            "department": "Engineering",
            "org_id": 1,
            "user_id": 100,
            "email": "john@example.com"
        }, indent=2))
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()