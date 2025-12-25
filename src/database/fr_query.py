
import os
import logging
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import json
import logging

load_dotenv()
logger = logging.getLogger("detection")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

try:
    pool = SimpleConnectionPool(
        minconn=1,        
        maxconn=15,       
        **DB_CONFIG
    )
    logger.info("✅ PostgreSQL Connection Pool Created")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")
    raise




logger = logging.getLogger("detection")


def insert_data(d,s3_url):
    """
    Insert activity detection data into activity_detections table
    STRICTLY follows the activity response structure
    """

    conn = None
    try:
        conn = pool.getconn()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO fr_detections (
                org_id,
                cam_id,
                frame_number,
                status,
                rekognition_enabled,
                rekognition_called,
                detections_count,
                detections,
                s3_url
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s,
                %s::jsonb,%s
            )
            RETURNING id;
        """

        cursor.execute(
            insert_query,
            (
                d["org_id"],                    # int
                d["cam_id"],                    # int
                d["frame_number"],              # int
                d["status"],                    # text

                d["rekognition_enabled"],       # boolean
                d["rekognition_called"],        # boolean

                d["detections_count"],          # int

                json.dumps(d["detections"])  ,   #  FULL detections[] stored as-is
                s3_url
            )
        )

        inserted_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            f"✅ Activity data inserted | cam_id={d['cam_id']} | frame={d['frame_number']} | id={inserted_id}"
        )
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Failed to insert activity data: {e}")
        return False

    finally:
        if conn:
            pool.putconn(conn)




