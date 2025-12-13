
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


def insert_data(d, s3_url):
    """Insert data into queue_monitoring table using connection pooling."""

    conn = None
    try:
        conn = pool.getconn()   # ⬅ Borrow connection from pool
        cursor = conn.cursor()

        insert_query = """
                INSERT INTO shop_lifting (
                    cam_id,
                    org_id,
                    user_id,
                    frame_id,
                    timestamp,
                    persons,
                    alerts,      -- now used for guns JSON
                    message,
                    s3_url,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                RETURNING id;
            """

        cursor.execute(
                insert_query,
                (
                    d["cam_id"],
                    d["org_id"],
                    d["user_id"],
                    d["frame_id"],
                    d["timestamp"],

                    # persons might not exist → default empty list
                    d.get("persons", []),

                    json.dumps(d["guns"]),   # store guns[] into JSONB column

                    d["message"],
                    s3_url,
                    d["status"]
                )
            )

        conn.commit()
        cursor.close()

        logger.info(f"✅ Data inserted for frame: {d['Frame_Id']}")
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Failed to insert data: {e}")
        return False

    finally:
        if conn:
            pool.putconn(conn)   # ⬅ Return connection back to pool



