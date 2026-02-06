import psycopg2

# ==========================
# DB CONNECTION
# ==========================
def get_db_conn():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="test",
        user="postgres",
        password="admin123"
    )

# ==========================
# CREATE TABLES (FRESH DB)
# ==========================
def create_tables():
    conn = get_db_conn()
    cur = conn.cursor()

    print("🛠️  Initializing Fresh Database Tables...")

    # --------------------------
    # 1. WORKSTATIONS (ROIs)
    # --------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS workstations (
            workstation_id SERIAL PRIMARY KEY,
            org_id INT NOT NULL,
            cam_id INT NOT NULL,
            name TEXT NOT NULL,
            x1 INT NOT NULL,
            y1 INT NOT NULL,
            x2 INT NOT NULL,
            y2 INT NOT NULL,
            created_at TIMESTAMP DEFAULT now(),
            
            -- Prevent duplicate workstation names for the same camera
            CONSTRAINT unique_workstation_per_cam UNIQUE (org_id, cam_id, name)
        );
    """)

    # --------------------------
    # 2. DAILY ANALYTICS
    # --------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS workstation_daily_analytics (
            id SERIAL PRIMARY KEY,

            org_id INT NOT NULL,
            cam_id INT NOT NULL,
            workstation_name TEXT NOT NULL,
            analytics_date DATE NOT NULL,

            -- Time tracking
            active_seconds FLOAT DEFAULT 0,
            vacant_seconds FLOAT DEFAULT 0,
            utilization_percent FLOAT DEFAULT 0,

            -- Missing Person tracking
            missing_count INT DEFAULT 0,
            missing_duration FLOAT DEFAULT 0,

            -- Timestamps
            first_seen_time TIME,
            last_present_time TIME,

            updated_at TIMESTAMP DEFAULT now(),

            -- CRITICAL: Ensures one row per workstation per day
            CONSTRAINT unique_daily_entry UNIQUE (org_id, cam_id, workstation_name, analytics_date)
        );
    """)

    # --------------------------
    # 3. VACANCY LOGS (Timeline of events)
    # --------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS workstation_vacancy_logs (
            id SERIAL PRIMARY KEY,
            org_id INTEGER,
            cam_id INTEGER,
            workstation_name VARCHAR(50),
            analytics_date DATE,
            
            start_time TIME,
            end_time TIME,
            duration_seconds REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All tables created successfully.")

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    create_tables()