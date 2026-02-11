"""
Simple example with automatic database updates
Loads workstations from DB and updates metrics every 10 seconds
"""

from workstation_inference import WorkstationInference, DatabaseConfig
import cv2

def main():
    # =====================
    # CONFIGURATION
    # =====================
    VIDEO_PATH = "assets/day_2.avi"  # or 0 for webcam
    MODEL_PATH = "yolov8s.pt"
    
    # Database configuration
    DB_CONFIG = DatabaseConfig(
        host="localhost",
        port=5432,
        dbname="test",
        user="postgres",
        password="admin123"
    )
    
    # Organization and Camera IDs
    ORG_ID = 1
    CAM_ID = 1
    
    # =====================
    # SETUP
    # =====================
    print("🔧 Initializing inference system...")
    print("   - Loading workstations from database")
    print("   - Auto-update enabled (every 10 seconds)")
    print()
    
    # Create inference instance
    # Workstations are automatically loaded from database
    # Database updates happen automatically every 10 seconds
    inference = WorkstationInference(
        model_path=MODEL_PATH,
        confidence_threshold=0.4,
        missing_threshold=3.0,
        db_config=DB_CONFIG,
        org_id=ORG_ID,
        cam_id=CAM_ID,
        db_update_interval=10.0,  # Update DB every 10 seconds
        auto_update_db=True        # Enable automatic updates
    )
    
    # Check if workstations were loaded
    if not inference.workstations:
        print("❌ No workstations found in database")
        print("   Please add workstations to the 'workstations' table:")
        print("   INSERT INTO workstations (org_id, cam_id, name, x1, y1, x2, y2)")
        print("   VALUES (1, 1, 'Desk-A', 100, 100, 400, 500);")
        return
    
    # Open video source
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video source")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS")
    print(f"🎯 Monitoring {len(inference.workstations)} workstations:")
    for name in inference.workstations.keys():
        print(f"   • {name}")
    print()
    print("🚀 Starting inference...")
    print("   Press ESC to quit")
    print("   Press M for metrics")
    print()
    
    # =====================
    # MAIN LOOP
    # =====================
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video finished")
                break
            
            # Process frame - database updates happen automatically!
            annotated_frame = inference.process_frame(frame)
            
            # Display
            cv2.imshow("Workstation Monitoring", annotated_frame)
            
            frame_count += 1
            
            # Status update every 100 frames
            if frame_count % 100 == 0:
                active = sum(1 for ws in inference.workstations.values() if ws.status == "ACTIVE")
                print(f"📊 Frame {frame_count}: {active}/{len(inference.workstations)} active")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹️  Stopped by user")
                break
            elif key == ord('m') or key == ord('M'):
                # Show detailed metrics
                inference.print_metrics()
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # =====================
        # CLEANUP & FINAL REPORT
        # =====================
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        
        # Display final metrics
        inference.print_metrics()
        
        print(f"\nTotal frames processed: {frame_count}")
        print("="*80 + "\n")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()