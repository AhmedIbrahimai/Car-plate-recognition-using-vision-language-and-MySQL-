import cv2
import os
import base64
import threading
import mysql.connector  
from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
from shapely.geometry import LineString
from openai import OpenAI

# Set up OpenAI API Key (Use a secure method to store it)
OPENROUTER_API_KEY = ""  # Replace with your actual API key
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# MySQL Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "vehicle_data"
}

def initialize_database():
    """Creates the database and table if they do not exist."""
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()

        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS vehicle_data")
        cursor.close()
        conn.close()

        # Connect to MySQL Server with database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create table (without storing image path)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            track_id INT,
            speed INT,
            date_time DATETIME,
            vehicle_model VARCHAR(255),
            vehicle_color VARCHAR(100),
            vehicle_company VARCHAR(255),
            number_plate VARCHAR(50)
        )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database and table are ready!")

    except Exception as e:
        print(f"❌ Database Initialization Error: {e}")

# Call the function to initialize the database & table
initialize_database()

def insert_into_database(track_id, speed, timestamp, model, color, company, number_plate):
    """Insert detected vehicle details into MySQL database (without image path)."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
        INSERT INTO vehicle_records (track_id, speed, date_time, vehicle_model, vehicle_color, vehicle_company, number_plate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (track_id, speed, timestamp, model, color, company, number_plate)

        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Data inserted for Track ID: {track_id}")
    except Exception as e:
        print(f"❌ Database Insert Error: {e}")

def get_license_plate(track_id):
    """Fetch the license plate number for a given track_id from the database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = "SELECT number_plate FROM vehicle_records WHERE track_id = %s ORDER BY date_time DESC LIMIT 1"
        cursor.execute(query, (track_id,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result and result[0]:
            return result[0]
        return "Unknown"
    except Exception as e:
        print(f"❌ Error fetching license plate for Track ID {track_id}: {e}")
        return "Unknown"

class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.spd = {}  
        self.trkd_ids = []  
        self.trk_pt = {}  
        self.trk_pp = {}  
        self.saved_ids = set()  
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        os.makedirs("crop", exist_ok=True)

    def analyze_and_save_response(self, image_path, track_id, speed, timestamp):
        """Analyzes the image with OpenAI API and saves response to MySQL (without storing image path)."""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
                },
                extra_body={},
                model="google/gemini-2.0-pro-exp-02-05:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract ONLY these details:\n"
                             "| Vehicle Model | Color | Company | Number Plate |\n"
                             "|--------------|--------|---------|--------------|"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Detected vehicle"}
                        ]
                    }
                ]
            )

            response_text = completion.choices[0].message.content.strip()

            valid_rows = [
                row.split("|")[1:-1] for row in response_text.split("\n")
                if "|" in row and "Vehicle Model" not in row and "---" not in row
            ]

            vehicle_info = valid_rows[0] if valid_rows else ["Unknown", "Unknown", "Unknown", "Unknown"]

            insert_into_database(track_id, speed, timestamp, vehicle_info[0], vehicle_info[1], vehicle_info[2], vehicle_info[3])

        except Exception as e:
            print(f"❌ Error invoking OpenAI API: {e}")

    def estimate_speed(self, im0):
        """Estimate speed of detected vehicles in the video."""
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = time()
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = box  

            prev_pos = self.trk_pp[track_id]
            curr_pos = box

            if LineString([prev_pos[:2], curr_pos[:2]]).intersects(LineString(self.region)):
                direction = "known"
            else:
                direction = "unknown"

            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = np.linalg.norm(np.array(curr_pos[:2]) - np.array(prev_pos[:2])) / time_difference
                    self.spd[track_id] = round(speed)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = curr_pos

            speed_value = self.spd.get(track_id, 0)
            label = f"ID: {track_id} {speed_value} km/h"
            self.annotator.box_label(box, label=label, color=colors(track_id, True))

            # Fetch the license plate number for this track_id
            license_plate = get_license_plate(track_id)

            # Calculate the middle of the bounding box to place the license plate text
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Add the license plate text in the middle of the bbox
            text_size, _ = cv2.getTextSize(license_plate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2

            # Draw a background rectangle for better visibility
            cv2.rectangle(
                im0,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )

            # Put the license plate text
            cv2.putText(
                im0,
                license_plate,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (255, 255, 255),  # White text
                2  # Thickness
            )

            if track_id in self.spd and track_id not in self.saved_ids:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = im0[y1:y2, x1:x2]

                if cropped_image.size != 0:
                    image_filename = f"crop/{track_id}_{speed_value}kmh.jpg"
                    cv2.imwrite(image_filename, cropped_image)
                    print(f"📷 Saved image: {image_filename}")

                    threading.Thread(
                        target=self.analyze_and_save_response,
                        args=(image_filename, track_id, speed_value, current_time),
                        daemon=True
                    ).start()

                    self.saved_ids.add(track_id)

        self.display_output(im0)
        return im0

# Main execution block with video saving
cap = cv2.VideoCapture('tc.mp4')
region_points = [(0, 119), (1018, 119)]

speed_obj = SpeedEstimator(region=region_points, model="yolo12s.pt", line_width=2)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video file
output_video_file = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_file, fourcc, fps, (1020, 500))  # Initialize VideoWriter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    result = speed_obj.estimate_speed(frame)

    # Write the processed frame to the output video
    out.write(result)

    cv2.imshow("Speed Estimation", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
print(f"✅ Video saved as {output_video_file}")