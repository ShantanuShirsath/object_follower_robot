import cv2
from ultralytics import YOLO
import time
from display_frame import display_bounding_box
from controller import PID
import asyncio
from cvbot.communication.txtapiclient import TxtApiClient
from cvbot.controller.easy_drive_controller import EasyDriveController
from cvbot.config.drive_robot_configuration import DriveRobotConfiguration
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

HOST = "192.168.7.2" # Fur USB it is "192.168.7.2"
PORT = 80 # Port of the web server hosting the api, for the TXT its 80
KEY = "phH2o7" # API key for the TXT controller, can be found in the UI of the TXT controller

async def main():
	min_time = 0.4
	last_time = 0
	# Initialize API client and controller
	api_client = TxtApiClient(HOST, PORT, KEY)
	await api_client.initialize()

	controller = EasyDriveController(api_client, DriveRobotConfiguration())

	# Load YOLO model (use yolov5s.pt, yolov8n.pt, etc.)
	model = YOLO("/home/pi/Downloads/best_ncnn_model")  # You can change to yolov5s.pt if needed

	# Initialize video capture (0 = default USB camera, or use your Pi camera stream URL if needed)
	cap = cv2.VideoCapture(0)

	# Check if camera is opened
	if not cap.isOpened():
		print("Error: Could not open camera.")
		exit()
		
	ret, frame = cap.read()
	H, W = frame.shape[:2]
	img_cx,img_cy = W/2, H/2
	frame_area = H*W
	desired_area = 0.90* frame_area
	error_tolerance_lat = 10
	error_tolerence_long = 10000
	#pid for lateral centering
	pid_lat = PID(kp=0.6, kd= 0.1, ki=0, setpoint = W/2, output_limits = (-80,80))
	lat_error_history = []
	long_error_history = []
	time_history = []
	
	#pid for distance control
	pid_lon = PID(kp=5e-4, kd=0 ,ki=0, setpoint = desired_area, output_limits = (-80,80))
	init_time = time.time()
	while True:

		start_time = time.time()
		ret, frame = cap.read()
		if not ret:
			break

		#Run YOLO inference on the frame
		results = model(frame)
		
		#display the bounding box
		area, xc = display_bounding_box(results,img_cx,img_cy,H,W,start_time)
		print(f"area = {area}, center = {xc}")
		
		
		if (start_time - last_time) > min_time:
			current_time = time.time() - init_time  # relative time from start
			lat_error = pid_lat.error_call(xc)
			long_error = pid_lon.error_call(area)
			lat_error_history.append(lat_error)
			long_error_history.append(long_error)
			time_history.append(current_time)
			
			if((lat_error > error_tolerance_lat and long_error > error_tolerence_long)):
				vx = round(int(pid_lat.out()))
				vy = round(int(pid_lon.out()))
				await controller.drive(np.array([-vx,vy,0]))
				print(f"vx = {vx}, vy = {vy}")
				last_time = time.time()
			
			else:
				await asyncio.sleep(1.0)
				await controller.stop()
				last_time = time.time()
				
		# Press 'q' to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	# Convert to absolute errors
	lat_error_abs = [abs(e) for e in lat_error_history]
	long_error_abs = [abs(e) for e in long_error_history]
	# Create subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

	# --- Lateral Error Plot ---
	ax1.plot(time_history, lat_error_abs, label='Lateral Error', color='blue')
	ax1.fill_between(time_history, 0, error_tolerance_lat, color='blue', alpha=0.1, label='Tolerance Band')
	ax1.set_ylabel('Absolute Error')
	ax1.set_title('Lateral Error Convergence')
	ax1.legend()
	ax1.grid(True)

	# --- Longitudinal Error Plot ---
	ax2.plot(time_history, long_error_abs, label='Longitudinal Error', color='orange')
	ax2.fill_between(time_history, 0, error_tolerence_long, color='orange', alpha=0.1, label='Tolerance Band')
	ax2.set_xlabel('Time (s)')
	ax2.set_ylabel('Absolute Error')
	ax2.set_title('Longitudinal Error Convergence')
	ax2.legend()
	ax2.grid(True)

	# Release resources
	await asyncio.sleep(1.0)
	await controller.stop()
	cap.release()
	cv2.destroyAllWindows()
	
	# Layout
	plt.tight_layout()
	plt.show()
	

	
	
# Entry point
if __name__ == "__main__":
    asyncio.run(main())
