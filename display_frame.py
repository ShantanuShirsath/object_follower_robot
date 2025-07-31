import cv2
import time

def display_bounding_box(results:object, img_cx,img_cy,H,W,start_time):
	# Plot results on the frame
	res = results[0]
	annotated_frame = res.plot()
	xc=0
	yc=0
	
	#extract centers
	if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
		confs = res.boxes.conf.cpu().numpy()  # shape: (num_boxes,)
		max_idx = confs.argmax()  # index of highest confidence

		# Get the best box
		best_box = res.boxes.xywh[max_idx].cpu().numpy()
		centers = (best_box[0],best_box[1])
		area = best_box[2]*best_box[3]
	else:
		centers = []
		area = 0
		
	
	#draw cross at image center
	cv2.drawMarker(annotated_frame,
					(int(img_cx),int(img_cy)),
					color=(255,0,0),
					markerType=cv2.MARKER_CROSS,
					markerSize = 20,
					thickness=2)
					
	if centers:
		xc,yc = centers
		ix,iy = int(xc),int(yc)
		#draw cross at detection center
		cv2.drawMarker(annotated_frame,
					(int(ix),int(iy)),
					color=(255,0,0),
					markerType=cv2.MARKER_CROSS,
					markerSize = 15,
					thickness=2)
					
		dx = xc - img_cx
		dy = yc - img_cy
		
		#print centers in obunding box
		cv2.putText(annotated_frame,
					f"dx={dx:.1f},dy={dy:.1f}",
					(ix+10,iy+10),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,(0,255,0),2)
					
		#print area in bounding box
		cv2.putText(annotated_frame,
            f"area={area:.1f}",
            (int(xc), int(yc) - 10),  # a little above the top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2)
					
	# Display the resulting frame
	fps = 1/(time.time() - start_time)

	cv2.putText(annotated_frame,f"FPS: {fps: .2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) ,2)
	cv2.imshow('YOLO Live Detection', annotated_frame)
	
	return area, xc
