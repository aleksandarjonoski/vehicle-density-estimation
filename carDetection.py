import cv2
import time
from datetime import datetime

# Create full body classifier
classify_car = cv2.CascadeClassifier("haarcascade_car.xml")

# Capture the video
# TO DO, connect the camera here
vid_capture = cv2.VideoCapture("road_cars.mp4")

while vid_capture.isOpened():

    # Read the frame
    ret, frame = vid_capture.read()

    # Convert the image to greyscale 
    greyscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass the frame to the full body classifier
    cars_detected = classify_car.detectMultiScale(greyscale_image, 1.4, 2)


    # Density detection logic
    # TO DO show red green blue collor respectivly on
    # the bonnet vision kit or raw rasberry pi 
    if(len(cars_detected ) <= 1):
        print("LOW DENSITY " + str(datetime.today()))
    if(len(cars_detected) == 2):
        print("MEDIUM DENSITY" + str(datetime.today()))
    if(len(cars_detected) > 2):
        print("HIGH DENSITY" + str(datetime.today()))
    
    # Draw bounding boxes around finded objects
    for(x,y,w,h) in cars_detected:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow("Here are the cars", frame)

    # Exit the window with q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid_capture.release()
cv2.destroyAllWindows()                