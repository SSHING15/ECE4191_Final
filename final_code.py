import cv2
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
import numpy as np
from picamera2 import Picamera2
#import subprocess
#from ultralytics import YOLO
#CODE WAS WRITTEN BY CHATGPT WITH SPECIFIC PROMPTS

#______________________________________________FU0.3: Tennis Ball Detection____________________________________________________

    # return filename
# def capture_image():
#     subprocess.run(["sudo", "python3", "camera_access_script.py"])
#     return

def capture_image(filename='captures/image.jpg'):
    os.makedirs('captures', exist_ok=True)
    os.environ['LIBCAMERA_LOG_LEVELS'] = '*:ERROR'

    # Initialize the camera
    picam2 = Picamera2()
    
    try:
        
        picam2.start()
        time.sleep(1.5)

        # Capture the image as a NumPy array
        image = picam2.capture_array()
       
        # COLOUR CORRECT TEST
        # Convert to a different color space and adjust colors
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(frame)

        # # Adjust the 'a' and 'b' channels to correct the color
        # a = cv2.add(a, -55)  # Adjust as needed
        # b = cv2.add(b, 70)  # Adjust as needed

        # corrected_frame = cv2.merge([l, a, b])
        # corrected_frame = cv2.cvtColor(corrected_frame, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )

        cv2.imwrite(filename, image)
    
    except Exception as e:
        print(f"An error occurred while capturing the image: {e}")
    
    finally:
        picam2.stop()
        picam2.close()
    return filename

# def detect_largest_ball(image_path="captures/image.jpg", model_path='best_LACH.pt', output_path='captures/largest_annotated.jpg'):
#     # Load the trained YOLOv8 model
#     model = YOLO(model_path)
    
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Check if the image was loaded successfully
#     if image is None:
#         raise ValueError("Error: Image could not be loaded. Check the image path.")
    
#     # Run YOLOv8 predictions on the image
#     results = model(image, conf=0.5)  # Adjust confidence threshold if needed
    
#     # Variables to store the largest tennis ball's data
#     largest_area = 0
#     largest_ball_center_x = None
#     largest_ball_center_y = None
#     largest_ball_width = None
    
#     # Annotate the image with predictions
#     annotated_image = image.copy()
    
#     for result in results:
#         # Loop through detections
#         for detection in result.boxes:
#             # Assuming '1' is the class ID for the tennis ball
#             if detection.cls == 1 and detection.conf > 0.5:
#                 # Get the bounding box coordinates (x_min, y_min, x_max, y_max)
#                 x_min, y_min, x_max, y_max = detection.xyxy[0].cpu().numpy()
                
#                 # Calculate the bounding box area
#                 area = (x_max - x_min) * (y_max - y_min)
                
#                 # If this is the largest area so far, update the largest ball data
#                 if area > largest_area:
#                     largest_area = area
#                     largest_ball_center_x = int((x_min + x_max) / 2)
#                     largest_ball_center_y = int((y_min + y_max) / 2)
#                     largest_ball_width = int(x_max - x_min)
                    
#                     # Draw bounding box and center cross on the image
#                     cv2.rectangle(annotated_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
#                     cv2.drawMarker(annotated_image, (largest_ball_center_x, largest_ball_center_y), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
#     # Save the annotated image
#     cv2.imwrite(output_path, annotated_image)
    
#     # Return the x, y coordinates and width of the largest ball
#     if largest_ball_center_x is not None and largest_ball_center_y is not None:
#         return largest_ball_center_x, largest_ball_center_y, largest_ball_width
#     else:
#         return None, None, None

# Function to get prediction from Roboflow. if no object detected, will retun None
def get_prediction(image_path, api_key, model_endpoint):
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            model_endpoint,
            files={'file': image_file},
            headers={'Authorization': f'Bearer {api_key}'}
        )
    response.raise_for_status()
    prediction = response.json()
    #print(prediction)
    # # Check if there are any predictions
    if not prediction['predictions']:
        print("No objects detected in the image.")
        return None
    
    return prediction

def save_original_prediction_image(image_path, prediction, output_path='captures/original_output.jpg'):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for pred in prediction['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(x - width / 2, y - height / 2 - 10, pred['class'], color='red')

    plt.savefig(output_path)
    plt.close()
    print(f"Original prediction image saved to {output_path}")

# Function to display and save the annotated prediction image
def save_annotated_image(image_path, prediction, output_path='captures/annotated_output.jpg'):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for pred in prediction['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Plot a red 'X' in the middle of the bounding box
        ax.plot(x, y, 'rx')

        # Plot a blue horizontal line through the red 'X'
        ax.plot([x - width / 2, x + width / 2], [y, y], 'b-')

        # Print the coordinates of the red 'X' and the length of the blue line
        print(f"Red X Coordinates: (x={x}, y={y})")
        print(f"Blue Line Length: {width}")

        # Annotate the class label
        plt.text(x - width / 2, y - height / 2 - 10, pred['class'], color='red')

    plt.savefig(output_path)
    plt.close()
    print(f"Annotated image saved to {output_path}")

#___________________________________________FU1.2 Closest Ball Calculator & FU0.4: Object Location Calculator_____________________________________
# Function to display and save the annotated image with the largest tennis ball
def annotate_largest_tennis_ball(prediction,image_path="captures/image.jpg", output_path='captures/largest_ball_output.jpg'):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Find the largest tennis ball
    largest_pred = max(prediction['predictions'], key=lambda pred: pred['width'])

    x, y, width, height = largest_pred['x'], largest_pred['y'], largest_pred['width'], largest_pred['height']
    rect = patches.Rectangle(
        (x - width / 2, y - height / 2), width, height,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

    # Plot a red 'X' in the middle of the bounding box
    ax.plot(x, y, 'rx')

    # Plot a blue horizontal line through the red 'X'
    ax.plot([x - width / 2, x + width / 2], [y, y], 'b-')

    # Print the coordinates of the red 'X' and the length of the blue line
    print(f"Largest Ball Red X Coordinates: (x={x}, y={y})")
    print(f"Largest Ball Blue Line Length: {width}")

    # Annotate the class label
    plt.text(x - width / 2, y - height / 2 - 10, largest_pred['class'], color='red')

    plt.savefig(output_path)
    plt.close()
    #print(f"Annotated image of the largest tennis ball saved to {output_path}")

    #Return ball size as largest out of ball hieght or width (allow half tennis ball images to be correct)
    if width>=height:
        ball_size=width
    else:
        ball_size=height

    return x,y,ball_size

def find_tennis_balls():
    API_KEY = 'hDTlMgYwl8tO6DkW0SiI'  # Replace with your Roboflow API key
    #MODEL_ENDPOINT = 'YOUR_ROBOFLOW_MODEL_ENDPOINT'  # Replace with your Roboflow model endpoint
    #MODEL_ENDPOINT = 'https://detect.roboflow.com/tennisball-dzhrf/1?api_key=hDTlMgYwl8tO6DkW0SiI'
    MODEL_ENDPOINT = 'https://detect.roboflow.com/tennis-ball-model-izpej/1?api_key=hDTlMgYwl8tO6DkW0SiI'

    #start_time = time.time()
    image_path = capture_image()
    #time_after_block_1 = time.time()
    #elapsed_time_block_1 = time_after_block_1 - start_time
    #print(f"Time taken for Image: {elapsed_time_block_1} seconds")
    prediction = get_prediction(image_path, API_KEY, MODEL_ENDPOINT)
    #time_after_block_2 = time.time()
    #elapsed_time_block_2 = time_after_block_2 - time_after_block_1
    #print(f"Time taken for Prediction: {elapsed_time_block_2} seconds")
    return prediction, image_path
def find_tennis_ball_local():
    #image_path = capture_image()
    capture_image()
    x,y,width=detect_largest_ball()
    return x,y,width

def find_deposit_box():
    API_KEY = 'hDTlMgYwl8tO6DkW0SiI'  # Replace with your Roboflow API key
    #MODEL_ENDPOINT = 'YOUR_ROBOFLOW_MODEL_ENDPOINT'  # Replace with your Roboflow model endpoint
    MODEL_ENDPOINT = 'https://detect.roboflow.com/cardboardbox-detection/1?api_key=hDTlMgYwl8tO6DkW0SiI'
    image_path = capture_image()
    prediction = get_prediction(image_path, API_KEY, MODEL_ENDPOINT)
    return prediction, image_path

############__________________________ANGEL AND DIST CALC____________
def calculate_angle(pixel_x, pixel_y):
    camera_h=480
    camera_w=640
    # Calculate the coordinates of the middle base of the image
    middle_base_x = camera_w / 2
    middle_base_y = camera_h+435

    # Calculate the differences in x and y coordinates
    delta_x = pixel_x - middle_base_x
    delta_y = middle_base_y - pixel_y
    # Calculate the angle in radians
    angle_radians = math.atan2(delta_x, delta_y)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

     # Ensure the angle is negative if the pixel is to the left
    if delta_x < 0:
        angle_degrees = -abs(angle_degrees)
    else:
        angle_degrees = abs(angle_degrees)
    return angle_degrees

    # retrn distance to ball in meter (cm)
def ball_width_to_distance(b_width):
    #distance= 78.64*pow(1.164,(-0.130*b_width+12.2))+22.573 #v2.1
    distance= 94.502*pow(1.401,(-0.126*b_width+4.938))+22.914 # for v1.3 cam
    return distance

#IR SENSOR SETUP (for tennis ball in claw detection)

import RPi.GPIO as GPIO
import pigpio
import time
import math
from gpiozero import DistanceSensor # Using this library
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

# Define pins
ENCA = 16 # GPIO 16 (Pin 36)
ENCB = 26  # GPIO 26 (Pin 37)
PWM = 11#17  # GPIO 17 (Pin 11)
PWM_RIGHT= 8#25 #GPIO 25 (pin 22)
IN1 = 23  # GPIO 23 (Pin 16)
IN2 = 24  # GPIO 24 (Pin 18)
IN3 = 7 #5  # GPIO 5 (Pin 29)
IN4 = 0#6  # GPIO 6 (Pin 31)
#LED_PIN = 27  # GPIO 27 (Pin 13)
#SERVO_PICKUP = 12  # GPIO 22 (Pin 15) 

#ENCODER LOCAL
# Encoder ticks and wheel direction
ticks_per_revolution = 900  # Adjust based on your encoder
wheel_radius = 0.027*100  # Radius of wheels in cm
wheel_base = 21.5#100*0.227  # Distance between wheels in cm
direction1 = 1  # Direction of wheel 1
direction2 = 1  # Direction of wheel 2
# Global variables for encoder positions
posi1 = 0
posi2 = 0
# Robot position and orientation
x_pos = 0
y_pos = 0
theta_pos = 0  # Orientation in radians

# List to keep track of robot's path
path_x = [x_pos]
path_y = [y_pos]


# Global variables
posi = 0
prevT = time.time()
eprev = 0
eintegral = 0
start_slow=0
#TARGET=int(700*(20)/(2*math.pi*2.7)) #cm 800
#TARGET=int(720*(15)/(2*math.pi*2.7)) #cm 800

#TARGET=int(700*(19*180/90)/(2*math.pi*2.7)) #cm 800
#TARGET=int(655*(19*180/90)/(2*math.pi*2.7)) #cm 800

GPIO.setwarnings(False)
# Setup
GPIO.setmode(GPIO.BCM)

#GPIO.setup(ENCA, GPIO.IN)
GPIO.setup(ENCA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENCB, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#GPIO.setup(ENCB, GPIO.IN)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
#GPIO.setup(LED_PIN, GPIO.OUT)
#GPIO.output(LED_PIN, GPIO.LOW)

# GPIO.setup(SERVO_PICKUP, GPIO.OUT)
# PWM_SERVO_PICKUP= GPIO.PWM(SERVO_PICKUP, 50)  # 50 Hz (20 ms PWM period)
# PWM_SERVO_PICKUP.start(0)

pi = pigpio.pi()
pi.set_mode(PWM, pigpio.OUTPUT)
pi.set_PWM_frequency(PWM, 1000)  # Set PWM frequency to 1kHz

pi = pigpio.pi()
pi.set_mode(PWM_RIGHT, pigpio.OUTPUT)
pi.set_PWM_frequency(PWM_RIGHT, 1000)  # Set PWM frequency to 1kHz
#____________________________ULTRASONIC_______________________
# Ultra sonic 
sensor = DistanceSensor(echo=4, trigger=25)

#_____________SERVO___________________________
pi_factory = PiGPIOFactory()

servo = AngularServo(13, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=pi_factory)

servo2 = AngularServo(12, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=pi_factory)

#_______________________________ENCODER______________________



def read_encoder1(channel):
    global posi1
    global posi
    global direction1
    if direction1 == 1:
        posi1 += 1
        
    else:
        posi1 -= 1
    posi+=1
def read_encoder2(channel):
    global posi2
    global direction2
    if direction2 == 1:
        posi2 += 1
    else:
        posi2 -= 1
    
GPIO.add_event_detect(ENCA, GPIO.RISING, callback=read_encoder1)
GPIO.add_event_detect(ENCB, GPIO.RISING, callback=read_encoder2)
RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180

def update_position():
    global x_pos, y_pos, theta_pos, posi1, posi2
    
    # Calculate the distance traveled by each wheel
    d2 = (posi1 / ticks_per_revolution) * (2 * np.pi * wheel_radius)
    d1 = (posi2 / ticks_per_revolution) * (2 * np.pi * wheel_radius)
    #print(d1)
    #print(d2)

    #print(f"Distance of motor 1: {d1}, Distance of motor 2: {d2}")
    #print(f"Difference in d1 and d2: {d1-d2}")
    # Reset encoder counts
    posi1 = 0
    posi2 = 0
    
    # Compute robot's change in position and orientation
    d = (d1 + d2) / 2
    delta_theta = np.arcsin((d1 - d2) / wheel_base)*RAD_TO_DEG
    #print(f"Change in Theta: {delta_theta} ")
    # Update the Y coordinate (forward direction)
    y_pos += d * np.cos((theta_pos + delta_theta)*DEG_TO_RAD)
    # Update the X coordinate (sideways movement)
    #print(f"Angle in sin/cos: {theta_pos + delta_theta}")

    x_pos += d * np.sin((theta_pos + delta_theta)* DEG_TO_RAD)
    # Update the orientation (theta)
    theta_pos += delta_theta
    #print(f"Theta Deg: {math.degrees(theta_pos)}")
    #print(f"ANGLE: {theta_pos}")

    # Update the path
    path_x.append(x_pos)
    path_y.append(y_pos)
    
    # Print the updated position and orientation
    #print(f"Position: x={x_pos:.2f} m, y={y_pos:.2f} m, theta={theta_pos:.2f} rad")
    return d1,d2,theta_pos
def save_plot():
    plt.plot(path_x, path_y, 'b-')
    plt.title('Robot Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')  # Set equal scaling for x and y axes

    plt.savefig('robot_path.png')  # Save the plot as a PNG file


def set_motor_drive(dir, pwm_val,pwm_adjustment):
    global direction1
    global direction2
    if dir == 1:
        
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        direction1=1
        direction2=1
    elif dir == -1:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        direction1=-1
        direction2=-1
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
    threshold_value=14
    if pwm_val-threshold_value<0:
        right_drive=0
    else:
        right_drive=pwm_val-threshold_value
    #pi.set_PWM_dutycycle(PWM_RIGHT, right_drive)
    #pi.set_PWM_dutycycle(PWM, pwm_val)

       # Adjust PWM for each wheel
    pi.set_PWM_dutycycle(PWM_RIGHT, np.clip((right_drive + pwm_adjustment),0,255))
    pi.set_PWM_dutycycle(PWM, np.clip((pwm_val - pwm_adjustment),0,255))
    # print(f"pwm_RIGHT: {np.clip((right_drive + pwm_adjustment),0,255)}")
    # print(f"pwm_LEFT: {np.clip((pwm_val - pwm_adjustment),0,255)}")


def set_motor_rotate(dir, pwm_val):
    global direction1
    global direction2
    if dir == 1:
        
        GPIO.output(IN1, GPIO.HIGH)  # Need to change depending on rotation direction (opposite for forward)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        direction1=1
        direction2=-1
    elif dir == -1:

        GPIO.output(IN1, GPIO.LOW)  # Need to change depending on rotation direction (opposite for forward)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        direction1=-1
        direction2=1
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
    pi.set_PWM_dutycycle(PWM_RIGHT, pwm_val)
    pi.set_PWM_dutycycle(PWM, pwm_val)



def rotate(angle):
    #TARGET=int(655*(19*angle/90)/(2*math.pi*2.7)) #cm 800
    #TARGET=int(620*(19*angle/90)/(2*math.pi*2.7)) #cm 800
    #TARGET=int(810*(19*angle/90)/(2*math.pi*2.7)) #cm 800
    global theta_pos
    eintegral = 0
    captured_theta_pos = theta_pos
    target_angle=captured_theta_pos+angle
    #global posi
    #posi=0
    start_slow=0
    prevT = time.time()
    eprev = 0
    pos_stored=[]
    i=1


    kp = 5.7
    kd = 0.12
    ki = 0.0001
    while True:
        # kp = 5
        # kd = 0.009
        # ki = 0.001

       
        #Need to check which PID values wer wthe one for turning the the one for driving
        
        # Time difference
        currT = time.time()
        deltaT = currT - prevT
        prevT = currT
        
        # Error calculation
        #pos = np.sign(TARGET)*abs(posi)
        pos_stored.append(theta_pos)
        e = target_angle-theta_pos
        
        # Derivative
        dedt = (e - eprev) / deltaT
        
        # Integral
        eintegral += e * deltaT
        
        # Control signal
        #print(f"Error: {e}, Derivitive: {dedt}, Intergral: {eintegral}")

        u = kp * e + kd * dedt + ki * eintegral
        #print(f"U Value (PWM): {u}")

        
        # Motor power
        pwr = abs(u)
        if pwr > 255:
            pwr = 255
        #pwr = np.interp(pwr, [0, 255], [45, 255])
        pwr = np.interp(pwr, [0, 255], [50, 255])

        #print(f"PWM: {pwr}")
        # Motor direction
        dir = 1 if -angle >= 0 else -1
        #print(f"PMW: {abs(int(pwr-(255-start_slow)))}")
        # Signal the motor
        # if (pwr+255-start_slow)<40:
        #     set_motor_rotate(0, 0)
        #     break

        if len(pos_stored)>4:
            last_four_values = pos_stored[-4:]
            #print(last_four_values)
            # Check if all values are the same
            #print(set(last_four_values))
            #print(len(set(last_four_values)))
            if len(set(last_four_values)) == 1:
                if i==1:
                    int_pwm=pwr
                    i+=1
                #print("adding to pwm")
                # Do something if all values are the same
                value=15
                if (int_pwm+value)>255:
                    pwr=255
                else:
                    int_pwm+=value
                    pwr=int_pwm

                #print(abs(int(pwr-(255-start_slow))))
        #print(f"pwm: {int(pwr-(255-start_slow))}")
        


        set_motor_rotate(dir, abs(int(pwr-(255-start_slow))))
        
        # Store previous error
        eprev = e
        
        # Output
        if abs(e)<0.5:
            set_motor_rotate(0, 0)
            break

        if start_slow==255:
            pass
        else:
            start_slow+=5
        update_position()
        time.sleep(0.01)  # Small delay to prevent CPU overload

def drive(distance):
    #TARGET=int(720*(distance)/(2*math.pi*2.7)) #cm 800
    #TARGET=int(1000*(distance)/(2*math.pi*2.7)) #cm 800
    TARGET=int(900*(distance)/(2*math.pi*2.7)) #cm 800
    #Target_sign=TARGET
    eintegral = 0
    global posi
    global direction1
    posi=0
    start_slow=0
    prevT = time.time()
    eprev = 0

    kp = 0.675
    kd = 0.005
    ki = 0.01
    #kp_pwm=1500
    kp_pwm=1100
    pos_stored=[]
    i=1

    while True: 
        
        
        #kp = 0.55
        #kd = 0.005
        #ki = 0.01  
        # Time difference
        currT = time.time()
        deltaT = currT - prevT
        prevT = currT
        
        # Error calculation
        pos = abs(posi)
        pos_stored.append(pos)

        e = pos - abs(TARGET)
        
        # Derivative
        dedt = (e - eprev) / deltaT
        
        # Integral
        eintegral += e * deltaT
        
        # Control signal
        u = kp * e + kd * dedt + ki * eintegral
        
        # Motor power
        pwr = abs(u)
        if pwr > 255:
            pwr = 255
        
        if len(pos_stored)>4:
            last_four_values = pos_stored[-4:]
            #print(last_four_values)
            # Check if all values are the same
            #print(set(last_four_values))
            #print(len(set(last_four_values)))
            if len(set(last_four_values)) == 1:
                if i==1:
                    int_pwm=pwr
                    i+=1
                #print("adding to pwm")
                # Do something if all values are the same
                value=15
                if (int_pwm+value)>255:
                    pwr=255
                else:
                    int_pwm+=value
                    pwr=int_pwm


        # Motor direction
        dir = 1 if TARGET >= 0 else -1
        #print(dir)

        # Signal the motor
        if (pwr+255-start_slow)<50:
            set_motor_drive(0, 0,0)
            time.sleep(0.5)

            break
        
        # IF IR sensor senses tennis ball then break drive loop
        #if read_IR_sensor()>0:
        #    set_motor_drive(0, 0)
        #    break

        #IF UNLTA SONIC DETECTION
        # STOP MOTORS 
        # WAIT UNTILL OBJECT PASSES
        # THEN CONTUINE
        d1,d2,_=update_position()
        #print(f"D1: {d1}, D2: {d2}")
        pwm_adjustment = kp_pwm * -(d1 - d2)
        if direction1==-1:
            pwm_adjustment=-pwm_adjustment
        #print(f"PWM ADJSUT: {pwm_adjustment}")
        set_motor_drive(dir, abs(int(pwr-(255-start_slow))),pwm_adjustment)
        
        # Store previous error
        eprev = e
        
        # Output
        #print(f"Target: {TARGET}, Position: {pos}")
        print(f"Error to drive: {e}")
        if abs(e)<5:
            set_motor_drive(0, 0,0)
            time.sleep(0.5)
            break

        if start_slow==255:
            pass
        else:
            start_slow+=5
        time.sleep(0.01)  # Small delay to prevent CPU overload


#____________________LOCALISATION FUNCTION__________________
def calculate_new_position(x, y, angle, distance):
    # Convert angle to radians for trigonometric functions
    angle_rad = math.radians(angle)
    # Calculate new x and y coordinates
    new_x = x + distance * math.cos(angle_rad)
    new_y = y + distance * math.sin(angle_rad)
    return new_x, new_y

def calculate_return_home(x, y, current_angle):
    # Calculate distance to home (0,0)
    distance_to_home = math.sqrt(x**2 + y**2)
    # Calculate angle to home
    #angle_to_home = math.degrees(math.atan2(-y, -x)) - current_angle
    # Calculate the angle to the origin from the current position
    angle_to_origin = math.degrees(math.atan2(-x, -y))
    
    # Calculate the difference between the current orientation and the angle to the origin
    angle_difference = angle_to_origin - current_angle
    
    # Normalize the angle to be within -180 to 180 degrees
    while angle_difference > 180:
        angle_difference -= 360
    while angle_difference < -180:
        angle_difference += 360
    
    return distance_to_home, angle_difference


def ultrasonic_ball():
    dist = sensor.distance * 100 
    print("Distance: {:.1f} cm".format(dist))
    if dist<7.5:#9:
        print("PICK UP")
        return True
    return False

def ultrasonic_box():
    dist = sensor.distance * 100 
    print("Distance: {:.1f} cm".format(dist))
    if dist<35:#30:
        print("stop")
        return True
    return False
############################____________MAIN_____________###########
# Main script to integrate everything

#_______________ LINE CHECK______________
def detect_lines_and_check_point(image_f, x, y):
    # Load image
    image = np.copy(image_f)
    original_image = np.copy(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    #cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    # Make the top half black
    image[0:height//3, :] = 0 
    #Convert the image to HLS color space for better white detection
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    #Define white color range in HLS
    #lower_white = np.array([0, 233, 0], dtype=np.uint8)
    #lower_white = np.array([0, 185, 0], dtype=np.uint8)
    lower_white = np.array([0, 210, 0], dtype=np.uint8)

    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # coour def in HSV
    # lower_white = np.array([0, 0,155], dtype=np.uint8)
    # upper_white = np.array([255, 160, 255], dtype=np.uint8)
    
    # # Create a mask for white colors
    # white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=white_mask)
    #cv2.imwrite("drive_vid_3_line_mask.jpg", masked_image)

    cv2.imwrite("drive_vid_3_line_mask.jpg", white_mask)

    # Convert image to grayscale
    #gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # Convert the masked image from HSV to BGR
    # bgr_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)

    # # Now, convert the BGR image to grayscale
    # gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None:
        print("No lines detected.")
        return 0, None  # or return an appropriate response indicating no lines were found
    
    # Function to extend lines across the image height
    # def extend_line(slope, intercept, height):
    #     y1 = height  # bottom of the image
    #     y2 = 0  # top of the image
    #     # Calculate x1 and x2 based on the slope and intercept
    #     x1 = int((y1 - intercept) / slope)
    #     x2 = int((y2 - intercept) / slope)
        #     return x1, y1, x2, y2
    def extend_line(slope, intercept, height, width):
        if slope == 0:
            # Horizontal line case
            y1 = intercept
            y2 = intercept
            x1 = 0
            x2 = width
        else:
            # Calculate x1 and x2 based on the slope and intercept
            y1 = height  # bottom of the image
            y2 = 0  # top of the image
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        
        return x1, y1, x2, y2
    # Helper function to merge similar lines
    def merge_similar_lines(lines_slope_intercept, slope_threshold=0.1, intercept_threshold=100):
        merged_lines = []
        for slope, intercept in lines_slope_intercept:
            if not merged_lines:
                merged_lines.append((slope, intercept))
            else:
                added = False
                for i, (merged_slope, merged_intercept) in enumerate(merged_lines):
                    if abs(slope - merged_slope) < slope_threshold and abs(intercept - merged_intercept) < intercept_threshold:
                        # Average the slopes and intercepts
                        new_slope = (slope + merged_slope) / 2
                        new_intercept = (intercept + merged_intercept) / 2
                        merged_lines[i] = (new_slope, new_intercept)
                        added = True
                        break
                if not added:
                    merged_lines.append((slope, intercept))
        return merged_lines

    # Create a list to store the lines in slope-intercept form
    lines_slope_intercept = []

    # Calculate the slope and intercept for each detected line
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            lines_slope_intercept.append((slope, intercept))

    # Merge similar lines
    merged_lines = merge_similar_lines(lines_slope_intercept)
    print(merged_lines)
    # merged_lines = merged_lines[:-2] + merged_lines[-1:]

    # Draw the extended and merged lines on the original image
    for slope, intercept in merged_lines:
        x1, y1, x2, y2 = extend_line(slope, intercept, height,width)
        #print(x1, y1, x2, y2)
        cv2.line(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Save the image with lines
    cv2.imwrite("drive_vid_4_line_img.jpg", original_image)

    # Function to determine if the point is above or below the detected lines
    # def is_point_above_or_below(x, y, lines_slope_intercept):
    #     for slope, intercept in lines_slope_intercept:
    #         line_y = slope * x + intercept
    #         if y < line_y:
    #             print(f"The point ({x}, {y}) is above the line with slope {slope:.2f} and intercept {intercept:.2f}.")
    #             return 1  # Above the line
    #         elif y > line_y:
    #             print(f"The point ({x}, {y}) is below the line with slope {slope:.2f} and intercept {intercept:.2f}.")
    #             return 0  # Below the line
    #     return -1  # Default case, if no line is found
    def is_point_above_or_below(x, y, lines_slope_intercept):
        above_all = True
        below_all = True
        
        for slope, intercept in lines_slope_intercept:
            # Handle horizontal lines separately
            if slope == 0:
                line_y = intercept
            else:
                line_y = slope * x + intercept
            
            if y < line_y:
                below_all = False
            elif y > line_y:
                above_all = False
        
        # Determine the result based on above and below conditions
        if above_all:
            print(f"The point ({x}, {y}) is above all lines.")
            return 1  # Above all lines
        elif below_all:
            print(f"The point ({x}, {y}) is below all lines.")
            return 0  # Below all lines
        else:
            print(f"The point ({x}, {y}) is above at least 1 line.")
            return 1 # If neither above nor below all lines
    # Check if the point is above or below the merged lines
    result = is_point_above_or_below(x, y, merged_lines)
    return result, merged_lines
rot_counter=0
inital_angle=0
def colour_drive(min_contour_area=50):#100 
            # Capture frame-by-frame
        global rot_counter, inital_angle
        # lower_bound = np.array([28, 40, 75])
        # upper_bound = np.array([32, 255, 250])

        lower_bound2 = np.array([28, 30, 75])
        #upper_bound2 = np.array([38, 255, 250])
        upper_bound2 = np.array([36, 255, 250])


        #lower_bound = np.array([28, 40, 75])

        # upper_bound = np.array([35, 255, 255])

        frame = picam2.capture_array()

        # Resize the frame for faster processing (optional)
        resized_frame = cv2.resize(frame, (640, 480))
        #print("Initial image color space:", detect_color_space(resized_frame))
        #resized_frame[0:195, :] = 0  # Set the top half to black (0 for all color channels)
        resized_frame[0:205, :] = 0  # Set the top half to black (0 for all color channels)

        # Convert to HSV color space
        #resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2RGB)

    #    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)

        # Apply color thresholding
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        cv2.imwrite('drive_vid_1_colour_thresh.jpg', mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        

        if contours:
            # Find the largest contour
            #largest_contour = max(contours, key=cv2.contourArea)
            #largest_contour = max(contours, key=lambda c: c[c[:, :, 1].argmin()][0][1])
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            if valid_contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) >300:
                    pass
                else: 
                    largest_contour = max(valid_contours, key=lambda c: c[c[:, :, 1].argmin()][0][1])

            # Check if the largest contour area is above the minimum threshold
            # if cv2.contourArea(largest_contour) > min_contour_area:

                #CHECK IF INSIDE LINE________________________________
                lowest_point = max(largest_contour, key=lambda point: point[0][1])
                lowest_x, lowest_y = lowest_point[0]
                above,det_lines=detect_lines_and_check_point(frame, lowest_x, lowest_y)
                if det_lines is None or len(det_lines) == 0:
                    print("No lines were detected in the image.")
                else: 
                    intersection=0
                    for m,c in det_lines:
                        for point in largest_contour:
                            x, y = point[0]
                            # Calculate the y-value of the line at this x
                            line_y = m * x + c
                            # Check if the contour point is close to the line (within a tolerance)
                            if abs(y - line_y) < 3:  # You can adjust the tolerance (1 pixel in this case)
                                print(f"Contour intersects the line at point: ({x}, {y})")
                                intersection=1
                                break
                        if intersection==1:
                            break

                #__________________________________________________
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Draw the contour and center on the frame

                cv2.drawContours(resized_frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(resized_frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(resized_frame, "Ball", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Simulate motor control with print statements
                # print(f"X: {cX}")
                # print(f"Y: {cY}")
                #print("ball detected")
                if above==0 or intersection==1:
                    
                    rot_counter=0
                    if cX < 200:
                        #print("Turning Left")
                        #set_motor_drive(1, 255,200-cX)
                        set_motor_drive(1, 255,-(200-cX))#+cY/2)

                    elif cX > 420:
                        #print("Turning RIGHT")
                        # set_motor_drive(1, 255,-(cX-420))
                        set_motor_drive(1, 255,(cX-420))#+cY/2))

                    else:
                        #print("Going Forward")
                        set_motor_drive(1, 255,0)
                else:
                    print("Ball out of Bounds")
                    set_motor_drive(1, 0,0)
                    set_motor_rotate(1, 255*2/3)
                    rot_counter+=1
                    if rot_counter==1:
                        _,_,inital_angle=update_position()
                    _,_,new_angle=update_position()
                    if rot_counter>30:
                        if abs(inital_angle-new_angle)>300:
                            rotate(180)
                            drive(40)
                            rot_counter=0

                # elif cY < 420: #400:
                #     print("Going Forward")
                #     set_motor_drive(1, 255,0)

                # else:
                #     print("Ball Centered")
                #     set_motor_drive(1, 0,0)

            else:
                # If the largest contour is too small, treat it as no ball detected
                print("Ball not detected - contour too small")
                set_motor_drive(1, 0,0)
                set_motor_rotate(1, 255*2/3)
                rot_counter+=1
                if rot_counter==1:
                    _,_,inital_angle=update_position()
                _,_,new_angle=update_position()
                if rot_counter>30:
                    if abs(inital_angle-new_angle)>300:
                        drive(40)
                        rot_counter=0

        else:
            # Print statement if no ball is detected
            print("Ball not detected")
            set_motor_drive(1, 0,0)
            set_motor_rotate(1, 255*2/3)
            rot_counter+=1
            if rot_counter==1:
                _,_,inital_angle=update_position()
            #print
            _,_,new_angle=update_position()
            if rot_counter>30:
                if abs(inital_angle-new_angle)>300:
                    drive(40)
                    rot_counter=0
        #print(abs(inital_angle-new_angle))
        #print(f"Rot COunter: {rot_counter}")
        final_output = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

        cv2.imwrite('drive_vid_2_tracked_ball.jpg', final_output)
        update_position()
        time.sleep(0.1)
def box_drive_2(min_contour_area=1000):#100 
        # lower_bound = np.array([14, 117, 146])
        # upper_bound = np.array([25, 255, 255])
        global rot_counter, inital_angle

        #lower_bound = np.array([0, 40, 95])
        # lower_bound = np.array([0, 40, 80])
        # upper_bound = np.array([25, 255, 255])

        lower_bound2 = np.array([142, 18, 50])
        upper_bound2 = np.array([179, 255, 100])
        
        #NEW
        lower_bound = np.array([0, 50, 50])
        upper_bound = np.array([29, 205, 255])
        # Capture frame-by-frame
        frame = picam2.capture_array()

        # Resize the frame for faster processing (optional)
        resized_frame = cv2.resize(frame, (640, 480))

        # Make the top half black
        resized_frame[0:210, :] = 0  # Set the top half to black (0 for all color channels)

# 
        #print("Initial image color space:", detect_color_space(resized_frame))

        # Convert to HSV color space
        #resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2RGB)

    #    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)

        # Apply color thresholding
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
        mask=mask | mask2
        cv2.imwrite('drive_vid_5_box_colour_thresh.jpg', mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        if contours:
            # Find the largest contour
            #largest_contour = max(contours, key=cv2.contourArea)
            #largest_contour = max(contours, key=lambda c: c[c[:, :, 1].argmin()][0][1])
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            if valid_contours:
                largest_contour = max(contours, key=cv2.contourArea)
                print(f"AREA OF CONTOUR: {cv2.contourArea(largest_contour)}")
                if cv2.contourArea(largest_contour) >2500:
                    if cv2.contourArea(largest_contour) >70000: #75000
                        set_motor_drive(1, 0,0)
                        print("AT BOX STOP")
                        return True
                        time.sleep(5)                        
                    pass
                else: 
                    largest_contour = max(valid_contours, key=lambda c: c[c[:, :, 1].argmin()][0][1])

            # Check if the largest contour area is above the minimum threshold
            # if cv2.contourArea(largest_contour) > min_contour_area:

                #CHECK IF INSIDE LINE________________________________
                lowest_point = max(largest_contour, key=lambda point: point[0][1])
                lowest_x, lowest_y = lowest_point[0]
                above,det_lines=detect_lines_and_check_point(frame, lowest_x, lowest_y)
                if det_lines is None or len(det_lines) == 0:
                    print("No lines were detected in the image.")
                else: 
                    intersection=0
                    for m,c in det_lines:
                        for point in largest_contour:
                            x, y = point[0]
                            # Calculate the y-value of the line at this x
                            line_y = m * x + c
                            # Check if the contour point is close to the line (within a tolerance)
                            if abs(y - line_y) < 3:  # You can adjust the tolerance (1 pixel in this case)
                                print(f"Contour intersects the line at point: ({x}, {y})")
                                intersection=1
                                break
                        if intersection==1:
                            break

                #__________________________________________________
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Draw the contour and center on the frame

                cv2.drawContours(resized_frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(resized_frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(resized_frame, "Box", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Simulate motor control with print statements
                # print(f"X: {cX}")
                # print(f"Y: {cY}")
                #print("ball detected")
                if above==0 or intersection==1:
                    
                    rot_counter=0
                    if cX < 200:
                        #print("Turning Left")
                        #set_motor_drive(1, 255,200-cX)
                        set_motor_drive(1, 255,-(200-cX))#+cY/2)

                    elif cX > 370:#420:
                        #print("Turning RIGHT")
                        # set_motor_drive(1, 255,-(cX-420))
                        set_motor_drive(1, 255,(cX-370))#(cX-420))

                    else:
                        #print("Going Forward")
                        set_motor_drive(1, 255,0)
                else:
                    print("Detected out of Bounds")
                    set_motor_drive(1, 0,0)
                    set_motor_rotate(1, 255*2/3)
                    rot_counter+=1
                    if rot_counter==1:
                        _,_,inital_angle=update_position()
                    _,_,new_angle=update_position()
                    if rot_counter>30:
                        if abs(inital_angle-new_angle)>300:
                            rotate(180)
                            drive(40)
                            rot_counter=0

                # elif cY < 420: #400:
                #     print("Going Forward")
                #     set_motor_drive(1, 255,0)

                # else:
                #     print("Ball Centered")
                #     set_motor_drive(1, 0,0)

            else:
                # If the largest contour is too small, treat it as no ball detected
                print("Box not detected - contour too small")
                largest_contour_small = max(contours, key=cv2.contourArea)
                print(f"AREA OF CONTOUR: {cv2.contourArea(largest_contour_small)}")

                set_motor_drive(1, 0,0)
                set_motor_rotate(1, 255*2/3)
                rot_counter+=1
                if rot_counter==1:
                    _,_,inital_angle=update_position()
                _,_,new_angle=update_position()
                if rot_counter>30:
                    if abs(inital_angle-new_angle)>300:
                        drive(40)
                        rot_counter=0

        else:
            # Print statement if no ball is detected
            print("Box not detected")
            set_motor_drive(1, 0,0)
            set_motor_rotate(1, 255*2/3)
            rot_counter+=1
            if rot_counter==1:
                _,_,inital_angle=update_position()
            print
            _,_,new_angle=update_position()
            if rot_counter>30:
                if abs(inital_angle-new_angle)>300:
                    drive(40)
                    rot_counter=0
        #print(abs(inital_angle-new_angle))
        print(f"Rot COunter: {rot_counter}")
        final_output = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

        cv2.imwrite('drive_vid_6_tracked_box.jpg', final_output)
        update_position()
        time.sleep(0.1)
        return False



def box_drive(min_contour_area=100):
        
        # lower_bound = np.array([14, 117, 146])
        # upper_bound = np.array([25, 255, 255])
        
        lower_bound = np.array([0, 40, 95])
        upper_bound = np.array([25, 255, 255])


        lower_bound2 = np.array([142, 18, 50])
        upper_bound2 = np.array([179, 255, 100])

        # Capture frame-by-frame
        frame = picam2.capture_array()

        # Resize the frame for faster processing (optional)
        resized_frame = cv2.resize(frame, (640, 480))

        # Make the top half black
        resized_frame[0:195, :] = 0  # Set the top half to black (0 for all color channels)

# 
        #print("Initial image color space:", detect_color_space(resized_frame))

        # Convert to HSV color space
        #resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2RGB)

    #    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2HSV)

        # Apply color thresholding
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
        mask=mask | mask2
        cv2.imwrite('box_colour_thresh.jpg', mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Check if the largest contour area is above the minimum threshold
            if cv2.contourArea(largest_contour) > min_contour_area:
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Draw the contour and center on the frame

                cv2.drawContours(resized_frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(resized_frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(resized_frame, "Ball", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Simulate motor control with print statements
                #print(f"X: {cX}")
                #print(f"Y: {cY}")
                if cX < 200:
                    #print("Turning Left")
                    #set_motor_drive(1, 255,200-cX)
                    set_motor_drive(1, 255,-(200-cX))#+cY/2)

                elif cX > 420:
                    #print("Turning RIGHT")
                    # set_motor_drive(1, 255,-(cX-420))
                    set_motor_drive(1, 255,(cX-420))#+cY/2))

                else:
                    #print("Going Forward")
                    set_motor_drive(1, 255,0)
                
            else:
                # If the largest contour is too small, treat it as no ball detected
                print("Box not detected - contour too small")
                set_motor_drive(1, 0,0)
                set_motor_rotate(1, 255)

        else:
            # Print statement if no ball is detected
            print("Box not detected")
            set_motor_drive(1, 0,0)
            set_motor_rotate(1, 255)

        final_output = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

        cv2.imwrite('box_drive_int.jpg', final_output)
        update_position()
        time.sleep(0.1)


def servo_pickup():
        servo.angle = 90 #90
        time.sleep(4)
        servo.angle = -8
        time.sleep(2.5)

def servo_deposit():
        servo2.angle = 90
        time.sleep(4)
        # servo2.angle = -15
        # time.sleep(2)

if __name__ == "__main__":
    start_time=time.time()
    #current_x, current_y = 0, 0
    #current_angle = 0  # Initial angle facing right (0 degrees)
    #path = [(current_x, current_y)]
    ball_counter=0
    STORAGE_CAPACITY=4
    drive_count=0
    num_of_photos=3
    turn_count=0

    lower_bound = np.array([28, 40, 75])
    upper_bound = np.array([35, 255, 255])

    # Minimum contour area to consider as a valid tennis ball
    min_contour_area = 100  # Adjust as needed
    ball_det=0

    # Initialize the servo with the pigpio pin factory
    servo.angle = -10
    #servo2.angle = 0

    servo2.angle = -15

    #print("Moving servo")
    #time.sleep(2)
    function_mode=1 # 0= Tennis Balls, 1=Deposit Box
    try: 
        # Initialize the camera
        picam2 = Picamera2()
        camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(2)  # Allow the camera to warm up

        while True:
            #ultrasonic_box()
            #if bounding box dection =0 then colour drive (tennis ball)
            run_time=time.time()-start_time
            stop_time=60*8.5
            if run_time>=stop_time:
                    print("1.5 minute")
            if (run_time>stop_time) and not ultrasonic_box() and ball_counter>=1:
                function_mode=1
                print("1.5 minute left, deposit balls!")

            
            if (function_mode==0):
                # ultrasonic_ball()
                colour_drive()
                if ultrasonic_ball():
                    set_motor_drive(1,0,0)
                    servo_pickup()
                    ball_counter+=1
                    if ball_counter==STORAGE_CAPACITY:
                        function_mode=1
            else:
                # ultrasonic_box()
                stop_at_box=box_drive_2()
                if stop_at_box: # now at box
                #if ultrasonic_box() or stop_at_box: # now at box
                    set_motor_drive(1,0,0)
                    time.sleep(2)
                    #deposit box sequence (reverse and rotate)
                    # drive(-30)
                    # time.sleep(2)
                    rotate(180)
                    time.sleep(2)
                    drive(-20)
                    time.sleep(2)
                    servo_deposit()
                    function_mode=0     
                    ball_counter=0  
                    drive(25)
                    servo2.angle = -15
                    time.sleep(2)
         
    except KeyboardInterrupt:
        pass

    finally:
        save_plot()
        picam2.stop()
        picam2.close()
        GPIO.cleanup()
        pi.stop()
        pi_factory.close()

