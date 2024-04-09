#!/usr/bin/env python
# coding: utf-8

# In[5]:





# In[1]:


import cv2
from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection

# Initialize cascade classifier for face detection
#This line specifies the path to the XML file containing the pre-trained Haar cascade classifier for face detection.
face_cascade_path = "D:/django_setup/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
#This line initializes a cascade classifier object for face detection using the path specified above.
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if the cascade classifier is loaded successfully
if face_cascade.empty():
    print("Error: Failed to load cascade classifier.")
    exit()

# Initialize video capture
#This line initializes a video capture object, which is used to capture video frames from a camera. 
#The argument 0 specifies the index of the camera device to use (usually the primary camera).
video_cap = cv2.VideoCapture(0)

# Initialize YOLO model for mask detection
#These lines initialize the YOLO model for mask detection using the specified
#pre-trained model from the Hugging Face model hub.
feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-finetuned-masks')
model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-masks')

#This line starts an infinite loop, which continuously captures video frames, detects faces, 
#and performs mask detection until the program is manually terminated.

while True:
    # Read frame from video capture
    #This line reads a frame from the video capture object. 
    #The ret variable indicates whether the frame was successfully read, and frame contains the captured frame data.
    ret, frame = video_cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    #This line detects faces in the grayscale frame using the Haar cascade classifier. 
    #It returns a list of rectangles specifying the positions of detected faces.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    #This line starts a loop over each detected face, where (x, y) are the coordinates of the top-left corner of the face rectangle, 
    #and (w, h) are the width and height of the rectangle.
    for (x, y, w, h) in faces:
        # Extract face region This line extracts the region of interest (ROI), i.e., the detected face, from the frame. 
        face = frame[y:y+h, x:x+w]
        
        # Convert face to PIL image  This line converts the extracted face region from OpenCV's 
        #BGR color space to RGB color space and then creates a PIL Image object from it.
        pil_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        

        #These lines perform object detection using the YOLO model on the face region extracted from the frame.
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Retrieve predicted bounding boxes and logits
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        
        # Threshold for mask confidence
        threshold = 0.1
        
        # Initialize has_mask variable to False
        has_mask = False
        
        # Check if any bounding box has a high confidence for the "mask" class
        #This line starts a loop over each predicted bounding box and its corresponding confidence score 
        #from the YOLO model's output.
        for logit, bbox in zip(logits[0], bboxes[0]):
            
            # Extract the confidence score for the "mask" class (class 1)
            mask_confidence = logit[1].item()  # Index 1 corresponds to class 1 (mask)
            
            # Check if the confidence score is above the threshold
            if mask_confidence > threshold:
                has_mask = True
                break
        
        # Draw rectangle around the face and display mask detection result
        #This line sets the color for drawing the rectangle around the face based on whether a mask is detected or not.
        color = (0, 255, 0) if has_mask else (0, 0, 255)
        #This line draws a rectangle around the detected face on the frame using the specified color
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        #This line adds text indicating mask detection result (either "Mask" or "No Mask") near the
        #top of the rectangle around the face.
        cv2.putText(frame, "Mask" if has_mask else "No Mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the frame with detected faces and mask detection result
    #This line displays the frame with detected faces and mask detection results.
    cv2.imshow("Face Mask Detection", frame)
    
    # Check for key press to exit
    #This line checks for the 'q' key press. If the 'q' key is pressed, it breaks out of the infinite loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#These lines release the video capture object and close all OpenCV windows after the loop exits.
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




