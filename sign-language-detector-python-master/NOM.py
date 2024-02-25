import os
import mediapipe as mp
import cv2

import matplotlib.pyplot as plt
# LETS PLOT HOW THIS IMAGE LOOKS LIKE


data_dir = 'sign-language-detector-python-master\sign-language-detector-python-master\data'
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

data= []
labels= []
# these object are used to draw the landmarks over over
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands()
for dir_ in os.listdir(data_dir):
    
        # mtplot requires your image to be in rgb to analyse the image
    for img_path in os.listdir(os.path.join(data_dir,dir_)):
        #TO CONVERT INTO RGB TO READ IT IN MEDIA PIPE
        img = cv2.imread(os.path.join(data_dir,dir_,img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data_aux =[]
        
        results = hands.process(img_rgb)
        #  these object are used to draw the landmarks over over
        if results.multi_hand_landmarks:
            #to check atleast one hand is detected
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i])
                    # using coordinate value we wwill calculate a very long array 
                    # then we r going to train the classifier
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
                    data_aux.append(y)
                
                
                data.append(data_aux)
                labels.append(dir_)
                # for i in range(len(hand_landmarks.landmark)):
                #     x = hand_landmarks.landmark[i].x
                #     y = hand_landmarks.landmark[i].y
                #     data_aux.append(x - min(x_))
                #     data_aux.append(y - min(y_))

f = open('data.pickle', 'wb')
data.pickle.dump({'data': data, 'labels': labels}, f)
f.close()

            

  
  
  
  
  