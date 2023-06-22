import cv2 #opencv
import os
import time
import uuid

IMAGES_PATH=r'C:\Users\hardi\MyProjects/Project-Sign_Detection/Tensorflow/workspace/images/collectedimages'

labels = ['Hello','Thanks','Yes','No','ILoveYou']
number_imgs =15


for label in labels:
    cnt = 0
    os.mkdir(os.path.join(IMAGES_PATH, label))
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label+'-'+'{}.jpg'.format(str(cnt)))
        cnt+=1
#         imgname = os.path.join(IMAGES_PATH, label, label+'-'+'{}.jpg'.format(str(uuid.uuid1())))
        print(f'answer is : {cap.isOpened()}')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame',frame)
        time.sleep(2)
        
    if cv2.waitKey(1):
        print("Ended")
        continue
        #         break
    cap.release()

    
cv2.destroyAllWindows()