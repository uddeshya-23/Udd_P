# importing modules
import cv2
import pytesseract
import re
import numpy as np
import glob
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
essera
def rotate_bound(image, angle):
       
        (h, w) = image.shape[:2]
        ### centroid
        (cX, cY) = (w // 2, h // 2)
        ### creating rotation matrix
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH))







images = [cv2.imread(file) for file in glob.glob("M/*.jpeg")]#paste the path of image folder and format of image should be jpeg
for image in images:

    image = cv2.resize(image,(0,0),fx=1.5,fy=1.5)
    # print(image)
    newdata=pytesseract.image_to_osd(image)
        ###filter angle value
    angle=re.search('(?<=Rotate: )\d+', newdata).group(0)
    print('osd angle:',angle)
        ### rotating image with angle
    skew_corrected_image=rotate_bound(image,float(angle))
    #converting image into gray scale image
    gray_image = cv2.cvtColor(skew_corrected_image, cv2.COLOR_BGR2GRAY)
    # # converting it to binary image by Thresholding
    
    # this step is require if you have colored image because if you skip this part
    
    # # then tesseract won't able to detect text correctly and this will give incorrect result
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # display image
    
    
    cv2.imshow("threshold image",threshold_img)
    
    # Maintain output window until user presses a key
    
    cv2.waitKey(0)
    
    # Destroying present windows on screen
    
    cv2.destroyAllWindows()
    #configuring parameters for tesseract
    
    custom_config = r'--psm 6'
    
    # now feeding image to tesseract
    
    details = pytesseract.image_to_data(threshold_img , output_type=Output.DICT, config=custom_config)
    
    print(details.keys())
    total_boxes = len(details['text'])
    
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) >20:
            (x,y,w,h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
        # threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
   
    parse_text = []
    
    word_list = []
    
    last_word = ''
    capital =0
    number = 0
    
    for word in details['text']:
       
        if word!='':
    
            word_list.append(word)
    
            last_word = word
    
        if (last_word!='' and word == '') or (word==details['text'][-1]):
    
            parse_text.append(word_list)
    
            word_list = []
    
    
    import csv
    
    with open('ret_sultext.txt','w', newline="") as file:
    
        csv.writer(file, delimiter=" ").writerows(parse_text)
  
    textfile = open('ret_sultext.txt', 'r')
    filetext = textfile.read()
    textfile.close()
    matches = re.findall("[A-Z]{5}[0-9]{4}[A-Z]{1}", filetext)
    dob=re.findall("[0-9]{2}/[0-9]{2}/[0-9]{4}", filetext)
    print("PAN is:",matches,"DOB is: ",dob)
  