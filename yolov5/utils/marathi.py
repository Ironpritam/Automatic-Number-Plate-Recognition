from googletrans import Translator
import cv2
import pytesseract
import re
from check2 import enlarge_img
from check2 import Preprocessing
def Marathi(img):
    try:
        #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.GaussianBlur(img, (3,3), 0)

        img = cv2.bilateralFilter(img, 11, 17, 17)
        #img = enlarge_img(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("cropped",img)
        #cv2.waitKey(0)
        cv2.imwrite("uploads/marathi_preprocessed.png", img)
        #img =Preprocessing(img)
        options = "-l {} ".format("mar")
        text = pytesseract.image_to_string(img, config=options,)
        trans = Translator()
        print(text)
        translated = trans.translate(text,src="mr")
        text1 = translated.text
        print(text1)
        text = re.sub(r'[^a-zA-Z0-9\n]', '', text1)
        #text = re.sub('[Mm]aharashtra', 'MH', text)
        text = re.sub('\n',' ',text)
        #print(text)
        return(text)
    except:
        return("Not Detected")
