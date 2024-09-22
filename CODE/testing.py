from googletrans import Translator
import cv2
import pytesseract
import re
def Marathi(img):
    try:
        #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        #img = enlarge_img(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("cropped",img)
        #cv2.waitKey(0)
        options = "-l {} ".format("mar")
        text = pytesseract.image_to_string(img, config=options,)
        trans = Translator()
        #print(text)
        translated = trans.translate(text,src="mr")
        text1 = translated.text
        text = re.sub(r'[^a-zA-Z0-9\n]', '', text1)
        #text = re.sub('[Mm]aharashtra', 'MH', text)
        text = re.sub('\n',' ',text)
        temp = 0
        for s in text:
            if s.isdigit():
                temp = text.index(s)
                break
        text = "MH"+str(text[temp:])
        #print(text)
        print(text)
    except:
        return("Not Detected")

Marathi(cv2.imread(r'C:\Users\PRITAM\Desktop\m4.jpg'))
