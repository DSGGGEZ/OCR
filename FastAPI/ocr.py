import torch
import cv2
import easyocr
from PIL import Image, ImageTk
import numpy as np
import os
import openpyxl
import time

reader = easyocr.Reader(['th'], gpu=True)
model = torch.hub.load(r'E:\NU\4term2\ControlDataLtdCDG\6thweek\FastAPI\yolov5', 'custom' , path=r"E:\NU\4term2\ControlDataLtdCDG\6thweek\FastAPI\yolov5\licenseplate.pt" , source='local')
    
def cut_symbols(text):
    text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("|", "")
    text = text.replace("\\", "")
    text = text.replace("/", "")
    text = text.replace("@", "")
    text = text.replace("^", "")
    text = text.replace(">", "")
    text = text.replace("<", "")
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = text.replace("#", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("=", "")
    text = text.replace("+", "")
    text = text.replace("-", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace("]", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    return text

def fulfill_word(text):
    words = ["กรง","เทพ","มหา","แทพ","นทา","กรเ","มทา","นหา","กรแ"]
    
    for word in words:
        index = text.find(word)
        if index != -1:
            text = "กรุงเทพมหานคร"
        else:
            text=text
    return text

def ocr(result_image):
    print("OCR")
    txts = []
    results = reader.readtext(result_image)
    for bbox, text, conf in results:
        text = cut_symbols(text)
        text = fulfill_word(text)
        txts.append(text)
        if text.strip():
            # f.write(f"{file_name}: {text}\n")
            if text == "กรุงเทพมหานคร":
                break
        else:
            print()
    
    return txts  
        
def transform(model_cropped , approx2):
    print("Transform")
    print(approx2)
    # maxX maxY minX minY medytop medybottom 
    tr = (approx2[0], approx2[5])
    tl = (approx2[2] , approx2[3])
    br = (approx2[0] , approx2[1])
    bl = (approx2[2] , approx2[4])

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(model_cropped, matrix, (640, 480))
    result = cv2.resize(result, (1000, 400))
    
    return result
    
def corners(approx):
    print("find corners")
    approx2 = []
    x = []
    y = []
    
    for i in range(0, 4, 1):
        x.append(approx[i][0][0])
    
    for i in range(0, 4, 1):
        y.append(approx[i][0][1])
       
    x.sort()
    print("X ",x)
    y.sort()
    print("Y ",y)   
    
    approx2.append(x[3]) #maxX
    approx2.append(y[3]) #maxY
    approx2.append(x[0]) #minX
    approx2.append(y[0]) #minY
    
    approx2.append(y[2]) #medYt
    approx2.append(y[1]) #medYb

    return approx2

def crop_from_contour(model_cropped):
    print("cropping from contour")
    result_image = model_cropped
    approx2 = None
    
    d, sigmaColor, sigmaSpace = 11,17,17
    filtered_img = cv2.bilateralFilter(model_cropped, d, sigmaColor, sigmaSpace)
    
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    
    #***********************************************************************************
    counter = 0
    er = 0
    dt = 0
    result = None
    ct = 0
    
    while ct != 1:
        print("erode at ",er ,"and dilitate at ", dt)

        ret3,threshold = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
        erode = cv2.erode(threshold, None, iterations=er)
        dilated_edges = cv2.dilate(erode, None, iterations=dt)

        edged = cv2.Canny(dilated_edges, 170, 200)
        
        cnts,hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        
        NumberPlateCnt = None
        print("Number of Contours found : " + str(len(cnts)))
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
                
            epsilon = 0.01 * peri
            approx = cv2.approxPolyDP(c, epsilon, True)
                
            if len(approx) == 4:  
                print(approx)
                NumberPlateCnt = approx  
                break
        # After finding the contour
        if NumberPlateCnt is not None:
            center1 = (NumberPlateCnt[0][0][0], NumberPlateCnt[0][0][1])
            center2 = (NumberPlateCnt[1][0][0], NumberPlateCnt[1][0][1])
            center3 = (NumberPlateCnt[2][0][0], NumberPlateCnt[2][0][1])
            center4 = (NumberPlateCnt[3][0][0], NumberPlateCnt[3][0][1])
        
            approx2 = corners(NumberPlateCnt)
            result = transform(model_cropped, approx2)
            ct = 1
        else:
            print("No contour with four corners found.retrying...")
            dt=dt+1
            if er <= 1:
                if dt == 3:
                    er = er+1
                    dt = 0
            else:
                if dt == 6:
                    er = er+1
                    dt = 0
            
            if er == 5 :
                print("No contour with four corners found.")
                ct = 1
                
        counter = counter + 1
        if counter == 30:
            print("Out of rounds")
            break
    #*********************************************************************************
    
    return result_image,approx2
        
def crop_from_model(image):
    print("cropping from model...")

    # Inference
    results = model(image)
    frame = results.pandas().xyxy[0]
    results.pandas().xyxy[0]

    xmin = int(frame.iloc[0, 0])
    ymin = int(frame.iloc[0, 1])
    xmax = int(frame.iloc[0, 2])
    ymax = int(frame.iloc[0, 3])

    # Crop the image
    cropped_image = image[ymin:ymax, xmin:xmax]
    
    return cropped_image

def start_ocr(image,process):
    print("Starting")
    model_cropped = crop_from_model(image)
    result_image,approx2 = crop_from_contour(model_cropped)
    if(approx2 is not None):
        txts = ocr(result_image)
        txts = sorted(txts, key=len)
        current_time = time.time()
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        # clock = time.strftime("%H:%M:%S", time.localtime(current_time))
        txts.append(date)
        # txts.append(clock)
        if(process == "in"):
            txts.append("in")
        elif(process == "out"):
            txts.append("out")
        print(txts)
        
        return txts
    else:
        return "Cannot Contour"
    