from typing import Optional
from fastapi import FastAPI, File, UploadFile
from ocr import start_ocr
import cv2
import numpy as np
import datetime
from datetime import timedelta

app = FastAPI()

car_in = []
car_out = []
car_list = []

@app.get("/cars_in")
def read_root():
    return {"Cars In": car_in}

@app.get("/cars_out")
def read_root():
    return {"Cars Out": car_out}

@app.get("/cars_list")
def read_root():
    return {"Cars list": car_list}

@app.post("/ocr_in/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    ocr_text = start_ocr(image,"in")
    if ocr_text != "Cannot Contour":
        car_in.append(ocr_text)
        car_list.append(ocr_text)
    
    return {"Car In": ocr_text}

@app.post("/ocr_out/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    ocr_text = start_ocr(image, "out")
    results = "This car never got in"
    result = []
    parking_time_str = "N/A"

    if ocr_text != "Cannot Contour":
        car_out.append(ocr_text)
        
        indices_to_remove = []

        for i, car_out_entry in enumerate(car_out):
            for j, car_in_entry in enumerate(car_list):
                if car_out_entry[0] == car_in_entry[0]:
                    print("Comparing ", car_out_entry[0], " and ", car_in_entry[0])
                    time1 = datetime.datetime.strptime(car_out_entry[2], "%Y-%m-%d %H:%M:%S")
                    time2 = datetime.datetime.strptime(car_in_entry[2], "%Y-%m-%d %H:%M:%S")
                    parking_time = time1 - time2
                        
                    parking_time_str = str(parking_time)
                    indices_to_remove.append(j)
                    result = ocr_text

        # Remove entries from car_list in reverse order to avoid index issues
        for index in reversed(sorted(set(indices_to_remove))):
            if 0 <= index < len(car_list):
                del car_list[index]
    if len(result) > 0:
        results = result[0] + " " + result[1] + " is out. Parking for " + parking_time_str + " Hours"    
    
    return {"text": results}
