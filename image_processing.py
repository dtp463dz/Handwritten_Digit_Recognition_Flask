import cv2
import numpy as np

def process_img(image_path):
    # Read img
    img = cv2.imread(image_path)

    # Chuyển sang gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # làm mịn
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Tìm ngưỡng động dựa trên giá trị trung bình của ảnh
    _, bin_img = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY_INV)

    # Tìm các contours
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # loc contours
    area_threshold = 50
    aspect_ratio_threshold = 0.2
    filtered_contours = []

    for i, cnt in enumerate(contours):
        if(hierarchy[0][i][3] == -1): # chi lay contour ngoai
            area = cv2.contourArea(cnt)
            x , y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if(area > area_threshold and aspect_ratio > aspect_ratio_threshold):
                filtered_contours.append(cnt)

    # sap xep contours
    filtered_contours = sorted(filtered_contours, key=lambda ctr:cv2.boundingRect(ctr)[0] )

    digits = []
    rois = []
    areas = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Tạo vùng đệm xung quanh chữ số
        m = 0.2  
        roi = bin_img[max(0, y - int(m * h)):y + h + int(m * h), max(0, x - int(m * w)):x + w + int(m * w)]
        # Cho roikích thước 28x28
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        digits.append(roi)
        rois.append((x, y, w, h))
        areas.append(area)

    return digits, rois, areas 
