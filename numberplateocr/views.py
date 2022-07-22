from unittest import result
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import cv2
from rest_framework import status, permissions
import json
import numpy as np
import io
import base64
from imageio import imread
import imutils
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
import pytesseract
# Create your views here.
def index(request):
    return HttpResponse("<h1>Beeotch Biatch</h1>")


def test(img):
    img = imutils.resize(img, width=300)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    smooth_img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
    # ret,smooth_img_gray = cv2.threshold(np.array(smooth_img_gray), 125, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(smooth_img_gray, 30, 200)
    contours, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img1 = img.copy()
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
    
    contours_selected = sorted(contours, key=cv2.contourArea, reverse=True) [:30]
    screenCnt = None

    img2 = img.copy()
    cv2.drawContours(img2, contours_selected, -1, (0, 255, 0), 3)
    i=7
    for cont in contours_selected:
        perimeter = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.018 * perimeter, True)
    
        # chooses the contours with four sides as this will probably be our number plate.
        if len(approx) == 4: 
            screenCnt = approx
            x,y,w,h = cv2.boundingRect(cont) 
            new_img=img[y:y+h,x:x+w]
            print(new_img)
            # cv2.imshow("original image", new_img)
            
            # Stores the new image of the cropped number plate.
            i+=1
            break

    print(f"{i}")

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
        return "No contour detected"
    else:
        detected = 1

    if(detected==1):
        plate = pytesseract.image_to_string(new_img, lang='eng')
        cleaned_plate = ''.join(c for c in plate if(c.isalnum()))
        print(f"{plate}")
        print(f"{cleaned_plate}")
    # if detected == 1:
    #     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    # mask = np.zeros(img_gray.shape,np.uint8)
    # new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    # new_image = cv2.bitwise_and(img,img,mask=mask)

    # (x, y) = np.where(mask == 255)
    # (topx, topy) = (np.min(x), np.min(y))
    # (bottomx, bottomy) = (np.max(x), np.max(y))
    # Cropped = img_gray[topx:bottomx+1, topy:bottomy+1]

    # text = pytesseract.image_to_string(Cropped, config='--psm 11')
    # text="MH31Z2384"
    return cleaned_plate


class ImageUploadAPI(APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = ()
    def post(self,request,format=None):
        print(request.data)
        image=request.data.get('image')
        print(image)
        image_b64 = base64.b64encode(image.read())
        img = imread(io.BytesIO(base64.b64decode(image_b64)))
        result=test(img)
        print(type(result))
        print(f"Result={result}")
        result_dict={"number":result}
        print(f"Result_dict={result_dict}")
        # result_dict["data"]=result
        return Response(data=result_dict,status=status.HTTP_200_OK)
        # return None





@csrf_exempt
def imgupload(request):
    if request.method=="POST":
        print(request)
        r1=json.loads(request.body)
        print(f"Request bidy= {r1}")
        image=r1.get("image")
        print(f"IMAGE={image}")
        image_b64 = base64.b64encode(image.read())
        img = imread(io.BytesIO(base64.b64decode(image_b64)))
        img = cv2.resize(img, (600,400) )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray = cv2.bilateralFilter(gray, 13, 15, 15) 

        edged = cv2.Canny(gray, 30, 200) 
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

        cn = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        screenCnt = None

        for c in contours:
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
            print ("No contour detected")
            return {"data":"Error,No contour detected"}
        else:
            detected = 1

        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        # text = pytesseract.image_to_string(Cropped, config='--psm 11')
        text="MH31Z2384"
        img = cv2.resize(img,(500,300))
        Cropped = cv2.resize(Cropped,(400,200))
        result_dict={}
        result_dict["data"]=text
        return JsonResponse(result_dict)

    else:
        return HttpResponse("<h1>Send a POST Request to this URL,not a GET request!</h1>")


