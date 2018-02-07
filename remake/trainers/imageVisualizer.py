import tensorflow as tf
import cv2

def show_image_with_crosshairs(imageTensor,tfPred):
    imageDir = imageTensor[2][0].decode('ascii')
    imageToShow = cv2.imread(imageDir)
    predicted_center = tfPred[0]
    #draw crosshair
    cv2.line(imageToShow,(int(predicted_center[1])-6,int(predicted_center[0])-6),(int(predicted_center[1])+6,int(predicted_center[0])+6),(0,0,255),2)
    cv2.line(imageToShow,(int(predicted_center[1])-6,int(predicted_center[0])+6),(int(predicted_center[1])+6,int(predicted_center[0])-6),(0,0,255),2)
    #draw box
    cv2.line(imageToShow,(int(predicted_center[1])-14,int(predicted_center[0])-14),(int(predicted_center[1])+14,int(predicted_center[0])-14),(255,255,255),1)
    cv2.line(imageToShow,(int(predicted_center[1])-14,int(predicted_center[0])-14),(int(predicted_center[1])-14,int(predicted_center[0])+14),(255,255,255),1)
    cv2.line(imageToShow,(int(predicted_center[1])+14,int(predicted_center[0])+14),(int(predicted_center[1])+14,int(predicted_center[0])-14),(255,255,255),1)
    cv2.line(imageToShow,(int(predicted_center[1])+14,int(predicted_center[0])+14),(int(predicted_center[1])-14,int(predicted_center[0])+14),(255,255,255),1)
    cv2.imshow("Current Image",imageToShow)
    if abs(imageTensor[1][0][0]-predicted_center[0])>10 or abs(imageTensor[1][0][1]-predicted_center[1])>10:
        cv2.waitKey(33)
    else:
        cv2.waitKey(0)
    return
