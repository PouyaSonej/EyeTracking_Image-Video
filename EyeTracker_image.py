import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

LEFT_EYE =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE =[7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]

detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

face_img = cv.imread("images/face.jpg")
face_img2 = face_img.copy()

face_img, bbox = detector.findFaces(face_img)
face_img, faces = meshdetector.findFaceMesh(face_img)

# cv.imwrite('output.jpg', face_img)

# cv.imshow('eye_roi', face_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

if bbox:
    center = bbox[0]["center"]   
    if faces:

        #left
        left_eye_points = np.array([[faces[0][p][0],faces[0][p][1]] for p in LEFT_EYE])
        #cv.fillPoly(face_img2, pts=[left_eye_points], color = 255)
        (ex,ey,ew,eh) = cv.boundingRect(left_eye_points)
        #cv.rectangle(face_img2, (ex,ey), (ex+ew,ey+eh), (255,255,255))
        eye_roi = face_img2[ey:ey+eh, ex:ex+ew]
        eye_roi_gr = cv.cvtColor(eye_roi, cv.COLOR_BGR2GRAY)
        _, iris = cv.threshold(eye_roi_gr, 40, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(iris, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

        if contours:
            (ix,iy,iw,ih) = cv.boundingRect(contours[0])
            ix_cntr, iy_centr = ix+int(iw/2)-1 + ex, iy+int(ih/2)+ey+1
            cv.circle(face_img2, (ix_cntr, iy_centr), 6, (0,0,255), -1)

            ix_cntr_e, iy_centr_e = ix+int(iw/2), iy+int(ih/2)
            if ix_cntr_e > int(ew/2):
                print("right")
            elif ix_cntr_e < int(ew/2):
                print("left")

        #right
        right_eye_points = np.array([[faces[0][p][0],faces[0][p][1]] for p in RIGHT_EYE])
        #cv.fillPoly(face_img2, pts=[left_eye_points], color = 255)
        (ex,ey,ew,eh) = cv.boundingRect(right_eye_points)
        #cv.rectangle(face_img2, (ex,ey), (ex+ew,ey+eh), (255,255,255))
        eye_roi = face_img2[ey:ey+eh, ex:ex+ew]
        eye_roi_gr = cv.cvtColor(eye_roi, cv.COLOR_BGR2GRAY)
        _, iris = cv.threshold(eye_roi_gr, 40, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(iris, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

        if contours:
            (ix,iy,iw,ih) = cv.boundingRect(contours[0])
            ix_cntr, iy_centr = ix+int(iw/2)-2 + ex, iy+int(ih/2)+ey+1
            cv.circle(face_img2, (ix_cntr, iy_centr), 6, (0,0,255), -1)

            ix_cntr_e, iy_centr_e = ix+int(iw/2), iy+int(ih/2)
            if ix_cntr_e > int(ew/2):
                print("right")
            elif ix_cntr_e < int(ew/2):
                print("left")



# face_img2, bbox = detector.findFaces(face_img2)
# cv.imwrite('output2.jpg', face_img2)

cv.imshow('eye_roi', face_img2)
cv.waitKey(0)
cv.destroyAllWindows()
