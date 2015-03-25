#encoding: utf8
import numpy as np
import cv2
import cv2.cv as cv
from skimage import morphology as morp



#Funcion que añade la cabeza y la mirada
def headAndEyes(frame):
    #Transforma el frame de RGB a Escala de grises
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detector de parte superior del cuerpo
    head = upperDetector.detectMultiScale(frameGray, scaleFactor=1.3, minNeighbors=4, minSize=(160, 140))
    #Para cada parte superior detectada    
    for ux,uy,uw,uh in head:
        #Redefinicion del tamaño para la cabeza
        hx = ux+45
        hy = uy+10
        hw = uw-90
        hh = uh-60
        #Se dibuja la deteccion de la cabeza
        cv2.rectangle(frame, (hx,hy), (hx+hw,hy+hh), (0,255,0))
        #Se extrae la region de la cabeza
        roiHead = frameGray[hy:hy+hh , hx:hx+hw]
        roiHeadCol = frame[hy:hy+hh , hx:hx+hw]
        #Detector de ojos en la region de cabeza extraida    
        eyes = eyesDetector.detectMultiScale(roiHead, minNeighbors=4)
        #Por cada par de ojos detectados    
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roiHeadCol, (ex,ey), (ex+ew,ey+eh), (0,255,0))
            #Se extrae la region del par de ojos
            roiEyes = roiHead[ey:ey+eh, ex:ex+ew-5]
            #Ecualizacion de histogramas para realzado del brillo y contraste de los ojos
            #roiEyes = cv2.equalizeHist(roiEyes)
            #cv2.imshow("eyes", roiEyes)
            #roiEyes = cv2.adaptiveThreshold(roiEyes, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3,8)
            #Binarizacion de la region de ojos para extraer la pupila
            #_,pupils = cv2.threshold(roiEyes, 1, 255, cv2.THRESH_BINARY_INV)
            #Dilatacion para agrandar las pupilas
            #pupils = cv2.dilate(pupils, kernel, iterations = 2)




#Abre el archivo de video
cap = cv2.VideoCapture('expo2.avi')
#Detector de Parte superior del cuerpo
upperDetector = cv2.CascadeClassifier("C:\OpenCV2.4.11\sources\data\haarcascades\haarcascade_mcs_upperbody.xml")
#Detector de ojos
eyesDetector = cv2.CascadeClassifier("C:\OpenCV2.4.11\sources\data\haarcascades\haarcascade_mcs_eyepair_small.xml")

#Kernel para aplicaciones morfologicas
mat = [[0.8,1,0.8],[1,1,1],[0.8,1,0.8]]
kernel = np.asarray(mat, np.uint8)
#kernel = np.ones((3,3),np.uint8)

#Mientras el archivo de video esté abierto 
while(cap.isOpened()):
    #Obtiene el siguiente frame del video
    _,frame = cap.read()
    #Si llega al final entonces no habra frame y saldrá del ciclo
    if(frame is None):
        #Se espera cualquier tecla para salir
        cv2.waitKey(0)
        break
    else:
        #Añade cabeza y ojos
        headAndEyes(frame)
     
        #Muestra el frame
        cv2.imshow("Video",frame)

        #Se espera la tecla Esc para salir
        c = cv2.waitKey(33)
        if(c == 27):
            break
        
print("Finalizado...")
#Libera la memoria usada por el archivo de video
cap.release()
#Destruye las ventanas
cv2.destroyAllWindows()






    
