import cv2 as cv
#Armazena a imagem em uma variavel
img=cv.imread('woman.jpg')
#Inicio a imagem em uma escala de cinza
cinza=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#Mostro a imagem em uma janela passando o parametro cinza
cv.imshow("Pessoa",cinza)


#imagem com blur

imagem_borrada=cv.GaussianBlur(cinza, (5,5), cv.BORDER_DEFAULT)
cv.imshow("Borrado",imagem_borrada)

#contornos da imagem
contorno=cv.Canny(imagem_borrada,125,175)
cv.imshow("Contornos",contorno)



#Quantidade de contornos sem blur
contornos,hierarquia= cv.findContours(contorno,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(f'{len(contornos)} , contornos achados')

haar_cascade=cv.CascadeClassifier('haar_face.xml')

faces_react = haar_cascade.detectMultiScale(cinza,scaleFactor=1.1,minNeighbors=3)

for (x,y,w,h) in faces_react:
  cv.rectangle(img, (x,y) , (x+w,y+h) , (0,255,0) , thickness=3 )

cv.imshow("Rostos",img)  

cv.waitKey(0)
