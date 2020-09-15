import numpy as np
import os
import cv2
import time
from noise import  *

if __name__ == '__main__':

    path= 'D:/Documents/Universidad/Lala/Hola/Lenna.png'
    image = cv2.imread(path)
    tipo_ruido = input("Inserte el tipo de ruido que desea (gauss o s&p)")
    ruid = noise()
    ruid.noisy(tipo_ruido, image)

    if tipo_ruido == "gauss":

        gauss_noised = cv2.imread("NGauss.jpg")  # Leemos la imagen guardada con el ruido de Gauss
        #Gauss

       # tiempo_inicial = time.clock()
        image_gauss_f = cv2.GaussianBlur(gauss_noised, (7, 7), 1.5)
       # tiempo_transcurrido = time.clock() - tiempo_inicial
       # print("Tiempo transcurrido Gauss: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro Gauss', image_gauss_f)
        cv2.imwrite('FGauss_g.jpg', image_gauss_f)
        cv2.waitKey(0)

        #image_noise1 = abs(gauss_noised-image_gauss_f)
        #print("La estimación del ruido para el filtro Gauss con ruido Gauss es:",image_noise1)

        #Mediana
       # tiempo_inicial = time.clock()
        image_median = cv2.medianBlur(gauss_noised, 7)
       # tiempo_transcurrido = time.clock() - tiempo_inicial
       # print("Tiempo transcurrido Mediana: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro mediana', image_median)
        cv2.imwrite('FMediana_g.jpg', image_median)
        cv2.waitKey(0)

       # image_noise2 = abs(gauss_noised-image_median)
        #print("La estimación del ruido para el filtro de Mediana con ruido Gauss es:",image_noise2)

        #Bilateral
        #tiempo_inicial = time.clock()
        image_bilateral = cv2.bilateralFilter(gauss_noised, 15, 25, 25) #cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
        #tiempo_transcurrido = time.clock() - tiempo_inicial
        #print("Tiempo transcurrido Bilateral: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro bilateral', image_bilateral)
        cv2.imwrite('FBi_g.jpg', image_bilateral)
        cv2.waitKey(0)

      #  image_noise3 = abs(gauss_noised-image_bilateral)
      #  print("La estimación del ruido para el filtro bilateral con ruido Gauss es:",image_noise3)

        #Filtro nlm
      #  tiempo_inicial = time.clock()
        image_nlm = cv2.fastNlMeansDenoisingColored(gauss_noised, 5, 15, 25) # cv2.fastNlMeansDenoisingColored((src, hcolor, templateWindowsSize, searchWindowsSize)
      #  tiempo_transcurrido = time.clock() - tiempo_inicial
      #  print("Tiempo transcurrido nlm: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro nlm', image_nlm)
        cv2.imwrite('Fnlm_g.jpg', image_nlm)
        cv2.waitKey(0)

     #   image_noise4 = abs(gauss_noised-image_nlm)
     #   print("La estimación del ruido para el filtro nlm con ruido Gauss es:",image_noise4)

        '''
        #gauss
        mse1_1 = (np.square(image_gauss_f - image_median)).mean()
        print("El mse entre filtro Gauss y filtro mediana es:", mse1_1)
 
        mse1_2 = (np.square(image_gauss_f - image_bilateral)).mean()
        print("El mse entre filtro Gauss y filtro bilateral es:", mse1_2)

        mse1_3 = (np.square(image_gauss_f - image_nlm)).mean()
        print("El mse entre filtro Gauss y filtro nlm es:", mse1_1)

        #mediana
        mse2_1 = (np.square(image_median - image_gauss_f)).mean()
        print("El mse entre filtro mediana y filtro Gauss es:", mse2_1)

        mse2_2 = (np.square(image_median - image_bilateral)).mean()
        print("El mse entre filtro mediana y filtro bilateral es:", mse2_2)

        mse2_3 = (np.square(image_median - image_nlm)).mean()
        print("El mse entre filtro mediana y filtro nml es:", mse2_1)


        #bilateral
        mse3_1 = (np.square(image_bilateral - image_gauss_f)).mean()
        print("El mse entre filtro bilateral y filtro Gauss es:", mse3_1)

        mse3_2 = (np.square(image_bilateral - image_median)).mean()
        print("El mse entre filtro bilateral y filtro mediana es:", mse3_2)

        
        mse3_3 = (np.square(image_bilateral - image_nlm)).mean()
        print("El mse entre filtro bilateral y filtro nml es:", mse3_1)

        #nml
        mse4_1 = (np.square(image_nlm - image_gauss_f)).mean()
        print("El mse entre filtro nml y filtro Gauss es:", mse4_1)

        mse4_2 = (np.square(image_nlm - image_bilateral)).mean()
        print("El mse entre filtro nml y filtro bilateral es:", mse4_2)

        mse4_3 = (np.square(image_nlm - image_median)).mean()
        print("El mse entre filtro nml y filtro mediana es:", mse4_1)
        '''

    elif tipo_ruido == "s&p":

        sp_noised = cv2.imread("NSP.jpg")  # Leemos la imagen guardada con el ruido de Gauss

        # Gauss
        tiempo_inicial = time.clock()
        image_gauss_f = cv2.GaussianBlur(sp_noised, (7, 7), 1.5)
        tiempo_transcurrido = time.clock() - tiempo_inicial
        print("Tiempo transcurrido Gauss: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro Gauss', image_gauss_f)
        cv2.imwrite('FGauss_sp.jpg', image_gauss_f)
        cv2.waitKey(0)

        #image_noise1 = abs(sp_noised-image_gauss_f)
        #print("La estimación del ruido para el filtro Gauss con ruido s&p es:",image_noise1)

        # Mediana
        tiempo_inicial = time.clock()
        image_median = cv2.medianBlur(sp_noised, 7)
        tiempo_transcurrido = time.clock() - tiempo_inicial
        print("Tiempo transcurrido mediana: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro mediana', image_median)
        cv2.imwrite('FMediana_sp.jpg', image_median)
        cv2.waitKey(0)

        #image_noise2 = abs(sp_noised-image_median)
        #print("La estimación del ruido para el filtro mediano con ruido s&p es:",image_noise2)

        # Bilateral
        tiempo_inicial = time.clock()
        image_bilateral = cv2.bilateralFilter(sp_noised, 15, 25, 25)  # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
        tiempo_transcurrido = time.clock() - tiempo_inicial
        print("Tiempo transcurrido bilatera: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro bilateral', image_bilateral)
        cv2.imwrite('FBi__sp.jpg', image_bilateral)
        cv2.waitKey(0)

        #image_noise3 = abs(sp_noised-image_bilateral)
        #print("La estimación del ruido para el filtro bilateral con ruido s&p es:",image_noise3)

        # Filtro nlm
        tiempo_inicial = time.clock()
        image_nlm = cv2.fastNlMeansDenoisingColored(sp_noised, 5, 15, 25)  # cv2.fastNlMeansDenoisingColored((src, hcolor, templateWindowsSize, searchWindowsSize)
        tiempo_transcurrido = time.clock() - tiempo_inicial
        print("Tiempo transcurrido nlm: %0.10f segundos." % tiempo_transcurrido)
        cv2.imshow('Filtro nlm', image_nlm)
        cv2.imwrite('Fnlm_sp.jpg', image_nlm)
        cv2.waitKey(0)

        '''
        image_noise4 = abs(sp_noised-image_nlm)
        print("La estimación del ruido para el filtro nlm con ruido s&p es:",image_noise4)

        #gauss
        mse1_1 = (np.square(image_gauss_f - image_median)).mean()
        print("El mse entre filtro Gauss y filtro mediana es:", mse1_1)

        mse1_2 = (np.square(image_gauss_f - image_bilateral)).mean()
        print("El mse entre filtro Gauss y filtro bilateral es:", mse1_2)

        mse1_3 = (np.square(image_gauss_f - image_nlm)).mean()
        print("El mse entre filtro Gauss y filtro nlm es:", mse1_1)

        #mediana
        mse2_1 = (np.square(image_median - image_gauss_f)).mean()
        print("El mse entre filtro mediana y filtro Gauss es:", mse2_1)

        mse2_2 = (np.square(image_median - image_bilateral)).mean()
        print("El mse entre filtro mediana y filtro bilateral es:", mse2_2)

        mse2_3 = (np.square(image_median - image_nlm)).mean()
        print("El mse entre filtro mediana y filtro nml es:", mse2_1)


        #bilateral
        mse3_1 = (np.square(image_bilateral - image_gauss_f)).mean()
        print("El mse entre filtro bilateral y filtro Gauss es:", mse3_1)

        mse3_2 = (np.square(image_bilateral - image_median)).mean()
        print("El mse entre filtro bilateral y filtro mediana es:", mse3_2)

        mse3_3 = (np.square(image_bilateral - image_nlm)).mean()
        print("El mse entre filtro bilateral y filtro nml es:", mse3_1)

        #nml
        mse4_1 = (np.square(image_nlm - image_gauss_f)).mean()
        print("El mse entre filtro nml y filtro Gauss es:", mse4_1)

        mse4_2 = (np.square(image_nlm - image_bilateral)).mean()
        print("El mse entre filtro nml y filtro bilateral es:", mse4_2)

        mse4_3 = (np.square(image_nlm - image_median)).mean()
        print("El mse entre filtro nml y filtro mediana es:", mse4_1)
        '''