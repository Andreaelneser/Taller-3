import numpy as np
import cv2

class noise:

    def noisy (self, noise_typ,image):

        if noise_typ == "gauss":

            row,col,ch= image.shape
            mean = 0
            var = 0.001
            sigma = var**0.002
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss

            noise_gauss = noisy.astype(np.uint8)

            cv2.imshow('Ruido Gauss', noise_gauss)
            cv2.imshow('Solo ruido Gauss', gauss)
            cv2.imwrite('NGauss.jpg', noise_gauss)
            cv2.waitKey(0)
            return noisy

        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.03
            out = np.copy(image)

            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0

            cv2.imshow('Ruido S&P', out)
            cv2.imwrite('NSP.jpg', out)
            cv2.waitKey(0)
            return out