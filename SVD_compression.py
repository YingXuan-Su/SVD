import numpy as np
import cv2
import math
import pandas as pd
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

def main():

    imgs = [r'haha.jpg', r'Anya.webp', r'Wolf_Children.jpg', r'Marionette.jpg', r'night_sky.jpg']

    img = cv2.imread(imgs[0])
    M, N, C = img.shape
    # ranks = [np.linalg.matrix_rank(img[:,:,i]) for i in range(3)]
    # print("rank = ", ranks)

    mse_list = []
    psnr_list = []
    ssim_list = []
    CR_list = []
    #Max_k = min(img.shape[0], img.shape[1])
    #interval = math.ceil(Max_k/10)
    ks = [1, 2, 3, 5, 10, 50, 100, 300, 500, 1350]  #haha
    #ks = [1, 3, 5, 10, 50, 100, 200, 300, 400, 500]  #Anya #range(1, Max_k, interval)

    fig, ax = plt.subplots(2, 5, figsize=(12,6))
    for i in range(len(ks)):
        result_img = SVD(img, ks[i])
        #cv2.imwrite(f"result/Anya_k={ks[i]}.png", result_img)
        ax[int(i/5), i%5].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        ax[int(i/5), i%5].axis('off')
        ax[int(i/5), i%5].set_title(f"k={ks[i]}")
        mse_list.append(mean_squared_error(img, result_img))
        psnr_list.append(peak_signal_noise_ratio(img, result_img))
        ssim_list.append(structural_similarity(img, result_img, multichannel=True))
        CR_list.append( (M*N) / (ks[i]*(1+M+N) ))
    fig.tight_layout()
    plt.savefig("result/haha_result.png")
    #print(mse_list)

    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax[0,0].set_xticks(ks)
    ax[0,0].grid(color='black', ls = '-', lw = 0.25)
    ax[0,0].plot( ks, mse_list, c='r')
    ax[0,0].set_title('MSE')

    ax[0,1].set_xticks(ks)
    ax[0,1].grid(color='black', ls = '-', lw = 0.25)
    ax[0,1].plot( ks, psnr_list, c='g')
    ax[0,1].tick_params(axis='both', which='major')
    ax[0,1].set_title('PSNR')

    ax[1,0].set_xticks(ks)
    ax[1,0].grid(color='black', ls = '-', lw = 0.25)
    ax[1,0].plot( ks, ssim_list, c='b')
    ax[1,0].tick_params(axis='both', which='major')
    ax[1,0].set_title('SSIM')

    ax[1,1].set_xticks(ks)
    ax[1,1].grid(color='black', ls = '-', lw = 0.25)
    ax[1,1].plot( ks, CR_list, c='b')
    ax[1,1].tick_params(axis='both', which='major')
    ax[1,1].set_title('CR')

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig("result/haha_performance.png")

    df = pd.DataFrame({'k':ks, 'MSE': mse_list, 'PSNR': psnr_list, 'SSIM': ssim_list, 'CR': CR_list})
    df = df.round(3)
    df.to_csv('performance.csv', index=None)
    

def SVD(img, k):
    compresed_img = np.zeros_like(img)
    for ch in range(3):
        U, Sigma, V = np.linalg.svd(img[:,:,ch])
        compresed_img[:,:,ch] = U[:,:k].dot(np.diag(Sigma[:k])).dot(V[:k,:])
    return compresed_img

if __name__ == '__main__' :
    
    main()

