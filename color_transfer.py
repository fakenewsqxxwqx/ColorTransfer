"""
    2022.1.5
    Author: zcj
    https://blog.csdn.net/Mzcc_bbms/article/details/122341942#:~:text=%22%22%22%202022.1.5%20Author%3A%20zcj%20%E8%89%B2%E5%BD%A9%E8%BF%81%E7%A7%BB%E7%AE%97%E6%B3%95%EF%BC%9A%E3%80%8AColor%20Transfer%20between%20Images%E3%80%8B,and%20standard%20deviations%20of%20the%20L%2Aa%2Ab%2A%20color%20space.
    色彩迁移算法：《Color Transfer between Images》
    把参考图像 ref 的色彩迁移到图像 src
    原理：在 lab 色彩空间下，调整两幅图像的每个通道，使得均值和方差差不多
    公式：dst = (src - src_mean) * (src_std / ref_std) + ref_mean + 0.5
    推导：Y = (X - M) * a + b; -> Y = aX + (-aM + b)
"""

import cv2
import numpy as np

def colorTransferCpu(src, ref, test=False):
    """
    	Transfers the color distribution from the ref to the src
    	image using the mean and standard deviations of the L*a*b*
    	color space.

    	This implementation is (loosely) based on to the "Color Transfer
    	between Images" paper by Reinhard et al., 2001.

    	Parameters:
    	-------
    	src: NumPy array
    		OpenCV image in BGR color space (the src image)
    	ref: NumPy array
    		OpenCV image in BGR color space (the ref image)

    	Returns:
    	-------
    	transfer: NumPy array
    		OpenCV image (h, w, 3) NumPy array (uint8)
    	"""
    # 1. BRG -> lab
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)

    # 2. compute mean and std
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    src_mean, src_std = cv2.meanStdDev(src_lab)
    
    # 3. split src channels
    src_split = cv2.split(src_lab)
    src_split = np.float32(src_split)
    for i in range(3):
        if(test): a = (ref_std[i] / src_std[i] )
        else: a = (src_std[i] / ref_std[i])
        src_split[i] = src_split[i] * a - src_mean[i] * a + ref_mean[i] + 0.5

    src_split = np.clip(src_split, 0, 255) # clip the pixel intensities to [0, 255] if they fall outside this range

    # 4. merge channels
    src_transfer = cv2.merge(src_split)
    src_transfer = cv2.cvtColor(src_transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    return src_transfer


if __name__ == "__main__":

    src = cv2.imread("images/src.jpg")
    ref = cv2.imread("images/ref.jpg")
    src_transfer1 = colorTransferCpu(src, ref, test=False)
    src_transfer2 = colorTransferCpu(src, ref, test=True)
    #src_transfer = color_transfer.color_transfer(ref, src)

    src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)
    ref = cv2.resize(ref, (0, 0), fx=0.5, fy=0.5)
    src_transfer1 = cv2.resize(src_transfer1, (0, 0), fx=0.5, fy=0.5)
    src_transfer2 = cv2.resize(src_transfer2, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("src", src)
    cv2.imshow("ref", ref)
    cv2.imshow("src_transfer", src_transfer1)
    cv2.imshow("src_transfer2", src_transfer2)

    #保存图片
    cv2.imwrite("images/src_transfer1.jpg", src_transfer1)
    cv2.imwrite("images/src_transfer2.jpg", src_transfer2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
