from tkinter import *
import cv2
import cv2 as cv
import numpy as np
import os
import matplotlib.image as mpimg
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import time
from skimage.transform import resize 
from skimage.feature import hog

window = tk.Tk()

window.title("Xử lý ảnh")

window.geometry("1200x800+400+100")
window.config(background='Light blue')
lbl=Label(window,text="Chương trình xử lý ảnh", font='arial 35 bold',foreground='red',background='Light blue')
lbl.pack()

# Đường dẫn đến thư mục chứa các tệp ảnh
img_dir = "Image"


# Lấy danh sách các tệp ảnh
img_list = os.listdir(img_dir)
running = True

# Vòng lặp để đọc và hiển thị các ảnh
while running:
    for img_name in img_list:
        # Đường dẫn đầy đủ đến tệp ảnh
        img_path = os.path.join(img_dir, img_name)        
        # Đọc ảnh và hiển thị
        img = cv2.imread(img_path)
        cv2.imshow("Image", img)
                # Đợi 1 giây trước khi chuyển đến ảnh tiếp theo
        cv2.waitKey(1000)
        # Kiểm tra nếu người dùng đóng cửa sổ hiển thị ảnh
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            running = False
            break 
        def click():            
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.subplot(1,3,1)
            plt.title("Original ")
            plt.imshow(image)
            print('\n')
            brighness =10
            contrast =2.3
            image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brighness)
        
            plt.subplot(1,3,2)
            plt.title("Brightness = 10 & contrast = 2.3")
            plt.imshow(image2)

            brighness = 5
            contrast =1.5
            image3 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brighness)

            plt.subplot(1,3,3)
            plt.title("Brightness = 5 & contrast = 1.5")
            plt.imshow(image3) 
            plt.show()
        def click1():
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            image[:,:,0]= image [:,:,0]*0.7

            image[:,:,1]= image [:,:,1]*1.5

            image[:,:,2]= image [:,:,2]*0.5
            #Conver the image back to BGR color space
            image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            cv2.imshow("Tang cuong mau sac",image2)  
            plt.show()
        def click2():
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            sharpened_image = cv2.filter2D(img, -1, kernel)
            sharpened_image2= cv2.Laplacian(img, cv2.CV_64F)
            cv2.imshow("Do sat net",sharpened_image2)
            plt.show()
        def click3():
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Anh xam',img_gray)
            plt.show()
        def click4():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 100, 200)          
            cv2.imshow('Anh trich bien', edges)
            plt.show()
        def click5():
            img1=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            twoDimage=img.reshape((-1,3))
            twoDimage=np.float32(twoDimage)
            print(twoDimage)
            criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,10,1.0)
            K=4
            attempts=10
            ret,label,center=cv.kmeans(twoDimage,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
            center=np.uint8(center)
            res=center[label.flatten()]
            result_image=res.reshape((img.shape))

            plt.axis('off')
            plt.imshow(result_image)
            plt.show()
        def click6():
            img1 = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            _,thresh=cv.threshold(gray,np.mean(gray),255,cv.THRESH_BINARY_INV)
            plt.imshow(thresh)

            edges = cv.dilate(cv.Canny(thresh,0,255),None)
            plt.axis('off')
            plt.imshow(edges)

            cnt = sorted(cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2],key=cv.contourArea)[-1]
            mask = np.zeros((500,1000),np.uint8)
            masked = cv.drawContours(mask,[cnt],-1,255,-1)
            plt.axis('off')
            plt.imshow(masked)
            plt.show() 
        def click7():
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            resize_image = cv2.resize(image,(512,512))
            plt.imshow(resize_image)
            plt.show()
        def click8():
            flatImg = np.reshape(img, [-1, 3])
            bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(flatImg)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            print("Số lượng cụm ước tính: %d" % n_clusters_)
            segmentedImg = cluster_centers[np.reshape(labels, img.shape[:2])]
            plt.imshow(cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()
        def click9():
            
            resized_img = resize(img,(64*4,64*4))
            plt.axis("off")
            plt.imshow(resized_img)
            print(resized_img.shape)

            fd, hog_image =hog(resized_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=True, multichannel=True)
            plt.axis("off")
            plt.imshow(hog_image,cmap="gray")
            plt.show()
lbl=Label(window,text="Chọn một chức năng để chạy", font='arial 25 bold',foreground='red',background='Light blue')
lbl.pack()
    
frame1 = tk.Frame(window,background='Light blue')
frame1.pack(fill=tk.X, padx=5, pady=50)

frame2 = tk.Frame(window,background='Light blue')
frame2.pack(fill=tk.X, padx=5, pady=5)

# Tạo button cho mỗi frame
button1 = tk.Button(frame1, text="Độ sáng", width=20,font='arial 15 bold',command=click)
button1.pack(side=tk.LEFT, padx=50, pady=10)

button2 = tk.Button(frame1, text="Tăng cường màu sắc ",font='arial 15 bold',width=20,command=click1)
button2.pack(side=tk.LEFT, padx=50, pady=10)

button4 = tk.Button(frame1, text="Sắc nét",font='arial 15 bold', width=20,command=click2)
button4.pack(side=tk.LEFT, padx=50, pady=10)

button5 = tk.Button(frame1, text="Ảnh xám",font='arial 15 bold', width=20,command=click3)
button5.pack(side=tk.LEFT, padx=50, pady=10)

button6 = tk.Button(frame1, text="Trích biên",font='arial 15 bold', width=20,command=click4)
button6.pack(side=tk.LEFT, padx=50, pady=10)

button7 = tk.Button(frame2, text="Phân vùng ảnh Kmeans",font='arial 15 bold', width=20,command=click5)
button7.pack(side=tk.LEFT, padx=50, pady=10)

button8 = tk.Button(frame2, text="Phát hiện đường viền",font='arial 15 bold', width=20,command=click6)
button8.pack(side=tk.LEFT, padx=50, pady=10)

button9 = tk.Button(frame2, text="Đổi kích thước 512 x512",font='arial 15 bold', width=20,command=click7)
button9.pack(side=tk.LEFT, padx=50, pady=10)

button9 = tk.Button(frame2, text="Meanshift",font='arial 15 bold', width=20,command=click8)
button9.pack(side=tk.LEFT, padx=50, pady=10)

button10 = tk.Button(frame2, text="Rút trích",font='arial 15 bold', width=20,command=click9)
button10.pack(side=tk.LEFT, padx=50, pady=10)


window.mainloop()
# Giải phóng bộ nhớ và đóng cửa sổ hiển thị ảnh
cv2.destroyAllWindows()
