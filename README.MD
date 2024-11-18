# Manga Classifier 漫畫分類器


This is a C++ project that uses OpenCV to build a classifier for distinguishing between manga covers and real-life photographs. 
這是一個使用 OpenCV 開發的 C++ 專案，旨在建立一個分類器，用以區分漫畫封面與真實生活照片。


## Overview 概述


The project uses Histogram of Oriented Gradients (HOG) features and one of three classifiers to analyze images and determine whether they are manga covers or regular photographs. The three available classifiers are: Support Vector Machine (SVM), k-Nearest Neighbor (kNN), and Random Forest (RF) classifier.
此專案透過方向梯度直方圖（Histogram of Oriented Gradients，簡稱 HOG）特徵與支持向量機（SVM）分類器，最近鄰居分類器（kNN），隨機森林分類器（RF）來分析圖像，以判別它們是漫畫封面還是真實照片。


Note: Using a sample dataset of 8000 images, kNN performed around 56 percent accuracy, SVM performed around 52 percent accuracy, and RF performed around 89 percent accuracy, so using RF as your classifier is recommended.
補充：在此特定情況下，kNN表現約56%準確率，SVM表現約52%準確率，RF表現約89%準確率，所以建議用RF作為分類器






### Key Features 專案的主要功能


- **Data preparation 數據準備**: Splitting the dataset into training and testing sets, and extracting HOG features from the images. 
 將數據集拆分為訓練集和測試集，並從圖片中提取 HOG 特徵。
- **Model training 模型訓練**: Training SVM classifier, kNN classifier, and RF classifier to learn the differences between manga covers and real photographs. 
 訓練 SVM 分類器, kNN 分類器, RF 分類器，以學習漫畫封面與真實照片之間的差異。
- **Model evaluation 模型評估**: Testing the trained model on the test set and reporting the classification accuracy. 
 在測試集上測試訓練好的模型並報告分類準確率。
- **Model persistence 模型保存**: Saving the trained model to a file for future use. 
 將訓練好的模型保存到文件中，以便日後使用。


## Dependencies 依賴項


This project requires the following dependencies: 
此專案需要以下依賴項：


- OpenCV (version 4.10.0 or later) 
- CMake (version 3.10 or later) 
- C++ compiler with support for C++17 standard 
 支援 C++17 標準的 C++ 編譯器


- It is strongly recommended to re-write your own CMakeLists.txt because some of the context is specifically for arm64 but not X86.
強烈建議重新撰寫CMakeLists.txt，其中有些內容是針對arm64而非X86
## Building and Running 構建與執行


1. Clone the repository 
  克隆此存儲庫：


  ```bash
  git clone https://github.com/Gary0302/manga_classifier-Comic_or_Photo_images.git
  cd manga-classifier
  ```


2. Create a build directory and run CMake
  創建構建目錄並運行 CMake：
  ```bash
  mkdir build
  cd build
  cmake .. #cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64 for specifically arm64
  ```


3. Build the project
  編譯專案：
  ```bash
  make #for arm
  ```
  ```powershell
  cmake --build . #for x86
  ```




4. Run the classifier
  執行分類器：
  ```bash
  ./manga_classifier #for arm
  ```
  ```powershell
  your/drectory/manga_classifier.exe #for x86
  ```


The classifier will train on the provided dataset, test the model, and report the classification accuracy. The trained model will be saved as manga_classifier_{model you use}_model.yml.
分類器將會使用提供的數據集進行訓練，測試模型並報告分類準確率。訓練完成的模型將會保存為 manga_classifier_{model you use}_model.yml。


To swap between kNN, SVM, and RF model, un-comment and comment the specific main function
註解和取消註解需要的main函式以切換最近鄰居，支援向量機，以及隨機森林分類器


## Dataset 數據集
   The project expects the dataset to be organized as follows:
   本專案預期數據集組織如下：


   dataset/
   ├── manga_covers/
   └── real_photos/
   Each directory should contain 4000 images, with 3800 images used for training and 200 images used for testing.
   每個目錄應包含 4000 張圖像，其中 3800 張用於訓練，200 張用於測試。
  
   The user can adjust the numbers to match the dataset that will be used (this number is for the sample accuracy)
   請依照你的資料及去調整數字（示範精準度由此數字得出）
## Future Improvements 未來改進
   Some potential improvements for this project include:
   本專案的潛在改進包括：


   Implementing data augmentation techniques to increase the size and diversity of the training dataset.
   實現數據增強技術，以增加訓練數據集的大小和多樣性。
   Experimenting with different feature extraction methods or machine learning algorithms to improve classification performance.
   嘗試不同的特徵提取方法或機器學習算法，以提高分類性能。
   Developing a graphical user interface (GUI) to make the classifier more user-friendly.
   開發圖形用戶介面（GUI），使分類器更易於使用。
   Integrating the classifier into a larger application or web service.
   將分類器集成到更大的應用或網路服務中。



