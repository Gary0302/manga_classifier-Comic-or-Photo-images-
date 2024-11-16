#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// Function to extract HOG features from an image
std::vector<float> extractHOGFeatures(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(128, 128));
    
    cv::Mat gray;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = resized;
    }
    
    cv::HOGDescriptor hog(
        cv::Size(128, 128),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );
    
    std::vector<float> descriptors;
    hog.compute(gray, descriptors);
    return descriptors;
}

// Function to load and prepare dataset
void prepareDataset(const std::string& manga_dir, const std::string& photo_dir, 
                   int train_samples, int test_samples,
                   cv::Mat& trainData, cv::Mat& trainLabels,
                   cv::Mat& testData, cv::Mat& testLabels) {
    // 使用傳入的 manga_dir 和 photo_dir 參數，而不是硬編碼的路徑
    std::string manga_directory = manga_dir;  // 使用傳入的參數
    std::string real_photos_directory = photo_dir;  // 使用傳入的參數

    std::vector<std::string> manga_files;
    std::vector<std::string> photo_files;
    
    // Load manga files with absolute paths
    for (const auto& entry : fs::directory_iterator(manga_directory)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            manga_files.push_back(fs::absolute(entry.path()).string());
        }
    }
    
    // Load photo files with absolute paths
    for (const auto& entry : fs::directory_iterator(real_photos_directory)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            photo_files.push_back(fs::absolute(entry.path()).string());
        }
    }
    
    if (manga_files.empty() || photo_files.empty()) {
        throw std::runtime_error("No image files found in one or both directories");
    }
    
    // Shuffle files
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(manga_files.begin(), manga_files.end(), gen);
    std::shuffle(photo_files.begin(), photo_files.end(), gen);
    
    // Get feature size
    cv::Mat sample_img = cv::imread(manga_files[0], cv::IMREAD_COLOR);
    if (sample_img.empty()) {
        throw std::runtime_error("Failed to load sample image: " + manga_files[0]);
    }
    int feature_size = extractHOGFeatures(sample_img).size();
    
    // Initialize matrices
    trainData = cv::Mat(train_samples * 2, feature_size, CV_32F);
    trainLabels = cv::Mat(train_samples * 2, 1, CV_32S);
    testData = cv::Mat(test_samples * 2, feature_size, CV_32F);
    testLabels = cv::Mat(test_samples * 2, 1, CV_32S);
    
    // Process training data
    for (int i = 0; i < train_samples; i++) {
        // Process manga images
        cv::Mat img = cv::imread(manga_files[i], cv::IMREAD_COLOR);
        if (!img.empty()) {
            std::vector<float> features = extractHOGFeatures(img);
            for (size_t j = 0; j < features.size(); j++) {
                trainData.at<float>(i, j) = features[j];
            }
            trainLabels.at<int>(i) = 1;
        }
        
        // Process photo images
        img = cv::imread(photo_files[i], cv::IMREAD_COLOR);
        if (!img.empty()) {
            std::vector<float> features = extractHOGFeatures(img);
            for (size_t j = 0; j < features.size(); j++) {
                trainData.at<float>(i + train_samples, j) = features[j];
            }
            trainLabels.at<int>(i + train_samples) = -1;
        }
    }
    
    // Process test data
    for (int i = 0; i < test_samples; i++) {
        // Process manga test images
        cv::Mat img = cv::imread(manga_files[i + train_samples], cv::IMREAD_COLOR);
        if (!img.empty()) {
            std::vector<float> features = extractHOGFeatures(img);
            for (size_t j = 0; j < features.size(); j++) {
                testData.at<float>(i, j) = features[j];
            }
            testLabels.at<int>(i) = 1;
        }
        
        // Process photo test images
        img = cv::imread(photo_files[i + train_samples], cv::IMREAD_COLOR);
        if (!img.empty()) {
            std::vector<float> features = extractHOGFeatures(img);
            for (size_t j = 0; j < features.size(); j++) {
                testData.at<float>(i + test_samples, j) = features[j];
            }
            testLabels.at<int>(i + test_samples) = -1;
        }
    }
}




// //SVM classifier around 50 percent accuracy

// int main() {
//     try {
//         std::string manga_dir = "../dataset/manga_covers";
//         std::string photo_dir = "../dataset/real_photos";
        
//         // Check if directories exist
//         if (!fs::exists(manga_dir) || !fs::exists(photo_dir)) {
//             std::cerr << "Error: One or both directories do not exist!" << std::endl;
//             return 1;
//         }
        
//         cv::Mat trainData, trainLabels, testData, testLabels;
//         int train_samples = 650;
//         int test_samples = 150;
        
//         std::cout << "Preparing dataset..." << std::endl;
//         prepareDataset(manga_dir, photo_dir, train_samples, test_samples,
//                       trainData, trainLabels, testData, testLabels);
        

//         std::cout << "Training SVM classifier..." << std::endl;
//         cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
//         svm->setType(cv::ml::SVM::C_SVC);
//         svm->setKernel(cv::ml::SVM::RBF);
//         svm->setC(1.0);       // 設定懲罰參數 C，通常在 [0.01, 100] 之間選擇
//         svm->setGamma(0.5);   // 設定 gamma 參數，通常在 [0.01, 1.0] 之間選擇
//         svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));
//         svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);
//         std::cout << "Testing classifier..." << std::endl;
//         cv::Mat predictions;
//         svm->predict(testData, predictions);
//         int correct = 0;
//         for (int i = 0; i < predictions.rows; i++) {
//             if (predictions.at<float>(i) == testLabels.at<int>(i)) {
//                 correct++;
//             }
//         }
        
//         float accuracy = (float)correct / predictions.rows * 100;
//         std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

//         //svm
//         svm->save("manga_classifier_model.yml");

//         std::cout << "Model saved as 'manga_classifier_model.yml'" << std::endl;
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
    
//     return 0;
// }




// //KNN classifier around 52 percent accuracy
// int main() {
//     try {
//         std::string manga_dir = "../dataset/manga_covers";
//         std::string photo_dir = "../dataset/real_photos";
        
//         // Check if directories exist
//         if (!fs::exists(manga_dir) || !fs::exists(photo_dir)) {
//             std::cerr << "Error: One or both directories do not exist!" << std::endl;
//             return 1;
//         }
        
//         cv::Mat trainData, trainLabels, testData, testLabels;
//         int train_samples = 650;
//         int test_samples = 150;
        
//         std::cout << "Preparing dataset..." << std::endl;
//         prepareDataset(manga_dir, photo_dir, train_samples, test_samples,
//                       trainData, trainLabels, testData, testLabels);
        
//         // Normalize training and testing data
//         cv::normalize(trainData, trainData, 0, 1, cv::NORM_MINMAX, CV_32F);
//         cv::normalize(testData, testData, 0, 1, cv::NORM_MINMAX, CV_32F);

//         // KNN classifier
//         std::cout << "Training KNN classifier..." << std::endl;
//         // Train KNN classifier
//         cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
//         knn->setDefaultK(5);  // Set K value to 5 (you can experiment with this)
//         knn->setIsClassifier(true);       
//         // Train KNN model
//         knn->train(trainData, cv::ml::ROW_SAMPLE, trainLabels); 
//         std::cout << "Testing classifier..." << std::endl;
//         cv::Mat predictions;
//         knn->predict(testData, predictions);

//         // Calculate accuracy
//         int correct = 0;
//         for (int i = 0; i < predictions.rows; i++) {
//             if (predictions.at<float>(i) == testLabels.at<int>(i)) {
//                 correct++;
//             }
//         }
        
//         float accuracy = (float)correct / predictions.rows * 100;
//         std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
        
//         // Save KNN model
//         knn->save("manga_classifier_knn_model.yml");
//         std::cout << "Model saved as 'manga_classifier_knn_model.yml'" << std::endl;
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
    
//     return 0;
// }




//RF classifier around 85 percent accuracy
int main() {
    try {
        std::string manga_dir = "../dataset/manga_covers";
        std::string photo_dir = "../dataset/real_photos";
        
        // Check if directories exist
        if (!fs::exists(manga_dir) || !fs::exists(photo_dir)) {
            std::cerr << "Error: One or both directories do not exist!" << std::endl;
            return 1;
        }
        
        cv::Mat trainData, trainLabels, testData, testLabels;
        int train_samples = 650;
        int test_samples = 150;
        
        std::cout << "Preparing dataset..." << std::endl;
        prepareDataset(manga_dir, photo_dir, train_samples, test_samples,
                      trainData, trainLabels, testData, testLabels);
        
        // Train Random Forest classifier
        std::cout << "Training Random Forest classifier..." << std::endl;
        cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::create();
        rf->setMaxDepth(10);         // Maximum depth of the trees
        rf->setMinSampleCount(10);   // Minimum number of samples per leaf
        rf->setMaxCategories(10);    // Maximum number of categories in the leaf nodes
        rf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));
        
        rf->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);
        
        std::cout << "Testing classifier..." << std::endl;
        cv::Mat predictions;
        rf->predict(testData, predictions);
        
        int correct = 0;
        for (int i = 0; i < predictions.rows; i++) {
            if (predictions.at<float>(i) == testLabels.at<int>(i)) {
                correct++;
            }
        }
        
        float accuracy = (float)correct / predictions.rows * 100;
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
        
        rf->save("manga_classifier_rf_model.yml");
        std::cout << "Model saved as 'manga_classifier_rf_model.yml'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
