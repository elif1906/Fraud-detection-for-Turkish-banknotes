import os
from src_4.algorithm import SVM, KNN, PERCEPTRON

def turkish_banksnotes_fraud_detection_test():
    print("Available Algorithms:\n")
    
    algorithms = [
        "[1] - Support Vector Machine (SVM)",
        "[2] - K-Nearest Neighbors (KNN)",
        "[3] - Perceptron"
    ]
    
    for algorithm in algorithms:
        print(algorithm)
        
    algorithm_index = input("\nWhich algorithm do you want to use?: ")
    print()
    
    try:
        selected_algorithm_index = int(algorithm_index)
        if not (0 < selected_algorithm_index <= len(algorithms)):
            print("Invalid index on algorithm selection.")
            exit(2)
            
    except ValueError:
        print("Invalid input type on algorithm selection.")
        exit(1)
        
    options = [
        "[1] - Train with Dataset Images",
        "[2] - Test with Dataset Images",
        "[3] - Train with User Input Image"
    ]
    
    for option_index in options:
        print(option_index)
        
    option_index = input("\nWhich test option do you want to use?: ")
    print()
        
    try:
        selected_option_index = int(option_index)
        if not (0 < selected_option_index <= len(options)):
            print("Invalid index on test option selection.")
            exit(2)
                
    except ValueError:
        print("Invalid input type on test option selection.")
        exit(1)

    # Dataset yolunu belirt
    dataset_path = "src_4/dataset"
    
    if selected_option_index == 3:
        image_path = input("Please enter image path: ")
        
        if not os.path.exists(image_path):
            print("Invalid image path.")
            exit(2)
            
    # Veri setinin varlığını kontrol et
    if not os.path.exists(dataset_path):
        print("Dataset is missing or located in an invalid path.")
        print("Please download the dataset from the provided link or adjust the dataset path.")
        exit(2)
            
    if selected_algorithm_index == 1:
        if selected_option_index == 1:
            svm = SVM(dataset_path, dataset_path, "svm_model.pkl", True, 64, 3, 0.25, 1)
            svm.train()
            
        elif selected_option_index == 2:
            svm = SVM(dataset_path, dataset_path, "svm_model.pkl", True, 64, 3, 0.25, 1)
            svm.test(None)
            
        elif selected_option_index == 3:
            svm = SVM(dataset_path, dataset_path, "svm_model.pkl", False, 64, 3, 0.25, 1)
            svm.test(image_path)
            
    elif selected_algorithm_index == 2:
        if selected_option_index == 1:
            knn = KNN(dataset_path, dataset_path, "knn_model.pkl", True, 64, 3, 0.25, 1, 1)
            knn.train()
            
        elif selected_option_index == 2:
            knn = KNN(dataset_path, dataset_path, "knn_model.pkl", True, 64, 3, 0.25, 1, 1)
            knn.test(None)
            
        elif selected_option_index == 3:
            knn = KNN(dataset_path, dataset_path, "knn_model.pkl", False, 64, 3, 0.25, 1, 1)
            knn.test(image_path)
            
    elif selected_algorithm_index == 3:
        if selected_option_index == 1:
            perceptron = PERCEPTRON(dataset_path, dataset_path, "perceptron_model.pkl", True, 64, 3, 0.25, 1)
            perceptron.train()
            
        elif selected_option_index == 2:
            perceptron = PERCEPTRON(dataset_path, dataset_path, "perceptron_model.pkl", True, 64, 3, 0.25, 1)
            perceptron.test(None)
            
        elif selected_option_index == 3:
            perceptron = PERCEPTRON(dataset_path, dataset_path, "perceptron_model.pkl", False, 64, 3, 0.25, 1)
            perceptron.test(image_path)
            
    return 0


def main():
    turkish_banksnotes_fraud_detection_test()
    return 0


if __name__ == '__main__':
    main()
