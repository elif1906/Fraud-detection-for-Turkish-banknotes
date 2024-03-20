import sklearn.metrics
import pickle
import os
import threading

class ModelHandler:
    def __init__(self):
        self.is_model_suitable = False
        print("Model handler is not suitable for model training or testing. Please use a specific model.")
        return
    
    def train(self):
        if not self.is_model_suitable:
            print("Model is not suitable for training. Please check the error messages above.")
            return
        
        if self.check_for_model(True):
            self.timer.is_model_main_thread_finished = True
            return
        
        self.timer.is_model_main_thread_finished = False
            
        train_thread = threading.Thread(target=self.train_model)
        time_thread = threading.Thread(target=self.timer.print_elapsed_time, kwargs={ 'is_training': True })
        
        train_thread.start()
        time_thread.start()
        time_thread.join()
        
        return
    
    def test(self, user_image_path):
        if not self.is_model_suitable:
            print("Model is not suitable for testing. Please check the error messages above.")
            return
        
        if self.check_for_model(False):
            self.timer.is_model_main_thread_finished = True
            return
        
        self.timer.is_model_main_thread_finished = False
        
        if user_image_path is not None:
            user_image = self.dataset.process_image(user_image_path, True)
            test_thread = threading.Thread(target=self.test_user_image, kwargs={ 'user_image': user_image })
            
        else:
            test_thread = threading.Thread(target=self.test_model)
            
        time_thread = threading.Thread(target=self.timer.print_elapsed_time, kwargs={ 'is_training': False })
        
        test_thread.start()
        time_thread.start()
        time_thread.join()
        
        return
    
    def test_user_image(self, user_image):
        print("\nModel testing started.")
        print("Please wait while the model is being trained.")
        print("The time needed is dependent on your computer's hardware.")
        predictions = self.model.predict(user_image)
        self.timer.is_model_main_thread_finished = True
        print(f"\nModel testing finished.")
        
        prediction_text = "\nModel predicted this image as the "
        if len(predictions) > 1:
            prediction_text += "classes of "
            for index, prediction in enumerate(predictions):
                if index == len(predictions) - 1:
                    prediction_text = prediction_text.rstrip(", ") + " and "
                    
                prediction_text += f"'{prediction} TL', "
                
            prediction_text = prediction_text.rstrip(", ") + '.'
                
        else:
            prediction_text += ("class of '" + predictions[0] + "TL'.")
            
        print(prediction_text)
        
        return
    
    def train_model(self):
        if not self.is_model_suitable:
            print("Model is not suitable for training. Please check the error messages above.")
            return
        
        print("\nModel training started.")
        print("Please wait while the model is being trained.")
        print("The time needed is dependent on your computer's hardware.")
        self.model.fit(self.dataset.X_train, self.dataset.Y_train)
        self.timer.is_model_main_thread_finished = True
        print("\nModel training finished.")
        
        if not os.path.exists(self.model_save_folder_path):
            os.makedirs(self.model_save_folder_path)
            
        with open(self.model_save_path, "wb") as file:
            pickle.dump(self.model, file)
            
        print(f"Model has been saved to \"{self.model_save_path}\".")
        
        return
    
    def test_model(self):
        if not self.is_model_suitable:
            print("Model is not suitable for training. Please check the error messages above.")
            return
        
        print("\nModel testing started.")
        print("Please wait while the model is being trained.")
        print("The time needed is dependent on your computer's hardware.")
        predictions = self.model.predict(self.dataset.X_test)
        accuracy = sklearn.metrics.accuracy_score(self.dataset.Y_test, predictions)
        self.timer.is_model_main_thread_finished = True
        print(f"\nModel testing finished.")
        
        print(f"Accuracy: {round((accuracy * 100), 2)}%")
        
        return
    
    def check_for_model(self, is_training):
        if not self.is_model_suitable:
            print("Model is not suitable for training. Please check the error messages above.")
            return
        
        if is_training:
            if os.path.exists(self.model_save_path):
                overwrite_confirm = input("An already trained model is found. Do you want to retrain the model? (Y/N):").upper().rstrip().lstrip()
                if overwrite_confirm != 'Y':
                    self.timer.is_model_main_thread_finished = True
                
                    with open(self.model_save_path, "rb") as file:
                        self.model = pickle.load(file)
                    
                    return True
                
                else:
                    os.remove(self.model_save_path)
                    self.timer.is_model_main_thread_finished = False
                    return False
                
        else:
            if os.path.exists(self.model_save_path):
                with open(self.model_save_path, "rb") as file:
                    self.timer.is_model_main_thread_finished = False
                    
                    self.model = pickle.load(file)
                    return False
                
            else:
                print("No trained model found. Please train the model first.")
                self.timer.is_model_main_thread_finished = True
                return True
            
    def check_dataset_folders(self):
        if not os.path.exists(self.dataset_path):
            self.warn_about_dataset(True, None)
            return False
        
        else:
            dataset_must_have_folders = ["5", "10", "20", "50", "100", "200"]
            contents = os.listdir(self.dataset_path)
            missing_folders = []
            
            for folder in dataset_must_have_folders:
                if folder not in contents:
                    missing_folders.append(folder)
                    
            if len(missing_folders) > 0:
                self.warn_about_dataset(False, missing_folders)
                return False
            
            else:
                return True
            
    def warn_about_dataset(self, is_dataset_completely_missing, missing_folders):
        if is_dataset_completely_missing:
            print("Dataset is completely missing.")
        
        else:
            missing_folders_text = ""
            if len(missing_folders) > 1:
                missing_folders_text = "Folders; "
                for index, missing_folder in enumerate(missing_folders):
                    if index == len(missing_folders) - 1:
                        missing_folders_text = missing_folders_text.rstrip(", ") + " and "
                        
                    missing_folders_text += f"'{missing_folder}', "
                    
                missing_folders_text = missing_folders_text.rstrip(", ") + " are missing."
                
            else:
                missing_folders_text = f"Folder; '{missing_folders[0]}' is missing."
                
            print(missing_folders_text)
        
        print("Please, either;")
        return
    