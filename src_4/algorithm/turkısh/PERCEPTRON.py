import sklearn.linear_model
import os

from ...method import ModelHandler, DatasetHandler, TimerHandler

class PERCEPTRON(ModelHandler):
    def __init__(self, dataset_path, model_save_folder_path, model_file_name, should_load_from_dataset, resized_image_size, image_color_channel_count, test_ratio, print_update_time):
        self.is_model_suitable = False
        
        self.dataset_path = dataset_path
        self.is_model_suitable = self.check_dataset_folders()
        if not self.is_model_suitable:
            return
        
        self.model_save_folder_path = model_save_folder_path
        self.model_file_name = model_file_name
        self.model_save_path = os.path.join(self.model_save_folder_path, self.model_file_name)
        
        self.IMAGE_SIZE = resized_image_size
        self.IMAGE_COLOR_CHANNEL_COUNT = image_color_channel_count
        self.TEST_RATIO = test_ratio
        
        if should_load_from_dataset:
            self.dataset = DatasetHandler(self.dataset_path, self.IMAGE_SIZE, self.IMAGE_COLOR_CHANNEL_COUNT, self.TEST_RATIO)
            
        else:
            self.dataset = DatasetHandler(None, self.IMAGE_SIZE, self.IMAGE_COLOR_CHANNEL_COUNT, None)
            
        self.model = sklearn.linear_model.Perceptron()
        
        self.PRINT_UPDATE_TIME = print_update_time
        self.timer = TimerHandler(True, self.PRINT_UPDATE_TIME)
        
        return
    