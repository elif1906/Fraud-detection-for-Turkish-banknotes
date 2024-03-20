import time

class TimerHandler:
    def __init__(self, is_model_main_thread_finished, print_update_time):
        self.is_model_main_thread_finished = is_model_main_thread_finished
        self.PRINT_UPDATE_TIME =  print_update_time
        
        self.HOURS_TO_MINUTES = 60
        self.MINUTES_TO_SECONDS = 60
        self.HOURS_TO_SECONDS = self.HOURS_TO_MINUTES * self.MINUTES_TO_SECONDS
        
        return
    
    def print_elapsed_time(self, is_training):
        maximum_print_size = 0
    
        start_time = time.perf_counter()
        while not self.is_model_main_thread_finished:
            elapsed_time = time.perf_counter() - start_time
        
            if is_training:
                text_to_print = f"Model has been training for {self.format_time(elapsed_time)}."
            
            else:
                text_to_print = f"Model has been testing for {self.format_time(elapsed_time)}."
            
            length_of_text = len(text_to_print)
            print(text_to_print, " " * (maximum_print_size - length_of_text), end='\r')
            if length_of_text > maximum_print_size:
                maximum_print_size = length_of_text
                
            if self.PRINT_UPDATE_TIME >= 0:
                time.sleep(self.PRINT_UPDATE_TIME)
                
        return

    def format_time(self, time_in_s):
        hours, seconds = divmod(time_in_s, self.HOURS_TO_SECONDS)
        minutes, seconds = divmod(time_in_s, self.MINUTES_TO_SECONDS)
        miliseconds = (seconds * 1000) % 1000
    
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        milliseconds = int(miliseconds)
    
        formatted_time = ""
        if hours == 0 and minutes == 0 and seconds == 0 and milliseconds == 0:
            formatted_time = "0ms"
    
        else:
            if hours > 0:
                formatted_time += f"{hours}h "
            if minutes > 0:
                formatted_time += f"{minutes}m "
            if seconds > 0:
                formatted_time += f"{seconds}s "
            if milliseconds > 0:
                formatted_time += f"{milliseconds}ms"
                
        return formatted_time.lstrip().rstrip()
    