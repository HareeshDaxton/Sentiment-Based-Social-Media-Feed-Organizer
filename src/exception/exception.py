import sys
from src.logging import logger

class FeedException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message
        exc_type, exc_value, exc_tb = error_detail.exc_info()
        
        if exc_tb:
            self.line_number = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.line_number = 0  # Or None
            self.file_name = "Custom Exception"
    def __str(self):
        return "Error occurred in python script name => [{0}] line number => [{1}] error message => [{2}]".format(self.file_name, self.line_number, str(self.error_message))