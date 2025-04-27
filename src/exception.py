import sys 

import logging
def error_message_detail(error, error_detail: sys):
    """
    This function takes an error and its details and returns a formatted string with the error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line number: {line_number} with message: {str(error)}"
    return error_message

class CustomException(Exception): 
    """
    Custom exception class that inherits from the built-in Exception class.
    It takes an error and its details and returns a formatted string with the error message.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    try:
        1 / 0
    except Exception as e:
        logging.info("Dividing by zero")
        raise CustomException(e, sys) from e
        # This will raise a CustomException with the error message and details
        # The error message will include the file name, line number, and the original error message
        # The 'from e' syntax is used to chain exceptions, so the original exception is preserved
        # The __str__ method of the CustomException class will be called to get the formatted error message
        # The error message will be printed to the console 