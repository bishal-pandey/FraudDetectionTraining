import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    '''Return detailed error message including file name, line number, and error message.'''

    # Extract traceback information(Exception information)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name and line number
    file_name = exc_tb.tb_frame.f_code.co_filename

    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file '{file_name}' at line {line_number}: {str(error)}"
    
    #logging error
    logging.error(error_message)
    return error_message


class CustomException(Exception):
    '''Custom exception class for handling specific errors.'''
    def __init__(self,error_message: str, error_detail: sys):
        super().__init__(error_message)

        # Format the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message