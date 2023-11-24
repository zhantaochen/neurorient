import datetime

class Logger:
    """ A minimal logger class created by ChatGPT GPT-4.
    """
    def __init__(self):
        self._log_buffer = []

    def log(self, *args, print_message=True):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Join all the provided strings and append to the buffer
        message = f"[{timestamp}] " + " ".join(str(arg) for arg in args) + '\n'
        self._log_buffer.append(message)
        if print_message:
            print(message)

    def dump_to_file(self, directory):
        """Write all buffered log messages to a file."""
        with open(f'{directory}', 'w') as log_file:
            for message in self._log_buffer:
                log_file.write(message + '\n')

    def clear(self):
        """Clear the log buffer (optional)."""
        self._log_buffer.clear()
