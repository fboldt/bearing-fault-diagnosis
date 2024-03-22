import sys
import re

class ConsoleOutputToFile:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
        self.blocked_patterns = [
            re.compile(r".*\d+/\d+ \[=*"),
            re.compile(r".*loading acquisitions \d{1,2}\.\d{2} %"),  
        ]

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_lines_matching_regex()
        sys.stdout = self.stdout
        self.file.close()

    def write(self, text):
        if not any(pattern.match(text) for pattern in self.blocked_patterns):
            self.stdout.write(text)
            self.stdout.flush()
            self.file.write(text)
            self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def remove_lines_matching_regex(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if not any(re.search(pattern, line) for pattern in self.blocked_patterns)]

        with open(self.filename, 'w') as file:
            for line in filtered_lines:
                if line.strip():
                    file.write(line)