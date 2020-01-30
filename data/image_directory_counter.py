import os
import re


class DirectoryCounter:
    """
    A class that counts the number of directories with a matching hyperparameter string.
    """
    def __init__(self, hyperparameter_string):
        self.rootdir = os.path.dirname(os.path.realpath(__file__))
        self.regex = re.compile(hyperparameter_string)

    def count(self):
        """
        Count the number of directories with a hyperparameter string that matches hyperparameter_string.
        :return: An integer that indicates the number of directories with a matching hyperparameter string.
        """
        directory_count = 0
        for _, dirs, _ in os.walk(self.rootdir):
            for directory in dirs:
                print(directory)
                if self.regex.match(directory):
                    directory_count += 1
        return directory_count