# In this project, I am implementing a KNN Classifier to classify the Iris data. It should be selected via the console
# whether (1.) an exemplary run of the dataset should be carried out, or whether (2.) a cross-validation should be
# carried out to find the optimal k-range.

# For (1.) the desired K parameter should be entered via the console. Then the programm should output the accuracy of
# the algorithm and plot a diagram in which all not correctly classified data points are marked red.

# For (2.) the range of the k parameters, as well as the number of iterations over different data splits should be
# entered via the console. The algorithm should then output the average accuracy for each k and plot a diagram in which
# the average accuracy for each k is displayed.

# ----------------------------------------------------------------------------------------------------------------------
# 1. Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# ----------------------------------------------------------------------------------------------------------------------
# 2. Define classes and methods

# 2.1 Define the KNN Classifier class
class KNNClassifier:
    def __init__(self, k):
        self.k = k              # Initialize the K parameter
        self.x_train = None     # Initialize the training data
        self.y_train = None     # Initialize the training labels

    # Method to load a csv file
    def load_csv_file(self, filename):
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # skip the header
                dataset = list(csv_reader)
            return dataset
        except FileNotFoundError:
            print(f"Die Datei '{filename}' wurde nicht gefunden.")              # Error handling for file not found
            return None
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten beim Lesen der Datei: {e}")      # Error handling for other exceptions
            return None

    # Method to split the dataset into training and testing sets
    def split_data(self, dataset, train_ratio):
        np.random.seed(44)                  # Set Seeds for reproducibility
        np.random.shuffle(dataset)
        train_size = int(len(dataset) * train_ratio)
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]
        return train_set, test_set

    # Method to convert training and test data sets to numpy arrays via List Comprehension and convert strings
    # to numerics
    def convert_to_numpy_arrays(self, dataset):
        x = np.array([data[:-1] for data in dataset], dtype=np.float64)
        y = np.array([data[-1] for data in dataset], dtype=np.int64)
        return x, y

    # Method for saving the training data in class and normalizing the x_data
    # Chose min-max normalization over standardisation to normalize the data in the range of 0 to 1
    # Standardisation would be another option, but it would be more complex to implement and normalization
    # is just fine for this dataset
    def train_data(self, x_train, y_train):
        self.x_min = x_train.min(axis=0)
        self.x_max = x_train.max(axis=0)
        self.x_train = (x_train - self.x_min) / (self.x_max - self.x_min)
        self.y_train = y_train

    # Method to predict the class of the query data
    def predict(self, x_query):
        x_query = (x_query - self.x_min) / (self.x_max - self.x_min)    # normalize the test data
        dist = np.sqrt(np.sum((self.x_train - x_query) ** 2, axis=1))
        k_nearest = np.argsort(dist)[:self.k]   # sorting the k nearest neighbors by distance
        most_common = np.bincount(self.y_train[k_nearest]).argmax() # majority vote to get the class of the query data
        return most_common

    # Method to calculate the accuracy of the algorithm
    def calculate_accuracy(self, x_test, y_test):
        correct = sum(self.predict(x_test[i]) == y_test[i] for i in range(len(x_test)))
        accuracy = correct / len(y_test)
        return accuracy

    def perform_cross_validation(self, dataset, k, iterations):
        current_accuracies = []

        for _ in range(iterations):
                train_set, test_set = self.split_data(dataset, train_ratio=0.7)
                x_train, y_train = self.convert_to_numpy_arrays(train_set)
                x_test, y_test = self.convert_to_numpy_arrays(test_set)

                if x_train is None or y_train is None or x_test is None or y_test is None:
                    print("Error converting dataset to numpy arrays.")
                    return None

                self.train_data(x_train, y_train)
                accuracy = self.calculate_accuracy(x_test, y_test)
                current_accuracies.append(accuracy)

        mean_accuracy = np.mean(current_accuracies)
        return mean_accuracy

# 2.2 Define the Plot class
class Plot:
    def __init__(self, classifier):
        self.classifier = classifier        # Composition of the KNN Classifier class to access the predict method

    # Method to plot the accuracy of the KNN Classifier for different k
    def plot_accuracy(self, k_range, accuracy):
        plt.plot(k_range, accuracy, marker='o', color='b', ls='--', lw=2)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of the KNN Classifier for different k')
        plt.show()

    # Method to plot all data points
    def plot_data(self, x_train, y_train, x_test, y_test, k, classifier):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        colors = ['blue', 'green', 'purple']

        # First plotting the training data
        for i in range(len(x_train)):
            axs[0].scatter(x_train[i][0], x_train[i][1], color=colors[int(y_train[i])])
            axs[1].scatter(x_train[i][2], x_train[i][3], color=colors[int(y_train[i])])

        # Then plotting the test data
        for i in range(len(x_test)):
            predicted_class = classifier.predict(x_test[i])
            actual_class = y_test[i]

            # If the predicted class is not equal to the actual class, mark the data point as red
            if predicted_class != actual_class:
                color = 'red'
                alpha = 1
            else:
                color = colors[int(actual_class)]
                alpha = 0.2

            axs[0].scatter(x_test[i][0], x_test[i][1], color=color, edgecolor='k', alpha=alpha)
            axs[1].scatter(x_test[i][2], x_test[i][3], color=color, edgecolor='k', alpha=alpha)

        # Set the labels and titles
        axs[0].set(xlabel='Sepal Length', ylabel='Sepal Width', title='Sepal Length vs Sepal Width')
        axs[1].set(xlabel='Petal Length', ylabel='Petal Width', title='Petal Length vs Petal Width')

        # Add a legend with markers for each class
        patches = [mpatches.Patch(color=color, label=label) for color, label in
                   zip(['blue', 'green', 'purple', 'red'], ['Setosa', 'Versicolor', 'Virginica', 'Misclassified'])]
        plt.legend(handles=patches, loc='upper left')
        plt.tight_layout()
        plt.show()

# 2.3 Define the HomeScreen class
class HomeScreen:
    def __init__(self, classifier, plotter):
        self.classifier = classifier            # Composition of the KNN Classifier and Plot classes
        self.plotter = plotter                  # to access their methods in the HomeScreen class

    # Method to display the welcome message and return the selected option
    def welcome_message(self):
        print('\nWelcome to the KNN Classifier for the Iris Dataset!')
        print('Please select one of the following options:')
        print('1. Run the KNN Classifier with a specific k')
        print('2. Perform cross-validation to find the optimal k')
        print('3. Exit')
        option = self.get_integer_input('Please enter 1, 2 or 3: ')
        return option

    # Method to run the classifier with a specific k
    def run_classifier(self, dataset):
        k = self.get_integer_input('Enter the value of k: ')
        self.classifier.k = k  # Update the k value in classifier
        train_set, test_set = self.classifier.split_data(dataset, train_ratio=0.7)      # Split the dataset
        x_train, y_train = self.classifier.convert_to_numpy_arrays(train_set)           # Convert to numpy arrays
        x_test, y_test = self.classifier.convert_to_numpy_arrays(test_set)
        self.classifier.train_data(x_train, y_train)                                    # Normalize the training data
        accuracy = self.classifier.calculate_accuracy(x_test, y_test)                   # Calculate the accuracy
        print(f'Accuracy of the KNN classifier with k={k}: {accuracy:.3f}')
        self.plotter.plot_data(x_train, y_train, x_test, y_test, k, self.classifier)    # Plot the data

    # Method to get integer input from the user with custom prompt
    def get_integer_input(self, prompt):
        while True: # Loop until a valid integer is entered
            try:
                value = int(input(prompt))
                if value <= 0:
                    print("Please enter a positive integer.")
                else:
                    return value
            except ValueError:
                print("Please enter a valid integer.")

    # Method to perform cross-validation to find the optimal k
    def run_cross_validation(self, dataset):
        try:
            # Get the parameters for cross-validation
            k_start = self.get_integer_input('Enter the start of k range: ')
            k_end = self.get_integer_input('Enter the end of k range: ')
            if k_start >= k_end:
                print("Start of k range must be less than end of k range.")
                return

            iterations = self.get_integer_input('Enter the number of iterations for cross-validation: ')
            if iterations <= 0:
                print("Number of iterations must be greater than zero.")
                return

            accuracies = []
            k_range = range(k_start, k_end + 1)

            # Perform cross-validation for each k in the range
            for k in k_range:
                mean_accuracy = self.classifier.perform_cross_validation(dataset, k, iterations)
                if mean_accuracy is None:
                    print("An error occurred during cross-validation.")
                    return  # Error handling, stop execution if there was an issue
                accuracies.append(mean_accuracy)

            # Plot the accuracy for each k
            self.plotter.plot_accuracy(k_range, accuracies)
            best_k = k_range[np.argmax(accuracies)]
            print(f'Optimal k found to be {best_k} with an accuracy of {max(accuracies):.3f}')
        except Exception as e:
            print(f"An unexpected error occurred during cross-validation: {e}")

# ----------------------------------------------------------------------------------------------------------------------
# 3. Define main function

def main():
    # create instances of the necessary classes
    classifier = KNNClassifier(k=3) # initialize the KNN Classifier with k=3, can be updated by the user
    plotter = Plot(classifier)
    home_screen = HomeScreen(classifier, plotter)
    dataset = classifier.load_csv_file('iris_data.csv')

    # Structure of program-interface
    option = ''
    while option != 3:
        option = home_screen.welcome_message()
        match option:
            case 1:
                home_screen.run_classifier(dataset)
            case 2:
                home_screen.run_cross_validation(dataset)
            case 3:
                print('Goodbye!')
            case _:
                print('Invalid option. Please select 1, 2 or 3')

if __name__ == '__main__':
    main()