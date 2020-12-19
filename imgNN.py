#10x10 pixels of certain colors --> 100 NN abs
# 2 functions, one getImage() generates the getImage
# oracle(image) and it returns the color it thinks the image is


from PIL import Image

import random
import math
import random

class PixelImages():

    def __init__(self, number_of_images):
        self.number_of_images = number_of_images

    # function in charge of randomizing colors for when creatin the image
    def randomColors(self):
        red = (255,0,0)
        yellow = (255,255,0)
        blue = (0,0,255)
        orange = (255,128,0)
        purple = (127,0,255)
        green = (0,255,0)
        num = random.randrange(7)
        if num == 0:
            return red
        elif num == 1:
            return yellow
        elif num == 2:
            return blue
        elif num == 3:
            return orange
        elif num == 4:
            return purple
        elif num == 5:
            return green
        #make it so that one color is more prominent
        elif num == 6:
            return green

    #creates a 10x10 random image with randomly selected colors for each pixel
    def newImage(self):
        for n in range(self.number_of_images):
            im = Image.new("RGB", (10, 10))
            pix = im.load()
            for x in range(10):
                for y in range(10):
                    pix[x,y] = self.randomColors()
            im.save("images/test" + str(n)+ ".png", "PNG")

        # gets a list of pixel tuples with each tuple cointaining
        # the rbg values of its pixel
        def getPixelValues():
            images = []
            pix_val = []
            for i in range(self.number_of_images):
                images.append(Image.open('images/test'+ str(i) +'.png', 'r'))
                pix_val.append(list(images[i].getdata()))

            return pix_val


        self.getRGBValues()
        return getPixelValues()

    def getRGBValues(self):
        images = []
        pix_val = []
        rgb_val = []
        for i in range(self.number_of_images):
            images.append(Image.open('images/test'+ str(i) +'.png', 'r'))
            pix_val.append(list(images[i].getdata()))
            rgb_val.append([])
            for tuple in pix_val[i]:
                for n in tuple:
                    input = n/255
                    rgb_val[i].append(input)

        return rgb_val

    # generates the output based on the for each of the randomly created images
    def oracle(self, pixelList):
            #checks to see which of the pixels is the dominant one
        def colorCheck():
            if red_pixels >= yellow_pixels and red_pixels >= blue_pixels and red_pixels >= orange_pixels and red_pixels >= green_pixels and red_pixels >= purple_pixels:
                return [1,0,0,0,0,0]
            elif yellow_pixels >= red_pixels and yellow_pixels >= blue_pixels and yellow_pixels >= orange_pixels and yellow_pixels >= green_pixels and yellow_pixels >= purple_pixels:
                return [0,1,0,0,0,0]
            elif blue_pixels >= red_pixels and blue_pixels >= yellow_pixels and blue_pixels >= orange_pixels and blue_pixels >= green_pixels and blue_pixels >= purple_pixels:
                return [0,0,1,0,0,0]
            elif orange_pixels >= red_pixels and orange_pixels >= yellow_pixels and orange_pixels >= blue_pixels and orange_pixels >= green_pixels and orange_pixels >= purple_pixels:
                return [0,0,0,1,0,0]
            elif purple_pixels >= red_pixels and purple_pixels >= yellow_pixels and purple_pixels >= blue_pixels and purple_pixels >= orange_pixels and purple_pixels >= green_pixels:
                return [0,0,0,0,1,0]
            elif green_pixels >= red_pixels and green_pixels >= yellow_pixels and green_pixels >= blue_pixels and green_pixels >= orange_pixels and green_pixels >= purple_pixels:
                return [0,0,0,0,0,1]


        # def colorCheck():
        #     if red_pixels >= yellow_pixels and red_pixels >= blue_pixels and red_pixels >= orange_pixels and red_pixels >= green_pixels and red_pixels >= purple_pixels:
        #         return [1,0,0,0,0,0]
        #     elif yellow_pixels >= red_pixels and yellow_pixels >= blue_pixels and yellow_pixels >= orange_pixels and yellow_pixels >= green_pixels and yellow_pixels >= purple_pixels:
        #         return [0,1,0,0,0,0]
        #     elif blue_pixels >= red_pixels and blue_pixels >= yellow_pixels and blue_pixels >= orange_pixels and blue_pixels >= green_pixels and blue_pixels >= purple_pixels:
        #         return [0,0,1,0,0,0]
        #     elif orange_pixels >= red_pixels and orange_pixels >= yellow_pixels and orange_pixels >= blue_pixels and orange_pixels >= green_pixels and orange_pixels >= purple_pixels:
        #         return [0,0,0,1,0,0]
        #     elif purple_pixels >= red_pixels and purple_pixels >= yellow_pixels and purple_pixels >= blue_pixels and purple_pixels >= orange_pixels and purple_pixels >= green_pixels:
        #         return [0,0,0,0,1,0]
        #     elif green_pixels >= red_pixels and green_pixels >= yellow_pixels and green_pixels >= blue_pixels and green_pixels >= orange_pixels and green_pixels >= purple_pixels:
        #         return [0,0,0,0,0,1]
        #     else:
        #         return 22

            # total_pixels = 100
            # red_percentage = red_pixels / total_pixels
            # yellow_percentage = yellow_pixels / total_pixels
            # blue_percentage = blue_pixels / total_pixels
            # orange_percentage = orange_pixels / total_pixels
            # green_percentage = green_pixels / total_pixels
            # purple_percentage = green_pixels / total_pixels
            # return [red_percentage,yellow_percentage,blue_percentage,orange_percentage,green_percentage,purple_percentage]
            #

        red = (255,0,0)
        yellow = (255,255,0)
        blue = (0,0,255)
        orange = (255,128,0)
        purple = (127,0,255)
        green = (0,255,0)

        red_pixels = 0
        yellow_pixels = 0
        blue_pixels = 0
        orange_pixels = 0
        purple_pixels = 0
        green_pixels = 0


        results = []
        for list in pixelList:
            red_pixels = 0
            yellow_pixels = 0
            blue_pixels = 0
            orange_pixels = 0
            purple_pixels = 0
            green_pixels = 0
            # print(len(pixelList))
            for tuple in list:
                if tuple == red:
                    red_pixels += 1
                elif tuple == yellow:
                    yellow_pixels += 1
                elif tuple == blue:
                    blue_pixels += 1
                elif tuple == orange:
                    orange_pixels += 1
                elif tuple == purple:
                    purple_pixels += 1
                elif tuple == green:
                    green_pixels += 1
            results.append(colorCheck())
        return results



class NeuralNetwork1():
    # 6 methods for this class: 4 of them are private, indicated by
    # the prefix of double underscore "__" and 2 are public which we
    # will be able to  call externally

    def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		# For testing and debugging purposes
		# random.seed(1)

		# When the NeuralNetwork is initialized we want it to have 3 weights
        # because we will have 3 inputs coming into the NN
		# Create 3 weights and set them to random values in the range -1 to +1
        self.weights = []
        for i in range(300):
            self.weights.append(random.uniform(-1, 1))

    # Make a prediction
    def think(self, neuron_inputs):
        # First step is to calculate the weighted sum of the inputs
        neuron_output = []

        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        # And then we need to take that sum of the weighted inputs
        # and pass it through our activation function which is the sigmoid

        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            print("Iteration: ", iteration)
            for training_set_example in training_set_examples:

                # Predict the output based on the training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                # Calculate the error as the difference between the desired output and the predicted output

                error_in_output = training_set_example["output"][0] - predicted_output
                    # Iterate through the weights and adjust each one
                for index in range(len(self.weights)):

                    # Get the neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    # Calculate how much to adjust the weights by using the delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    # Adjust the weight
                    self.weights[index] += adjust_weight


	# Calculate the sigmoid (our activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):

        return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
#END OF CLASS

class NeuralNetwork2():
    # 6 methods for this class: 4 of them are private, indicated by
    # the prefix of double underscore "__" and 2 are public which we
    # will be able to  call externally

    def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		# For testing and debugging purposes
		# random.seed(1)

		# When the NeuralNetwork is initialized we want it to have 3 weights
        # because we will have 3 inputs coming into the NN
		# Create 3 weights and set them to random values in the range -1 to +1
        self.weights = []
        for i in range(300):
            self.weights.append(random.uniform(-1, 1))

    # Make a prediction
    def think(self, neuron_inputs):
        # First step is to calculate the weighted sum of the inputs
        neuron_output = []

        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        # And then we need to take that sum of the weighted inputs
        # and pass it through our activation function which is the sigmoid

        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            print("Iteration: ", iteration + number_of_iterations)
            for training_set_example in training_set_examples:

                # Predict the output based on the training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                # Calculate the error as the difference between the desired output and the predicted output

                error_in_output = training_set_example["output"][1] - predicted_output
                    # Iterate through the weights and adjust each one
                for index in range(len(self.weights)):

                    # Get the neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    # Calculate how much to adjust the weights by using the delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    # Adjust the weight
                    self.weights[index] += adjust_weight


	# Calculate the sigmoid (our activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):

        return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
#END OF CLASS
class NeuralNetwork3():
    # 6 methods for this class: 4 of them are private, indicated by
    # the prefix of double underscore "__" and 2 are public which we
    # will be able to  call externally

    def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		# For testing and debugging purposes
		# random.seed(1)

		# When the NeuralNetwork is initialized we want it to have 3 weights
        # because we will have 3 inputs coming into the NN
		# Create 3 weights and set them to random values in the range -1 to +1
        self.weights = []
        for i in range(300):
            self.weights.append(random.uniform(-1, 1))

    # Make a prediction
    def think(self, neuron_inputs):
        # First step is to calculate the weighted sum of the inputs
        neuron_output = []

        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        # And then we need to take that sum of the weighted inputs
        # and pass it through our activation function which is the sigmoid

        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            print("Iteration: ", iteration + number_of_iterations * 2)
            for training_set_example in training_set_examples:

                # Predict the output based on the training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                # Calculate the error as the difference between the desired output and the predicted output

                error_in_output = training_set_example["output"][2] - predicted_output
                    # Iterate through the weights and adjust each one
                for index in range(len(self.weights)):

                    # Get the neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    # Calculate how much to adjust the weights by using the delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    # Adjust the weight
                    self.weights[index] += adjust_weight


	# Calculate the sigmoid (our activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):

        return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
#END OF CLASS



class NeuralNetwork4():
    # 6 methods for this class: 4 of them are private, indicated by
    # the prefix of double underscore "__" and 2 are public which we
    # will be able to  call externally

    def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		# For testing and debugging purposes
		# random.seed(1)

		# When the NeuralNetwork is initialized we want it to have 3 weights
        # because we will have 3 inputs coming into the NN
		# Create 3 weights and set them to random values in the range -1 to +1
        self.weights = []
        for i in range(300):
            self.weights.append(random.uniform(-1, 1))

    # Make a prediction
    def think(self, neuron_inputs):
        # First step is to calculate the weighted sum of the inputs
        neuron_output = []

        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        # And then we need to take that sum of the weighted inputs
        # and pass it through our activation function which is the sigmoid

        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            print("Iteration: ", iteration + number_of_iterations * 3)
            for training_set_example in training_set_examples:

                # Predict the output based on the training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                # Calculate the error as the difference between the desired output and the predicted output

                error_in_output = training_set_example["output"][3] - predicted_output
                    # Iterate through the weights and adjust each one
                for index in range(len(self.weights)):

                    # Get the neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    # Calculate how much to adjust the weights by using the delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    # Adjust the weight
                    self.weights[index] += adjust_weight


	# Calculate the sigmoid (our activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):

        return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
#END OF CLASS


class NeuralNetwork5():
    # 6 methods for this class: 4 of them are private, indicated by
    # the prefix of double underscore "__" and 2 are public which we
    # will be able to  call externally

    def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		# For testing and debugging purposes
		# random.seed(1)

		# When the NeuralNetwork is initialized we want it to have 3 weights
        # because we will have 3 inputs coming into the NN
		# Create 3 weights and set them to random values in the range -1 to +1
        self.weights = []
        for i in range(300):
            self.weights.append(random.uniform(-1, 1))

    # Make a prediction
    def think(self, neuron_inputs):
        # First step is to calculate the weighted sum of the inputs
        neuron_output = []

        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        # And then we need to take that sum of the weighted inputs
        # and pass it through our activation function which is the sigmoid

        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            print("Iteration: ", iteration + number_of_iterations * 4)
            for training_set_example in training_set_examples:

                # Predict the output based on the training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                # Calculate the error as the difference between the desired output and the predicted output

                error_in_output = training_set_example["output"][4] - predicted_output
                    # Iterate through the weights and adjust each one
                for index in range(len(self.weights)):

                    # Get the neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    # Calculate how much to adjust the weights by using the delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    # Adjust the weight
                    self.weights[index] += adjust_weight


	# Calculate the sigmoid (our activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):

        return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
#END OF CLASS


class NeuralNetwork6():
    # 6 methods for this class: 4 of them are private, indicated by
    # the prefix of double underscore "__" and 2 are public which we
    # will be able to  call externally

    def __init__(self):

		# Seed the random number generator, so we get the same random numbers each time
		# For testing and debugging purposes
		# random.seed(1)

		# When the NeuralNetwork is initialized we want it to have 3 weights
        # because we will have 3 inputs coming into the NN
		# Create 3 weights and set them to random values in the range -1 to +1
        self.weights = []
        for i in range(300):
            self.weights.append(random.uniform(-1, 1))

    # Make a prediction
    def think(self, neuron_inputs):
        # First step is to calculate the weighted sum of the inputs
        neuron_output = []

        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        # And then we need to take that sum of the weighted inputs
        # and pass it through our activation function which is the sigmoid

        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

	# Adjust the weights of the neural network to minimise the error for the training set
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            print("Iteration: ", iteration + number_of_iterations * 5)
            for training_set_example in training_set_examples:

                # Predict the output based on the training set example inputs
                predicted_output = self.think(training_set_example["inputs"])

                # Calculate the error as the difference between the desired output and the predicted output

                error_in_output = training_set_example["output"][5] - predicted_output
                    # Iterate through the weights and adjust each one
                for index in range(len(self.weights)):

                    # Get the neuron's input associated with this weight
                    neuron_input = training_set_example["inputs"][index]

                    # Calculate how much to adjust the weights by using the delta rule (gradient descent)
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

                    # Adjust the weight
                    self.weights[index] += adjust_weight


	# Calculate the sigmoid (our activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

	# Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self, neuron_output):
        
        return neuron_output * (1 - neuron_output)

	# Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self, neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
#END OF CLASS



#Make the NN work with 300 images
pixel_images = PixelImages(300)
oracleList = pixel_images.oracle(pixel_images.newImage())
input_values = pixel_images.getRGBValues()

#Call our Class Object
neural_network1 = NeuralNetwork1()
neural_network2 = NeuralNetwork2()
neural_network3 = NeuralNetwork3()
neural_network4 = NeuralNetwork4()
neural_network5 = NeuralNetwork5()
neural_network6 = NeuralNetwork6()


print("Random starting weights: " + str(neural_network1.weights))

# Training sets that are stored in a list and each item in the list
# contains a dictionary with the examples input and output
training_set_examples = []
for i in range(pixel_images.number_of_images - 1):
    training_set_examples.append({"inputs": input_values[i], "output": oracleList[i]})

# Train the neural network using 300 iterations
number_of_iterations = 50

neural_network1.train(training_set_examples, number_of_iterations)
neural_network2.train(training_set_examples, number_of_iterations)
neural_network3.train(training_set_examples, number_of_iterations)
neural_network4.train(training_set_examples, number_of_iterations)
neural_network5.train(training_set_examples, number_of_iterations)
neural_network6.train(training_set_examples, number_of_iterations)


# print("New weights after training are: " + str(neural_network1.weights))

# Make a prediction for a new situation
new_situation = input_values[pixel_images.number_of_images - 1]
prediction1 = neural_network1.think(new_situation)
prediction2 = neural_network2.think(new_situation)
prediction3 = neural_network3.think(new_situation)
prediction4 = neural_network4.think(new_situation)
prediction5 = neural_network5.think(new_situation)
prediction6 = neural_network6.think(new_situation)

# gathers all of the answers from each output neuron and throws it on a list
prediction_output = [prediction1, prediction2, prediction3, prediction4, prediction5, prediction6]

print("Prediction for the new situation " + str(new_situation))
print()
print(" -->" + str(prediction_output))
