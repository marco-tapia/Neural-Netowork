TO RUN: simply run the whichever python file you want (the img NN or the basic NN).
For the image one, check back with the images folder to check the result
with "image.jpg", as that is the image that the NN is predicting as to 
what color is the moset prominent

-----------------------------------------------------------------------------------

The idea behind this project was to dive into the very basics of how
Artificial Intelligence works by programming some neural networks from
scratch (no machine learning libraries like tensor flow).

The project is split up into two sections, one is a simple neural network
with 3 inputs and an output, the other is a neural network that works with
images.

The second part is and what I spent most of my time working on, and
it goes through and grabs every RGB value of each pixel and feeds
it into the NN as input and returns an output list that tells with pixel
color was the most prominent. list[0] indicates Red, list[1] indicates
Yellow, list[2] = Blue, list[3] = Orange, list[4] = Purple, and list[5] =
Green. If the number one of the indices is equal to 1, it means it was the
most prominent. For the values representing zero, it will show as an exponent
that is extremly close to zero.

Be aware of the fact that when you run this program in will create 300,
10x10 pixel images in an 'images' folder within the same directory that
the file is placed, as the samples to train the NN with.

Running the program should only take about 8 seconds, there is a print
statement that comes out for every Iteration, right now the program is set
to 50 iterations (300 total, 50 for each output neuron) to cut down on
processing time, but should still produce good results.

Im not very proud of the way I solved the last bit of this project,
but for times sake it was a good enough solution. It was to have the
program work with multiple outputs rather than a single output.
The way i did this was pretty much taking all those single outputs
and then mashing them all together in the end by running the NN Class
multiple times.
