# 100DaysofML-Day10

### Simple RNN Text Generation with TensorFlow and Keras

This project demonstrates how to implement a simple RNN (Recurrent Neural Network) for text generation using TensorFlow and Keras. The model takes a seed text as input and generates new text based on the patterns it has learned during training.

### Table of Contents

- Getting Started
- Dataset
- Model
- Training
- Text Generation
- Testing
- Improvements

### Getting Started

You'll need Python and the following packages installed:

    pip install numpy
    pip install tensorflow

### Dataset

For this example, we'll use a simple dataset containing one paragraph of text. You can replace this with a larger dataset to achieve better results.


    text = "I love machine learning. It is breathtaking! Deep learning and natural language processing are pretty phenomenal."

### Model

We'll create a simple RNN model using Keras, a high-level neural networks API that works on top of TensorFlow. The model consists of an input layer, a single LSTM layer, and a dense output layer with a softmax activation function.

    model = Sequential()
    model.add(LSTM(units, input_shape=(sequence_length, 1)))
    model.add(Dense(len(chars), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

### Training

We train the model on the dataset, using a sliding window approach to create input-output pairs from the text. The model learns to predict the next character in the sequence, given the previous characters.

    for epoch in range(epochs):
        for i in range(len(input_sequences)):
            input_seq = input_sequences[i]
            output_char = output_chars[i]
            model.fit(input_seq, output_char, epochs=1, batch_size=1, verbose=2)

### Text Generation

After training, we can generate new text using the generate_text function. The function takes a seed text and generates a specified number of characters based on the seed text and the learned patterns.

    seed_text = "I love mach"
    generated_text = generate_text(seed_text, 50)
    print("Generated text:", generated_text)

### Testing

To ensure our code is working as expected, we can create a simple unit test that checks if the length of the generated text is equal to the specified number of characters.

    def test_generate_text():
        seed_text = "I love mach"
        num_chars = 50
        generated_text = generate_text(seed_text, num_chars)
        assert len(generated_text) == len(seed_text) + num_chars

### Improvements

This is a basic example, and there is room for improvement. To enhance the quality of the generated text, consider:

    Using a larger dataset
    Training the model for more epochs
    Experimenting with more complex architectures (e.g., LSTM or GRU networks)
    Fine-tuning the temperature parameter in the generate_text function to control the level of creativity

Happy text generation!
