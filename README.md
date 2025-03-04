# COBOL Neural Network for MNIST Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This project demonstrates that COBOL isn't just for legacy banking systems! I've implemented a fully-functional two-layer neural network to classify handwritten digits from the MNIST dataset. The model processes 28×28 grayscale images (784 pixels) and predicts digits 0-9 with surprising accuracy.

*Why COBOL?* Because sometimes the best way to understand modern concepts is to implement them in an unexpected environment. Also, it makes for a great conversation starter in tech interviews!

## 🧠 Network Architecture

The neural network features a simple yet effective structure:

- **Input Layer**: 784 neurons (one per pixel)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation

All implemented in pure COBOL without external machine learning libraries.

## 📈 Training Performance

| Epoch | Accuracy |
|-------|----------|
| 1     | 88.48%   |
| 2     | 90.09%   |
| 3     | 90.90%   |
| 4     | 91.23%   |
| 5     | 91.47%   |

Not bad for a language designed in 1959!

## 🚀 Getting Started

### Prerequisites

- GnuCOBOL (OpenCOBOL) compiler
- MNIST dataset in CSV format

### Dataset Preparation

The MNIST dataset can be downloaded from [GTDLBench's MNIST datasets page](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/). Download the CSV versions for easier integration with this COBOL implementation.

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/PierreRemacle/cobol-neural-network.git
   cd cobol-neural-network
   ```

2. Ensure your COBOL environment is properly configured.

3. Compile the main program:
   ```
   cobc -x -o mnist_nn Mnist.cob
   ```

### Usage

Run the executable:
```
./mnist_nn
```

## 📁 Project Structure

```
├── Mnist.cob         # Main MNIST neural network implementation
├── README            # Original README file
├── hello             # Example executable
├── hello.cob         # Example "Hello World" COBOL program
└── README.md         # This improved README
```

## 🔧 How It Works

The program follows these steps:
1. Loads MNIST data from CSV files
2. Initializes network weights with small random values
3. For each training epoch:
   - Performs forward propagation
   - Calculates loss
   - Performs backpropagation
   - Updates weights
4. Evaluates accuracy on test set

## 💡 Implementation Details

This implementation started with CSV file reading code based on [Queen of COBOL's tutorial](https://queenofcobol.com/reading-a-csv-file/) and expanded to include neural network functionality.

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

## 📚 Resources

- [GTDLBench MNIST Datasets](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
- [Queen of COBOL's CSV Tutorial](https://queenofcobol.com/reading-a-csv-file/)
- [GnuCOBOL Documentation](https://gnucobol.sourceforge.io/)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🧩 Future Improvements

- Add more hidden layers
- Implement mini-batch gradient descent
- Add dropout for regularization
- Create visualization tools for results
