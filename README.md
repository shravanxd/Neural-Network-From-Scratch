# Neural Network from Scratch (Non-AI)

This project implements a simple **feedforward neural network** from scratch using **only NumPy**, without using any AI/ML frameworks like TensorFlow or PyTorch.

## **Features**
- Fully connected **2-layer neural network**
- **Manual forward propagation**
- **Manual backpropagation** using gradient descent
- **Sigmoid activation function**
- Designed to work on small datasets (e.g., XOR problem)

## **Files**
- `neural_network_from_scratch.py` – Main Python script that implements the neural network.

## **How It Works**
The neural network consists of:
1. **Input layer** (2 neurons for XOR input)
2. **Hidden layer** (3 neurons)
3. **Output layer** (1 neuron with sigmoid activation)

It is trained using:
- **Forward propagation** to compute activations.
- **Backpropagation** to update weights using gradient descent.

## **Installation**
### **Prerequisites**
- Python 3.12.3
- NumPy

### **Setup**
1. Clone the repository:
   ```sh
   git clone https://github.com/shravanxd/Neural-Network-From-Scratch.git
   cd Neural-Network-From-Scratch

2. Install dependencies:
   pip install numpy

3. Run the Script:
   python neural_network_from_scratch.py   
