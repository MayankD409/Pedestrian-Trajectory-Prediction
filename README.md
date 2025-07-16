# Pedestrian Trajectory Prediction with Robot Awareness

A deep learning framework for predicting pedestrian trajectories in crowded environments using Spatial-Temporal Graph Convolutional Networks (ST-GCNN) combined with robot trajectory information.

## 🚀 Overview

This project implements a novel approach to pedestrian trajectory prediction that combines:
- **Spatial-Temporal Graph Convolutional Networks (ST-GCNN)** for modeling pedestrian social interactions
- **Bidirectional RNN/LSTM/GRU** for incorporating robot trajectory information  
- **Temporal Convolutional Networks (TXP-CNN)** for future trajectory generation
- **Bivariate Gaussian loss** for probabilistic trajectory prediction

The model predicts future pedestrian trajectories by considering both social interactions among pedestrians and the influence of robot movements in the environment.

## 📋 Features

- **Multi-modal Architecture**: Combines graph neural networks with recurrent networks
- **Robot-Aware Prediction**: Incorporates robot trajectory information to influence pedestrian behavior
- **Social Interaction Modeling**: Uses graph structures to capture pedestrian-pedestrian interactions
- **Probabilistic Outputs**: Generates trajectory distributions rather than deterministic paths
- **Multiple Dataset Support**: Works with ETH, Hotel, University, Zara1, and Zara2 datasets
- **Comprehensive Evaluation**: Includes ADE, FDE metrics and collision detection
- **Visualization Tools**: Generates trajectory plots for analysis
- **TensorBoard Integration**: Real-time training monitoring

## 🏗️ Architecture

The model consists of three main components:

### 1. ST-GCNN (Spatial-Temporal Graph Convolutional Network)
- Processes pedestrian trajectories and their spatial relationships
- Uses normalized Laplacian adjacency matrices for graph convolution
- Applies temporal convolution for sequence modeling

### 2. Bidirectional RNN/LSTM/GRU
- Processes robot trajectory information across the entire sequence
- Provides context about robot behavior to influence pedestrian predictions
- Supports multiple RNN variants (LSTM, GRU, vanilla RNN)

### 3. TXP-CNN (Temporal Convolutional Network)
- Fuses pedestrian and robot information
- Generates future trajectory predictions
- Outputs bivariate Gaussian parameters (μ, σ, ρ)

## 📊 Datasets

The project supports multiple benchmark datasets:

- **ETH**: Pedestrian trajectories in urban environments
- **Hotel**: Indoor pedestrian movement patterns
- **University**: Campus pedestrian trajectories
- **Zara1/Zara2**: Shopping mall pedestrian data

**Data Format**: Tab-separated values with columns: `<frame_id> <ped_id> <x> <y>`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/MayankD409/Pedestrian-Trajectory-Prediction.git
cd Pedestrian-Trajectory-Prediction
```

2. Install dependencies:
```bash
pip install torch torchvision numpy scipy matplotlib networkx tensorboardX tqdm
```

## 📖 Usage

### Training

Train the model on a specific dataset:

```bash
python train.py --dataset eth --n_stgcnn 1 --n_txpcnn 3 --num_epochs 250 --tag experiment_name
```

**Key Parameters:**
- `--dataset`: Dataset to use (eth, hotel, univ, zara1, zara2)
- `--n_stgcnn`: Number of ST-GCNN layers
- `--n_txpcnn`: Number of TXP-CNN layers
- `--obs_seq_len`: Observation sequence length (default: 8)
- `--pred_seq_len`: Prediction sequence length (default: 12)
- `--batch_size`: Training batch size (default: 128)
- `--lr`: Learning rate (default: 0.01)
- `--tag`: Experiment identifier for logging

### Testing

Evaluate the trained model:

```bash
python test.py
```

The test script will:
- Load all trained models from the checkpoint directory
- Evaluate on test datasets
- Generate trajectory visualizations
- Calculate ADE and FDE metrics
- Perform collision detection analysis

### Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir=./checkpoint/[experiment_name]/logs
```

## 📈 Metrics

The model is evaluated using standard trajectory prediction metrics:

- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance between predicted and ground truth final positions
- **Collision Detection**: Analysis of potential robot-pedestrian collisions

## 🔧 Model Configuration

### Architecture Variants

The model supports different RNN architectures:
- **BiLSTM**: Bidirectional LSTM (default hidden size: 32)
- **BiGRU**: Bidirectional GRU (default hidden size: 16)  
- **BiRNN**: Bidirectional vanilla RNN (default hidden size: 32)

### Hyperparameters

Key hyperparameters can be tuned:
- `input_size`: Input feature dimension (default: 2 for x,y coordinates)
- `output_size`: Output feature dimension (default: 5 for bivariate Gaussian)
- `kernel_size`: Convolutional kernel size (default: 3)
- `clip_grad`: Gradient clipping threshold
- `lr_sh_rate`: Learning rate decay schedule

## 📁 Project Structure

```
├── model.py              # Neural network architectures
├── train.py              # Training script
├── test.py               # Testing and evaluation script
├── utils.py              # Data processing utilities
├── utils_newds.py        # Additional dataset utilities
├── metrics.py            # Evaluation metrics
├── datasets/             # Dataset storage
│   ├── eth/
│   ├── hotel/
│   ├── univ/
│   ├── zara1/
│   └── zara2/
├── checkpoint/           # Model checkpoints and logs
└── README.md
```

## 🎯 Key Components

### Data Processing
- **Graph Construction**: Builds spatial graphs from pedestrian positions
- **Trajectory Normalization**: Converts absolute to relative coordinates
- **Sequence Padding**: Handles variable-length trajectories
- **Robot Integration**: Incorporates robot trajectory information

### Loss Function
- **Bivariate Gaussian Loss**: Models trajectory uncertainty
- **Negative Log-Likelihood**: Probabilistic loss formulation
- **Numerical Stability**: Includes epsilon clamping for training stability

### Evaluation
- **Multi-sample Prediction**: Generates multiple trajectory samples
- **Best Sample Selection**: Chooses optimal prediction from samples
- **Visualization**: Creates trajectory plots for analysis

## 🔬 Experimental Results

The model demonstrates improved performance by incorporating robot trajectory information:
- Enhanced prediction accuracy in robot-pedestrian interaction scenarios
- Better handling of complex social dynamics
- Robust performance across multiple benchmark datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ETH Pedestrian Dataset creators
- ST-GCN implementation reference
- PyTorch community for deep learning framework
- TensorBoard for visualization tools

## 📞 Contact

For questions or issues, please open an issue in the repository or contact [your-email@example.com].

---

**Note**: This project is part of research in pedestrian trajectory prediction and robot-human interaction modeling. 