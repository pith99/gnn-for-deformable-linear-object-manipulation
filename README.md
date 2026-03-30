# Graph Neural Networks for Robotic Manipulation of Deformable Linear Objects
This project investigates the use of Graph Neural Networks (GNNs) to model and predict the dynamics of Deformable Linear Objects (DLOs), such as cables, during robotic manipulation. The system uses image processing to extract keypoints from recorded cable motion, converts them into graph-structured temporal data, and trains GNN-based models to predict future DLO shapes. Two model variants were explored: a GCN-LSTM model and an Attention Temporal GCN model.

## Objectives
- Build a dataset of DLO movements using image processing
- Represent DLO states as graph-structured temporal data
- Train GNN-based models to predict future DLO keypoint positions
- Evaluate the prediction performance of different architectures

  ## Problem Formulation
The task is to predict the future shape of a deformable linear object using its historical shape. The DLO is represented by 11 equidistant keypoints, and the model predicts future `(x, y)` coordinates of these keypoints from previous timesteps.

## Dataset
The dataset was created using videos of a charging cable marked with 11 red tape keypoints. The left end of the cable was fixed, while the right end was moved in a 2D plane. Movements were grouped into three classes:
- x-axis movement
- y-axis movement
- diagonal movement

Videos were processed to extract keypoint coordinates for each frame, producing graph-structured temporal data for training and evaluation.

## Preprocessing Pipeline
The raw video data is processed through the following pipeline:
1. Detect red keypoint markers using OpenCV contour extraction
2. Select the 11 most relevant contours
3. Track keypoints across frames using proximity-based matching
4. Interpolate missed or obstructed keypoints
5. Re-center coordinates using the stationary keypoint as the origin
6. Apply min-max normalization
7. Convert sequences into graph-based temporal input-target batches

## Models
### 1. GCN-LSTM
A Graph Convolutional Network is used to encode spatial relationships between DLO keypoints, followed by an LSTM to learn temporal dependencies across timesteps.

### 2. Attention Temporal GCN
An A3T-GCN-based architecture is used to model both spatial and temporal dependencies, with an attention mechanism to emphasize more relevant historical states.

## Experimental Setup
- Train-test split: 80:20
- Graph nodes: 11 keypoints
- Node features: `(x, y)` coordinates
- Edge weights: uniform
- Frameworks: PyTorch, PyTorch Geometric, OpenCV, NumPy

## Results
The models were evaluated on their ability to predict future DLO keypoint positions across multiple movement configurations. Experiments included short-horizon prediction and longer-horizon recursive prediction. The Attention Temporal GCN showed stronger capability in capturing spatio-temporal dependencies in DLO motion.

## Author
Arpith Koshy

## Acknowledgements
Supervised by A/P Cheah Chien Chern.
Special thanks to Zhao Xinge for guidance and support throughout the project.
