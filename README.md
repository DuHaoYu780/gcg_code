Markdown

# Research on Visual SLAM in Low Illumination Environment Based on Image Contrast Enhancement

This repository contains the C++ implementation of the image enhancement algorithms proposed in our paper, along with the trajectory evaluation results on the EuRoC dataset.

## 1. Prerequisites 

The code has been tested on **Ubuntu 20.04**. Please ensure the following dependencies are installed:

* **C++ Compiler:** C++11 or higher
* **CMake:** ≥ 2.8
* **OpenCV:** ≥ 4.0 (Required components: core, imgproc, features2d)
* **Eigen3:** ≥ 3.1.0
* **Pangolin:** (Required for UI dependencies)

## 2. Build Instructions 

Clone the repository and compile the project using CMake:

```bash
cd your_project_folder
mkdir build
cd build
cmake ..
make
```

After successful compilation, an executable named test_app will be generated in the build directory.

## 3. Image Enhancement Test 
This program demonstrates how our method improves ORB feature extraction in low-illumination environments.

Usage
To run the test program, provide the path to a low-illumination image as an argument:

```bash
# Syntax: ./test_app <path_to_image>
# Example:
./test_app ../1.png
```

The program will display the original image vs. enhanced images (Gamma, CLAHE, Gaussian Blur) and output the processing time for each algorithm.

## 4. Trajectory Evaluation 
To verify the localization accuracy (ATE) reported in the paper, we provide the estimated trajectory files in the Euroc/ directory.

We use the evo package for evaluation.

### 4.1 Install evo

```Bash
pip install evo --upgrade --no-binary evo
```

### 4.2 Run Evaluation (MH_01 Sequence)
The following command compares our estimated trajectory (MH01_trajectory.txt) against the ground truth (groundtruth.tum).

```Bash
# Syntax: evo_ape tum <ground_truth_file> <estimated_trajectory_file> -va --plot
evo_ape tum Euroc/MH01/state_groundtruth_estimate0/groundtruth.tum Euroc/MH01/state_groundtruth_estimate0/MH01_trajectory.txt -va --plot
```

>**Note:**

MH01_trajectory.txt: The trajectory estimated by our proposed method using enhanced images.

groundtruth.tum: The ground truth trajectory provided by the EuRoC dataset.

The results demonstrate the accuracy improvement in low-illumination scenarios.

### 4.3 Comparison with Baseline (Optional)
We also provide the trajectory results of the original ORB-SLAM3 (`ORBSLAM3.txt`) in the same directory. You can run the following command to compare our method against the baseline:

```bash
evo_ape tum Euroc/MH01/state_groundtruth_estimate0/groundtruth.tum Euroc/MH01/state_groundtruth_estimate0/ORBSLAM3.txt -va
```

5. License
This source code is released under the MIT License.
