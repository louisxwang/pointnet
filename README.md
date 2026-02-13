
#### This is a simplified implementation of the PointNet for 3D point cloud segmentation demo.

# Setup

### Option 1 Using uv
Under project root, run
```
uv sync
```
### Option 2 Using pip
Create a venv and install the dependencies with pip
```
pip install -r requirements.txt
```
# Usage
Under project root, activate the venv, then run
```
python main.py
```
By default, the test is run with pretrained model and 3D visualization. 
It is possible to manually choose train routine by adding --train option.


# Cuda
pytorch cpu version is the default dependency for maximum compability, but it is recommended to install pytorch with cuda support for faster training.