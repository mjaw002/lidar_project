# lidar_project
download and install anaconda

create a virtual enviornment (any name, in this case I have used lidar) with python 3.8

'conda create -n lidar python=3.8 anaconda'

switch to the new enviornment

'conda activate lidar'

install opencv in your new enviornment

'pip install opencv-contrib-python'


I have created a mask (mask.jpg). Please run the code by 

'python inpaint.py' 

and it will create a masked image in the data directory.
