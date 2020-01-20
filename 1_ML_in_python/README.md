# Tutorial 1: Introduction to Machine Learning in Python

This tutorial contains exercise to get students familiar with implementing machine learning methods in python. 

In this, students are asked to use the IRIS dataset to gain experience in manipulating data in Pandas DataFrames, visualising the data in simple plots, and implementing the closed form solution of linear regression using least squares with NumPy.

## Setting up
N.B. This tutorial assumes you are using Python3.5 and above.
 
### Jupyter Notebooks

Jupyter Notebooks provide useful tools to learn and practice Machine Learning. It is a web application that allows you to organise your code into blocks, with the ability to execute one block at a time. In addition, you can plot graphs inline. As such, this is a valuable protyping tool.

This can be installed by running

```
pip3 install --upgrade pip
pip3 install jupyter
```

More details can be found [here.](https://jupyter.readthedocs.io/en/latest/install.html)
### Virtual Environments

We recommend the use of a Python virtual environment in order to sandbox the enviroment you will be using for this work. 

In order to do this, first install the `virtualenv` package by typing the following in your terminal. 

```pip3 install virtualenv```

If you are using Anaconda, please do

```conda install -c anaconda virtualenv```

Following this, navigate to **current folder** of the repository, wherever it is stored locally, and enter the following. 

```virtualenv venv```

This will create a new folder `venv` in your current directory, and will contain the necessary files to activate your new virtual environment. To do so, from the same location, type

```source ./venv/bin/activate```.

This process is explained in more detail [here.](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) 

### Installing required packages

Once you have activated your virtual environment, install the required packages using 

```pip install -r requirements.txt```

If you are using Anaconda, do

```conda install --file requirements.txt```

This will install the packages listed in `/requirements.txt`.

Finally, run the following to ensure that the current environment is available when you run the Jupyter Notebook.

```ipython kernel install --user --name=tutorial_1```

You can remove a kernel from by 

```jupyter kernelspec uninstall <name of kernel>```

In this case, `<name of kernel>` would be `tutorial_1`.

### Run the notebook

In the same folder, you can now run

```jupyter notebook```

This should start the jupyter server, which defaults to serving at `http://localhost:8888/`. Enter this in your browser. Then navigate to `1_ML_in_python.ipynb` to start the tutorial. 




