# Tutorial for using CART Decision Trees.

This tutorial goes through the use of the implementation of decision trees provided by scikit-learn on the Iris dataset and the [PIMA-INDIAN Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database/data).

Note that this tutorial uses the `graphviz` python modules. It is included in the `requirements.txt` file, so you can install it by running the following in from the current folder:

```pip install -r requirements.txt```

or 

```conda install --file requirements.txt``` 

if you are using `Anaconda`.

Otherwise, you should be able to install it individually using:

```pip install graphviz```

or 

```conda install -c anaconda graphviz```.

If you installed using the methods described above, you might get the following error when you try to plot a tree:

```
RuntimeError: failed to execute ['dot', '-Tpdf', '-O', 'test'], make sure the Graphviz executables are on your systems' path
```

Based on the advice given by  [StackOverflow page](https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft), you can do the following to solve the issue based on your operating system.

## MacOS X
Install `Graphviz` using [`Homebrew`](https://brew.sh). Do this by running:

```
brew install graphviz
```

## Linux
Install `Graphviz` using `apt-get` by:

```
sudo apt-get install graphviz
```

## Windows
For Windows, the following solution worked.

Download the source `graphviz` files for windows as a zip folder [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).

Move unzipped folder to somewhere safe.

Finally, run the following lines at the top of the python notebook.

```
import os
os.environ["PATH"] += os.pathsep + '<PATH/TO/GRAPHVIZ/FOLDER>/bin/'
```

Make sure to replace `<PATH/TO/GRAPHVIZ/FOLDER>` with the correct path to where you moved the `graphviz-2.38` folder. Note that this folder name is the current version at the time of writing this README; it could have changed if you are reading this from the distant future.

