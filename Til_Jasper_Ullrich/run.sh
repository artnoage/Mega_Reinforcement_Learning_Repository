#!/bin/bash

venv_folder=nec-venv
notebook=Neural-Episodic-Critic.ipynb

[ -d $venv_folder ]
venv_exists=$?

if [ ! -f $notebook ] ; then
    echo $notebook not found. Please run ./run.sh from inside the folder, not outside.
    exit
fi

# The code has only been tested with python 3.6.10.
# Similar versions might work but for example 3.8.x does not have a binary tensorflow package yet
# which is required for spinningup (even though it is not used for this notebook).
# If you have another python version, you might want to try https://github.com/pyenv/pyenv

if ! python3 --version | grep -q 'Python 3.6' ; then
    echo $(tput setaf 1)This code has only been tested with Python 3.6.10.\
        Your version is $(python3 --version). It might not work.$(tput sgr0)
fi

# The packages are installed in a virtual environment.
# You can probably just use this script to install it.

if [ $venv_exists == 1 ] ; then
    python3 -m venv $venv_folder
fi

source $venv_folder/bin/activate

if [ $venv_exists == 1 ] ; then
    pip install notebook
    pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

    # It is important to install the packages from the folders since they have been modified
    pip install -e ./pyflann
    pip install -e ./spinningup

    # SpinningUp requires 0.8.1 but we need a newer version.
    # This breaks plotting functionality of SpinningUp but since we don't use it,
    # this is not problem for us.
    # This package has to be installed after SpinningUp.
    pip install seaborn==0.10.0
fi

# If you use a server to execute this notebook on, you might want to expose it to the outside.
# Otherwise, you can remove the --ip 0.0.0.0 (and also --port 8080 if you want)
jupyter notebook --ip 0.0.0.0 --port 8080 $notebook
