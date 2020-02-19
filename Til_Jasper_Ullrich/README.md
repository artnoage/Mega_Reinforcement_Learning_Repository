If you just want to see the results, you can run `jupyter notebook Neural-Episodic-Critic.ipynb`.

If you want to execute the notebook, use the following instructions.
1. Use a good computer or server. Due to high variance, a larger number of experiments are required.
   The notebook took around 6 hours to execute on a 16 hyperthread-server.
   This number is recommended as it is equal to the number of experiments which are
   executed in parallel. More will not make it much faster due to the k-d tree not being parallel
   but less cores should be ok if you are ok with longer running time.
2. Install openmpi, which is usually available with your distributions package manager.
3. Look at `run.sh` for further instructions. The script sets up a virtual environment
   and installs the required packages in there. You might be able to just run the script but
   it would probably be useful to look at the comments to make sure the script works for you.
