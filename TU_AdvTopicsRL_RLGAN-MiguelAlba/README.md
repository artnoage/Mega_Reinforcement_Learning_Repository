# Seminar: Advanced topics in Reinforcement learning 
(I will add more plots and some code formatting fixes, soon)

## Paper: Transfer learning for Reinforcement learning tasks via Image-to-Image Translation (RL-GAN)
Main paper can be found in [Shani Gamrian, Yoav Goldberg, "Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation](https://arxiv.org/pdf/1806.07377.pdf)
This is only the implementation of RL-GAN for the game road fighter, the aim is to transfer the learned agent from level one to the other 4 without using common fine tunning but instead imitation learning and image-to-image translation.

![Road fighter levels](https://github.com/miguelalba96/TU_AdvancedTopicsRL_RLGAN/blob/master/images/main_levels.png)
(Figure 4, Original paper)
## Modules
The code consists of one main module and the GANs, one of the original implementation of UNIT GAN [UNIT](https://github.com/mingyuliutw/UNIT) and the other is a custom version of CycleGAN in TF 2.0 (Low level API) with slightly modifications in terms of loss compositions and in Tensorflow, the agent implementation is the famous A2C implementation of [pytorch-A2C](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) modified for the current problem, it is the Synchronous version of A3C.   
 
## Start 
In order to use transfer learning based on GAN generations an d imitation learning these are the steps: 
* clone and install [Special Retro with Road Fighter](https://github.com/ShaniGam/retro)  from authors of the paper
* It  requires `gym==0.10.5`. Updated to work with latest torch version, torchvision version.

1. It is required to train the initial agent as much as needed, between 110M-150M time steps is enough to get a median score of 10000, if a saving directory is not specified it saves the trained agent in the repository 
  
`python -m ROAD_FIGHTER.main --level 1 --save --num-processes 84`

![Training score](https://github.com/miguelalba96/TU_AdvancedTopicsRL_RLGAN/blob/master/images/level_1_roadfighter.png)


2. Once trained the first agent in level 1, we need to collect samples from the game to translate them between level domains, 100k images are enough according to the paper and original implementation. They are saved as numpy arrays 

`python -m ROAD_FIGHTER.main --level 1 --collect-images --num-collected-imgs 100000 --num-processes 2 --load`

`python -m ROAD_FIGHTER.main --level 2 --collect-images --num-collected-imgs 100000 --num-processes 2`

3. The way of training the GANs is different depending of the type of architecture, in the common case to run it with UNIT we train it from both folders containing images of level 1 and x

`python -m unit.train --trainer UNIT --config unit/configs/roadfighter-lvl2.yaml
`
For cycleGAN the model requires an input pipeline optimization, first we start saving both image folders as shards of tensorflow records. First thing is running:

`python CYCLEGAN/preprocessing.py`

This will create a folder containing a tf record dataset with separated shards between lvl-1 and lvl-x, In order to train the desired level change the config dictionary in `CYCLEGAN/train.py`, to train the GAN:

`python CYCLEGAN/train.py`

![Generation](https://github.com/miguelalba96/TU_AdvancedTopicsRL_RLGAN/blob/master/images/gen_lvl2_lvl1.jpg)
(top: lvl1, mid: generation, bottom: lvl2)

4. Then in order to apply the transfer learning from one level to the other you need to run 

`python -m ROAD_FIGHTER.main_imitation --load --gan-dir roadfighter-lvl2 --gan-imitation-file '00198000' --log-name lvl2.log --super-during-rl --level 2 --det-score 5350`

wrn: This part uses by default 30k steps by trajectory which can overload RAM!
This line starts the imitation learning process, applying the algorithm described in the paper, first it creates an object to save trajectories of using images from the original game and translations, it runs the game applying the pretrain a2c to the translated state observations and saving the original states.

There are required 5 general trajectories according to the paper, a transition with T as a trajectory buffer, T\leftarrow T \union (s_t, a_t, r_t), if in the given trajectory r_t (inmediate reward) > 5350 * 0.75, the transitions of the trajectory are saved with their respectives accumulated discounted rewards in a new buffer D.

Once there are enough trajectories, there is a switch to supervised learning on D where the real observation is taken as feature and the label is the correspoing action, the objective is minimize the categorical cross entropy between the actions took by the pretrain actor on the fake data and the actions taken by the random initialized actor. 

then in `main()` it is repeated again the reinforcement learning training, on policy updates based on the a2c loss and off policy updates based on the supervised learning `super_during_rl`, if the mean reward of all agents is less than 0.6 * the accumulared reward of all the trajector, the supervised learning starts again.

All credits to the authors and original implementation of [Shani Gamrian](https://github.com/ShaniGam/RL-GAN).

