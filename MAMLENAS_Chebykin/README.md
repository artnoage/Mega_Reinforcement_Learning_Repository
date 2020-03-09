# An attempt to bring Neural Architecture Search to Few-shot learning by combing MAML and ENAS

Basic description of MAML (Model-Agnostic Meta-Learning) and ENAS (Efficient Neural Architecture Search):

1) MAML: train a network on a variety of tasks such that it would be able to quickly adapt to a new task. [https://arxiv.org/abs/1703.03400]

2) ENAS: train a controller network to generate a task-specific network architecture. [https://arxiv.org/abs/1802.03268]

The idea of this mini-project was to combine the two algorithms: train a controller network such that it would be able to quickly generate a network architecture for a new task.

This idea didn't work out. Maybe it was a bad idea, maybe I haven't found enough tricks and good hyperparameters to make it work.

The bigger part of the code was taken from the following repostories: https://github.com/oscarknagg/few-shot/ (MAML), https://github.com/TDeVries/enas_pytorch (ENAS). I am thankful to the authors, and can recommend these implementations as both well-written and actually working.

## Description of the approach

For starters, here's a decription of training loops of vanilla MAML and ENAS:

1) MAML: at the start of the outer loop a batch of tasks is sampled; in the inner loop a copy of a network is trained for a small number of steps on each task; at the end of the outer loop the network parameters are updated using the updated parameters from the inner loop.

2) ENAS: in each epoch firstly the controller parameters are frozen while parameters of child networks are trained, and then vice versa.

It is worth noting that there are 2 versions of MAML: for RL and for non-RL. RL version seems more fitting, since ENAS is optimized as a reinforcement learning problem. However, there is not really a chain of state-action-rewards to sample, because there's no real state, and the reward can only be given in the end. [reward is the loss of the generated model on the validation set] Additionaly, I aimed at Few-shot classification on MiniImageNet, a non-RL problem. So in the end I went with a combination of RL and non-RL MAML.

I faced a number of problems when combining MAML and ENAS. For example, ENAS has child networks' parameters trained for a whole epoch before the controller is updated. But in the case of few-shot learning we simply don't have enough samples to do that, and the task keeps changing from batch to batch. The only way to proceed is simultaneous learning of both networks.

Another problem is that to get ENAS reward for the currently generated network, we need to first train it on the training set, and then measure validation accuracy on the validation set. This is not a wasted computation due to weight sharing of child networks, but it does mean that we need to sample two times more data. Why? Because original MAML requires train+val for each iteration. With ENAS this becomes (train_train+train_val)+(val_train+val_val).

In the end, pseudocode of my approach is this:

```
for _ in range(epochs):
    sample batch of tasks {T_i} and data for them (2 train sets + 2 validation sets)
    for T_i in {T_i}:
        clone controller to controller_clone
        # trainining controller-clone:
        sample child_net
        train child_net on train1
        evaluate child_net on val1
        update controller_clone by PG with validation accuracy as reward
        # evaluating controller-clone:
        sample child_net
        train child_net on train2
        evaluate child_net on val2
        backpropagate through controller_clone by PG with validation accuracy as reward
        save gradients of controller_clone
    update controller with all the saved gradients of different controller_clones
    
```

## Running details

MiniImageNet (https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view) is needed to run the code. The easiest way to run the code would be to add the dataset file to the root folder of your google drive, upload the .ipynb file to google Colab, and then simply run it. Otherwise proper paths to data need to be set in the code.