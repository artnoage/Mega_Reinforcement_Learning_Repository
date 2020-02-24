#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

#import base64
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import time
import csv


from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common



tf.compat.v1.enable_v2_behavior()

# HYPERPARAMETERS
num_iterations = 10000
collect_steps_per_iteration = 5
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 10  
eval_interval = 200  

curiosity_param = 1


# other parameters
augment_reward = True
plot = False

# can be used for hyperparametersearch
NEURONS_FIRST_HLAYER=[15]
NEURONS_SECOND_HLAYER=[5]
CURIOSITIY_PARAM=[1]





# INTERACT WITH ENVIRONMENT  # 


# used for evaluation
def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    # play game until finished
    while not time_step.is_last():
      # execute action with current state & policy
      action_step = policy.action(time_step)
      # how did actoin affect env?
      time_step = environment.step(action_step.action)
      # what reward did this generate
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# used in training
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)


def testRandomPolicy(number_of_times):

	train_env, eval_env = createEnvs()

	random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                            train_env.action_spec())

	for _ in range(number_of_times):
		print(compute_avg_return(eval_env, random_policy, num_eval_episodes))





# CREATE ESSENTIAL ARCHITECTURE FOR TF-AGENTS WORKFLOW  # 


def createEnvs():
  env_name = 'CartPole-v0'
  env = suite_gym.load(env_name)

  # Usually two environments are instantiated: one for training and one for evaluation.
  train_py_env = suite_gym.load(env_name)
  eval_py_env = suite_gym.load(env_name)

  # The Cartpole environment, like most environments, is written in pure Python. This is converted to TensorFlow using the TFPyEnvironment wrapper.
  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
  
  return train_env, eval_env


def createAgent(train_env):
  # Use tf_agents.networks.q_network to create a QNetwork, passing in the observation_spec, action_spec, 
  # and a tuple describing the number and size of the model's hidden layers.
  fc_layer_params = (100,)

  q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

  # Now use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. In addition to the time_step_spec, action_spec and the QNetwork, 
  # the agent constructor also requires an optimizer (in this case, AdamOptimizer), a loss function, and an integer step counter.
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

  train_step_counter = tf.Variable(0)

  agent = dqn_agent.DqnAgent(
	     train_env.time_step_spec(),
	     train_env.action_spec(),
	     q_network=q_net,
	     optimizer=optimizer,
	     td_errors_loss_fn=common.element_wise_squared_loss,
	     train_step_counter=train_step_counter)

  agent.initialize()

  return agent


def createReplayBuffer(agent, train_env):

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
	  data_spec=agent.collect_data_spec,										#trajectory-type
	  batch_size=train_env.batch_size,											#size of data added by add_batch method
	  max_length=replay_buffer_max_length)

  return replay_buffer

def createIterableDataset(replay_buffer):

  # The agent needs access to the replay buffer. This is provided by creating an iterable tf.data.Dataset pipeline which will feed data to the agent.
  # Each row of the replay buffer only stores a single observation step. But since the DQN Agent needs both the current and next observation to compute the loss, 
  # the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).
  dataset = replay_buffer.as_dataset(
	  num_parallel_calls=3, 
	  sample_batch_size=batch_size, 										
	  num_steps=2).prefetch(3)

  iterator = iter(dataset)

  return iterator






# FOR CURIOSITY AUGMENTATION # 


# creates the prediction network, which is then used to augment the reward, based on how well it predicted
def createPredictionNetwork(neurons_first_hidden_layer, neurons_second_hidden_layer):
  input_layer = Input(shape=(5,))
  first_hidden_layer = Dense(neurons_first_hidden_layer, activation='relu')(input_layer)
  second_hidden_layer = Dense(neurons_second_hidden_layer, activation='relu')(first_hidden_layer)
  output_layer = Dense(4, activation='sigmoid')(first_hidden_layer)
  pred_net = Model(input_layer, output_layer)
  pred_net.compile(optimizer='adam',loss='mean_squared_error')
  return pred_net

# extracts labeled data for the prediction network from a trajectory
def getObservationAndActions(experience):

	#observations:
	obs=experience[1].numpy()
	obs_1=obs[:,0,:]		#current state
	obs_2=obs[:,1,:]		#next state

	#actions:
	act=experience[2].numpy()
	act=act[:,0].astype(float)	#action performed in current state

	#concatenate current state & action
	obs_1_act = np.zeros((batch_size, 5))
	for i in range(batch_size):
		featurevector= np.append(obs_1[i,:],act[i])
		#print(obs_1[i,:])
		#print(act[i])
		#print(featurevector)
		
		obs_1_act[i,:]=featurevector

	return obs_1_act, obs_2

# augments reward in trajectory based on how bad the prediction network predicted the next state given an action
# high prediction error -> high reward for state-action tuple
# this should motivate the agent to explore unseen/unknown state-action tuples
def augmentReward(experience, pred_net):

	#compute augmented reward
	reward=experience[5].numpy()
	obs_1_act, obs_2 = getObservationAndActions(experience)
	predictions=pred_net.predict(obs_1_act, batch_size=10, verbose=0)
	mse=keras.losses.mean_squared_error(obs_2, predictions).numpy()

	augmented_reward=reward[:,0]+mse*curiosity_param       	#new reward = old reward + mse*constant
															

	#write augmented reward into experience
	rewardfiller= np.zeros((batch_size, 2))
	for i in range(batch_size):
		rewardvector= np.append(augmented_reward[i],reward[i,1])
		rewardfiller[i,:]=rewardvector

	rewardfiller = tf.convert_to_tensor(rewardfiller, dtype=tf.float32)

	experience = experience.replace(reward=rewardfiller)

	return experience








def train_eval(neurons_first_hidden_layer, neurons_second_hidden_layer, curiosity_param):

	#---------------------- CREATE ----------------------------------
	train_env, eval_env = createEnvs()

	agent = createAgent(train_env)

	replay_buffer = createReplayBuffer(agent, train_env)

	#execute the collect policy in the environment for a few steps, recording the data in the replay buffer.
	collect_data(train_env, agent.collect_policy, replay_buffer, steps=500)

	iterator = createIterableDataset(replay_buffer)

	pred_net=createPredictionNetwork(neurons_first_hidden_layer, neurons_second_hidden_layer)



	# (Optional) Optimize by wrapping some of the code in a graph using TF function.
	agent.train = common.function(agent.train)

	# Reset the train step
	agent.train_step_counter.assign(0)

	# Evaluate the agent's policy once before training.
	avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
	returns = [avg_return]

	# Evalute MSE for untrained prediction network.
	misprediction = []
	if augment_reward:
	    experience, unused_info = next(iterator)
	    obs_1_act, obs_2 = getObservationAndActions(experience)
	    predictions=pred_net.predict(obs_1_act, batch_size=10, verbose=0)
	    mse=keras.losses.mean_squared_error(obs_2, predictions)
	    mse=mse.numpy().sum()
	    average_mse=mse/batch_size
	    misprediction.append(average_mse)



	for _ in range(num_iterations):

	  # ---------- A) COLLECT ----------
	  # collect a few steps using collect_policy and save to the replay buffer.
	  for _ in range(collect_steps_per_iteration):
	    collect_step(train_env, agent.collect_policy, replay_buffer)

	  # ---------- B) TRAIN NNs ----------
	  # Sample a batch of data from the buffer and update the agent's network.
	  experience, unused_info = next(iterator)

	  #reward augmentation and training of pred_net
	  if augment_reward: 
	  	#augmentreward
	  	experience = augmentReward(experience, pred_net)


	  	obs_1_act, obs_2 = getObservationAndActions(experience)
	  	pred_net.fit(obs_1_act, obs_2,
                  epochs=5,
                  batch_size=4,
                  verbose=0)										#no output
		
	  # policy_net
	  train_loss = agent.train(experience).loss
	  step = agent.train_step_counter.numpy()

	  # ---------- C) Evaluate sometimes: ----------

	  if step % eval_interval == 0:

	    print(100*"-")
	  	#return
	    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
	    print('step = {0}: Average Return = {1}'.format(step, avg_return))
	    returns.append(avg_return)

	    #prediction-error
	    if augment_reward: 
	      collect_step(train_env, agent.collect_policy, replay_buffer)
	      experience, unused_info = next(iterator)
	      obs_1_act, obs_2 = getObservationAndActions(experience)
	      predictions=pred_net.predict(obs_1_act, batch_size=10, verbose=0)
	      mse=keras.losses.mean_squared_error(obs_2, predictions)
	      mse=mse.numpy().sum()
	      average_mse=mse/batch_size
	      print(100*"-")
	      print("Average MSE=")
	      print(average_mse)
	      print(100*"-")
	      misprediction.append(average_mse)


	# ---------- PLOT RESULTS----------
	iterations = range(0, num_iterations + 1, eval_interval)
	if plot:
	  plt.clf()
	  plt.plot(iterations,returns)
	  plt.ylabel('Average Return')
	  plt.xlabel('Iterations')
	  plt.ylim(top=250)
	  plt.show()

	  if False:
	    plt.clf()
	    plt.plot(iterations,misprediction)
	    plt.ylabel('Mean Squared Error')
	    plt.xlabel('Iterations')
	    plt.ylim(top=250)
	    plt.show()

	return returns, misprediction, iterations


def main():

	#testRandomPolicy(100)

	with open('metrics.csv', 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in NEURONS_FIRST_HLAYER:
			for j in NEURONS_SECOND_HLAYER:
				for k in CURIOSITIY_PARAM:
					filewriter.writerow([str(i),str(j),str(k)])
					start = time.time()
					average_return, misprediction, iterations = train_eval(i,j,k)
					end = time.time()
					measured_time= end-start
					print("measured_time")
					print(measured_time)
					filewriter.writerow([measured_time])
					filewriter.writerow(iterations)
					filewriter.writerow(average_return)
					filewriter.writerow(misprediction)
					filewriter.writerow("------------------------------------------")

if __name__ == '__main__':
  main()

