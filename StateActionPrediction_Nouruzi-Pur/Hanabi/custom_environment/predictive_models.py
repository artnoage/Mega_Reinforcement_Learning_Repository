from __future__ import absolute_import, division, print_function, unicode_literals

import time

import sys
import numpy
import csv
numpy.set_printoptions(threshold=sys.maxsize)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras



class StatePredictor(object):

    def __init__(self, neurons_first_layer, neurons_second_layer, num_envs, curiosity_param):
       self.neurons_first_layer = neurons_first_layer
       self.neurons_second_layer = neurons_second_layer
       self.pred_net = self._createPredictiveModel()

       self.num_envs = num_envs

       self.curiosity_param = curiosity_param


    def augmentReward(self, trajectory):

      # get state & reward
      current_states_actions, next_states=self._getConsecutiveStateActionNextStateData(trajectory)
      rewards = self._getRewardsFromTrajectory(trajectory)

      #predict
      predictions=self.pred_net.predict(current_states_actions, batch_size=10, verbose=0)

      #augment reward in trajectory depending on mse
      mse = (numpy.square(next_states - predictions)).mean(axis=1)
      print("average mse: ", mse.sum()/len(next_states))
      env_mse = numpy.zeros((rewards.shape))

      counter = 0
      for i in range(self.num_envs):
          for j in range(len(env_mse[i])):
              featurevector=mse[counter]
              env_mse[i,j]=featurevector
              counter = counter+1

      augmented_rewards=rewards+env_mse*self.curiosity_param          #new reward = old reward + mse*constant

      augTrajectory = trajectory.replace(reward=augmented_rewards.astype(float))

      return augTrajectory




    def train(self, trajectory, evaluate):

      current_states_actions, next_states=self._getConsecutiveStateActionNextStateData(trajectory)

      if evaluate:
        print("writing to csv..")

        predictions=self.pred_net.predict(current_states_actions, batch_size=10, verbose=0)

        #compute mse
        mse = (numpy.square(next_states - predictions)).mean(axis=1)
        print("average mse: ", mse.sum()/len(next_states))
        
        with open('metrics.csv', 'a') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
          filewriter.writerow([str(mse.sum()/len(next_states))])


      self.pred_net.fit(current_states_actions, next_states,
                epochs=2,
                batch_size=32,
                verbose=1)


    def _createPredictiveModel(self):

      pred_net = keras.Sequential([
            keras.layers.Dense(172),
            keras.layers.Dense(self.neurons_first_layer, activation='relu'),
            keras.layers.Dense(self.neurons_second_layer, activation='relu'),
            #keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(171, activation='sigmoid')])

      pred_net.compile(optimizer='adam',
              loss=tf.keras.losses.mean_squared_error)

      return pred_net




    # returns state, actions & next states in two vectors (state & action concatenated and the resulting state)
    # this can be used as training data for the prediciton network
    def _getConsecutiveStateActionNextStateData(self, trajectory):
      
      #get States & Actions:
      current_states, next_states = self._getStatesFromTrajectory(trajectory)
      actions = self._getActionsFromTrajectory(trajectory)

      #get number of datapoints
      num_datapoints = 0
      for i in range(len(current_states)):
          num_datapoints=num_datapoints+len(current_states[i])


      # create 2D-state-Vector from multiple environments
      next_states_vec = numpy.zeros((num_datapoints, 171))

      counter = 0
      for i in range(len(next_states)):
          for j in range(len(next_states[i])):
              featurevector=next_states[i,j,:]
              next_states_vec[counter,:]=featurevector
              counter = counter+1



      #concatenate current_states & action, and create 2D-vector form multiple environments
      current_states_actions_vec = numpy.zeros((num_datapoints, 171+1))

      counter = 0
      for i in range(len(current_states)):
          for j in range(len(current_states[i])):
              featurevector=numpy.append(current_states[i,j,:],actions[i,j])
              current_states_actions_vec[counter,:]=featurevector
              counter = counter+1

      return current_states_actions_vec, next_states_vec


    def _getActionsFromTrajectory(self, trajectory):
      actions = trajectory[2]

      # remove last element from actions, because it has no it has no consequtive state in next_states
      actions = actions[:,:-1]

      return actions


    def _getStatesFromTrajectory(self, trajectory):
      observations = trajectory[1]
      current_states = observations.get("state")
      next_states = observations.get("state2")

      # remove first element from next_states, because it has no it has no previous state in current_states
      current_states = current_states[:,1:,:]

      # remove last element from current_states, because it has no it has no consequtive state in next_states
      next_states = next_states[:,:-1,:]


      #print("first current_state:")
      #print(current_states[0,0,:])
      #print("second current_state:")
      #print(current_states[0,1,:])
      #...

      #iterate through all states
      #for i in range(len(next_states)):
      #    print("environment number: ",i)
      #    for j in range(len(next_states[i])):
      #        print("state number: ",j)
      #        time.sleep(0.01)

      return current_states, next_states

    def _getRewardsFromTrajectory(self, trajectory):
      rewards = trajectory[5]

      # remove last element from rewards, because it has no it has no consequtive state, which could function as a label for the predictive model
      rewards = rewards[:,:-1]

      return rewards






class ActionPredictor(object):

    def __init__(self, neurons_first_layer, neurons_second_layer, neurons_third_layer):
      self.neurons_first_layer = neurons_first_layer
      self.neurons_second_layer = neurons_second_layer
      self.neurons_third_layer = neurons_third_layer
      self.pred_net = self._createPredictiveModel()


    def train(self, trajectory, evaluate):

      features, labels = self._getFeaturesAndLabelsFromTrajectory(trajectory)
      labels = labels.astype(int)
      # Convert labels to categorical one-hot encoding
      one_hot_labels = keras.utils.to_categorical(labels, num_classes=11)

      if evaluate:
        predictions=self.pred_net.predict(features, batch_size=10, verbose=0)

        # compute categorical cross entropy
        cce= keras.losses.categorical_crossentropy(one_hot_labels, predictions, from_logits=False, label_smoothing=0)
        cce = cce.eval()                                          #this is not good practice and kills tensorflow after a while, better implement as OP if needed a lot!
        print("average cce: ", cce.sum()/len(cce))
        
        with open('metrics.csv', 'a') as csvfile:
          filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
          filewriter.writerow([str(cce.sum()/len(cce))])

        # for demo purposes
        print("--------------------------------------")
        print("entropy of 100th distribution: ", self._computeEntropy(predictions[100]))

        highest_prob = 0
        for i in range(len(predictions[100])):
          if predictions[100][i]> highest_prob:
            highest_prob=predictions[100][i]
        print("highest probability in distribution: ", highest_prob)

        #time.sleep(1000)
        #self._visualizeDistribution(predictions[100])
        print("--------------------------------------")

      self.pred_net.fit(features, one_hot_labels,
                epochs=2,
                batch_size=32,
                verbose=1)



    #this predictive models outputs a softmax distribution over all possible actions and gets as input the last three states of agent A and the last actions in between
    def _createPredictiveModel(self):


      pred_net = keras.Sequential([
            keras.layers.Dense(171+171+171+3),
            keras.layers.Dense(self.neurons_first_layer, activation='relu'),
            keras.layers.Dense(self.neurons_second_layer, activation='relu'),
            keras.layers.Dense(self.neurons_third_layer, activation='relu'),
            keras.layers.Dense(11, activation='softmax')])

      pred_net.compile(optimizer='adam',
                      loss='categorical_crossentropy')

      return pred_net

    def _visualizeDistribution(self, distribution):

      distribution = distribution*100
      distribution = distribution.astype(int)
      print("predicted distribution: ")
      for i in range(len(distribution)):
        print("action: ",i)
        print(distribution[i] * "#")
        print("")

    def _computeEntropy(self, distribution):

      entropy = -numpy.sum(distribution * numpy.log2(distribution), axis=0)

      return entropy


    # features are the last 3 states of player A and all actions of A nad B at these time steps, labels are the following action of player B
    def _getFeaturesAndLabelsFromTrajectory(self, trajectory):

      # get a vector of all state-keys in the observation-dict (containing Player A and Player B states)
      states_vec = self._getStatesFromTrajectory(trajectory)

      # get a vector of all actions (containing Player A and Player B actions)
      actions_vec = self._getActionsFromTrajectory(trajectory)

      #TODO TEST
      if len(states_vec) != len(actions_vec):
        print("states and actions have different length, this cannot be!")

      train_data_length= len(actions_vec)-5

      concatenated_states_action_features = numpy.zeros((train_data_length, 171+171+171+3))
      action_labels = numpy.zeros((train_data_length))

      for i in range(train_data_length):

        features = (states_vec[i,:],states_vec[i+2,:],states_vec[i+4],
                    numpy.array([actions_vec[i+2],actions_vec[i+3],actions_vec[i+4]]))
        featurevector=numpy.concatenate(features)
        concatenated_states_action_features[i,:]=featurevector

        label=actions_vec[i+5]
        action_labels[i]=label

      return concatenated_states_action_features, action_labels



    def _getStatesFromTrajectory(self, trajectory):

      observations = trajectory[1]
      states = observations.get("state")

      #get number of datapoints over all evironments
      num_datapoints = 0
      for i in range(len(states)):
          num_datapoints=num_datapoints+len(states[i])


      # create 2D-state-Vector from multiple environments
      states_vec = numpy.zeros((num_datapoints, 171))

      counter = 0
      for i in range(len(states)):
          for j in range(len(states[i])):
              featurevector=states[i,j,:]
              states_vec[counter,:]=featurevector
              counter = counter+1

      return states_vec


    def _getActionsFromTrajectory(self, trajectory):

      actions = trajectory[2]


      #get number of datapoints over all evironments
      num_datapoints = 0
      for i in range(len(actions)):
          num_datapoints=num_datapoints+len(actions[i])


      # create 1D-action-Vector from multiple environments (which has to be one)
      actions_vec = numpy.zeros((num_datapoints))

      counter = 0
      for i in range(len(actions)):
          for j in range(len(actions[i])):
              featurevector=actions[i,j]
              actions_vec[counter]=featurevector
              counter = counter+1

      return actions_vec