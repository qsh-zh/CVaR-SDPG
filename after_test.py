import tensorflow as tf
import numpy as np

from params import test_params
from agent import Agent
from params import train_params
from utils.gaussian_noise import GaussianNoiseGenerator


def test():
    # Set random seeds for reproducability
    np.random.seed(test_params.RANDOM_SEED)
    tf.set_random_seed(test_params.RANDOM_SEED)
         
    # Create sessionThe
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)     
            
    # Initialise agent
    agent = Agent(sess, test_params.ENV, test_params.RANDOM_SEED)
    # Build network
    agent.build_network(training=False)
    
    # Test network
    # agent.test()
    gaussian_noise = GaussianNoiseGenerator(train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.NOISE_SCALE)

    # noise_levels = [0,0.1,0.2,0.3,0.4]
    noise_levels = [0.0,0.5,1.0,1.5]
    for noise in noise_levels:
        agent.after_test(gaussian_noise,noise)
        # agent.noise_test(gaussian_noise,noise)
    
    sess.close()
    
    
if  __name__ == '__main__':
    test()