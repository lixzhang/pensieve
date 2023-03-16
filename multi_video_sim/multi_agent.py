import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import env
import a3cV13 as a3c


S_INFO = 7  # bit_rate, buffer_size, bandwidth_measurement, measurement_time, chunk_til_video_end
S_LEN = 10  # take how many frames in the past
A_DIM = 10
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.0001
NUM_AGENTS = 96
TRAIN_SEQ_LEN = 200  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [200,300,450,750,1200,1850,2850,4300,6000,8000]  # Kbps
HD_REWARD = [0.5, 1, 1.5, 2, 3, 12, 15, 20, 28, 38] # logic: linear interpretation for 200 and 450; for 6000 and 8000 use the same ratio as for 4300
# HD_REWARD = [e * 1.5 for e in HD_REWARD]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
M_IN_B = 1000000.0
FIRST_CHUNKS = 10
REBUF_PENALTY = 4.3 # 4.3  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY_FIRST = 4.3
SMOOTH_PENALTY = 1.
BUFFER_THRESH = 30
SMOOTH_NEGATIVE_MUL_HIGH = 1
SMOOTH_NEGATIVE_MUL_LOW = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
MODEL_DIR = './models/'
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
NN_MODEL = './models/nn_model_ep_9100.ckpt'
# NN_MODEL = None
epoch = 10000
epoch_to_train = 10000
end_epoch = epoch + epoch_to_train

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
# for multi-video setting,
# "bit_rate" is the action *after* masking
# e.g., bit_rate = 1, mask = [0, 0, 1, 0, 1, 1, ..., 0]
#                                         ^
#       it is selecting ------------------|------------
#                                     this value
# "action" is the output from actor network
# for the example above, action would be = 4

def action_to_bitrate(action, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert action >= 0
    assert action < a_dim
    assert mask[action] == 1
    # index starts at 0, ':' is non-inclusive
    return np.sum(mask[:action])  

def bitrate_to_action(bitrate, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert bitrate >= 0 
    assert bitrate < np.sum(mask)
    cumsum_mask = np.cumsum(mask) - 1
    action = np.where(cumsum_mask == bitrate)[0][0]
    return action


def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    os.system('python rl_test.py ' + nn_model)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = round(np.min(rewards), 1)
    rewards_5per = round(np.percentile(rewards, 5), 1)
    rewards_mean = round(np.mean(rewards), 1)
    rewards_median = round(np.percentile(rewards, 50), 1)
    rewards_95per = round(np.percentile(rewards, 95), 1)
    rewards_max = round(np.max(rewards), 1)

    log_file.write(str(epoch).rjust(6) + '\t' +
                   str(rewards_min).rjust(10) + '\t' +
                   str(rewards_5per).rjust(10) + '\t' +
                   str(rewards_mean).rjust(10) + '\t' +
                   str(rewards_median).rjust(10) + '\t' +
                   str(rewards_95per).rjust(10) + '\t' +
                   str(rewards_max).rjust(10) + '\n')
    log_file.flush()


def central_agent(net_params_queues, exp_queues):
    global epoch

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_test', 'w') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.compat.v1.global_variables_initializer())
        writer = tf.compat.v1.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.compat.v1.train.Saver(max_to_keep=100)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        # epoch = 0

        # while True:  # assemble experiences from agents, compute the gradients
        while True:
            if epoch > end_epoch:
                break
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                for i in range(len(actor_gradient)):
                    assert np.any(np.isnan(actor_gradient[i])) == False

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, MODEL_DIR + "nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                testing(epoch, 
                    MODEL_DIR + "nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)


def agent(agent_id, net_params_queue, exp_queue):

    net_env = env.Environment(random_seed=agent_id,
                              fixed_env=False,
                              trace_folder=TRAIN_TRACES)

    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        mask = net_env.video_masks[net_env.video_idx]

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action = bitrate_to_action(bit_rate, mask)
        last_action = action
        highest_action = bitrate_to_action(np.sum(mask)-1, mask)  
        
        action_vec = np.zeros(np.sum(mask))
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        prev_buffer = 0
        n_chunks = 0        
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, \
                rebuf, video_chunk_size, end_of_video, \
                video_chunk_remain, video_num_chunks, \
                next_video_chunk_size, mask, chunk_length, buffer_limit, next_video_chunk_duration, next_bw = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

#             reward = VIDEO_BIT_RATE[action] / M_IN_K \
#                      - REBUF_PENALTY * rebuf \
#                      - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
#                                                VIDEO_BIT_RATE[last_action]) / M_IN_K
            
            # 7. HD reward, weighted by chunk length
            reward = HD_REWARD[bit_rate] * chunk_length \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * max(HD_REWARD[last_action] - HD_REWARD[action], 0)

            r_batch.append(reward)

            prev_buffer = buffer_size
            last_bit_rate = bit_rate
            last_action = action

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[action] / float(np.max(VIDEO_BIT_RATE)) * 10  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K
            state[4, -1] = video_chunk_remain / float(video_num_chunks)
            state[5, :] = -1
            nxt_chnk_cnt = 0
            for i in range(A_DIM):
                if mask[i] == 1:
                    state[5, i] = next_video_chunk_size[nxt_chnk_cnt] / M_IN_B
                    nxt_chnk_cnt += 1
            assert(nxt_chnk_cnt) == np.sum(mask)
            state[6, -A_DIM:] = mask

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))

            # the action probability should correspond to number of bit rates
            assert len(action_prob[0]) == np.sum(mask)

            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            action = bitrate_to_action(bit_rate, mask)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[action]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action = bitrate_to_action(bit_rate, mask)
                last_action = action
                action_vec = np.zeros(np.sum(mask))
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(np.sum(mask))
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main():

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
