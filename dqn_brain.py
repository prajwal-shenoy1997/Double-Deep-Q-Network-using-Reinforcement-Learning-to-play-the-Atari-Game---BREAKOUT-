import tensorflow as tf
import numpy as np
import time

class deepQueueNetwork:
    def __init__(self,n_actions,
                 memory,
                 reward_decay=0.9,
                 learning_rate=0.005,
                 epsilon=0.5,
                 epsilon_decay=0.0000008,
                 merge_both_after_steps=10000,
                 batch_size=32):
        self.n_actions=n_actions
        self.memory = memory
        self.epsilon=epsilon
        self.gamma=reward_decay
        self.learning_rate=learning_rate
        self.epsilon_decay=epsilon_decay
        self.counter=0
        self.merge_both_after_steps=merge_both_after_steps
        self.sess=tf.Session()
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size=batch_size
        self.summary_writer = tf.summary.FileWriter("logs/")
    def build_model(self):

        # define placeholders for s,a,r,s_
        with tf.name_scope("inputs"):
            self.s  = tf.placeholder(tf.float32,shape=[None,16],name="State")
            self.s_ = tf.placeholder(tf.float32,shape=[None,16],name="State_")
            self.a  = tf.placeholder(tf.int32,shape=[None,],name="Action")
            self.r  = tf.placeholder(tf.float32,shape=[None,],name="Reward")
            self.done=tf.placeholder(tf.float32,shape=[None,],name="Done")
            self.value=tf.placeholder(tf.float32,shape=[None,self.n_actions],name="Q_value")

        w_initializer,b_initializer=tf.random_normal_initializer(mean=0.,stddev=0.3) , tf.constant_initializer(0.1)

        # building evaluator network (weights and biases are continuously varied)
        with tf.variable_scope("eval_net"):

            e_fc1=tf.layers.dense(inputs=self.s,units=512,activation=tf.nn.relu,
                                  kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer,name="e_fc1")
            e_fc2 = tf.layers.dense(inputs=e_fc1, units=1024, activation=tf.nn.relu,
                                    kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name="e_fc2")
            self.q_eval = tf.layers.dense(inputs=e_fc2,units=self.n_actions,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,name="e_prediction")




        # building target network (weights and biases are assigned of the evaluator network once in a while )
        with tf.variable_scope("target_net"):

            t_fc1 = tf.layers.dense(inputs=self.s_, units=512, activation=tf.nn.relu,
                                    kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name="t_fc1")
            t_fc2 = tf.layers.dense(inputs=t_fc1, units=1024, activation=tf.nn.relu,
                                    kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name="t_fc2")
            self.q_target = tf.layers.dense(inputs=t_fc2, units=self.n_actions,
                                           kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name="t_prediction")


        with tf.variable_scope("q_eval"):
            # using tf.range() create [0,1,2,3,4...] array and combine it with action so you get [x,y] as each element of a_indices
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_w_a = tf.gather_nd(self.q_eval,indices=a_indices)

        with tf.variable_scope("q_targ"):
            choose_action = tf.argmax(self.value, axis=1,output_type=tf.int32)
            a_indices = tf.stack([tf.range(tf.shape(choose_action)[0], dtype=tf.int32), choose_action], axis=1)
            q_target_r = tf.gather_nd(self.q_target, a_indices)
            q_target_ra = self.r + self.gamma * self.done * q_target_r
            self.q_target_w_r = tf.stop_gradient(q_target_ra)

        # loss function and optimizer
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_w_r , self.q_eval_w_a),name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)

        self.loss_summary = tf.summary.scalar("Loss_vs_epochs", self.loss)
        self.reward = tf.summary.scalar("Reward vs_epochs",tf.reduce_mean(self.r))
        self.reward_summary = tf.summary.histogram("Reward", self.r)

    def assign(self):
        print("merged")
        self.epsilon = 0.0 * ((2.71828) ** (-self.epsilon_decay * self.counter))
        t_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="target_net")
        e_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="eval_net")
        assign=[tf.assign(t,e) for t,e in zip(t_params,e_params)]
        self.sess.run(assign)
    def status(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        print(self.sess.run(e_params)[1][0:10])
        print(self.sess.run(t_params)[1][0:10])
        print("...........................................")
    def choose_action(self,observation):
        if np.random.uniform() <self.epsilon:
            action=np.random.randint(0,self.n_actions)
        else:
            action=np.argmax(self.sess.run(self.q_eval,feed_dict={self.s:observation}))
        return(action)
    def choose_action_for_state(self,observation):
        return(np.argmax(self.sess.run(self.q_eval,feed_dict={self.s:observation})))
    def learn(self,global_step):
        if self.counter % self.merge_both_after_steps == 0:
            self.assign()
        self.counter = (self.counter + 1) % 1000000000
        batch=self.memory.get_batch(batch_size=self.batch_size)
        if batch is None:
            return

        s,a,r,s_,done=map(np.array, zip(*batch))
        done=np.invert(done).astype(np.int32)
        temp=self.sess.run(self.q_target, feed_dict={self.s_:s_})
        _,loss_summary,reward , reward_summary = self.sess.run([self.optimizer,self.loss_summary,self.reward,self.reward_summary],feed_dict={self.s:s,self.s_:s_,self.a:a,self.r:r,self.done:done,self.value:temp})
        self.summary_writer.add_summary(loss_summary,global_step)
        self.summary_writer.add_summary(reward,global_step)
        self.summary_writer.add_summary(reward_summary,global_step)
        if self.counter % 100 == 0:
            print(self.sess.run(self.loss, feed_dict={self.s: s, self.s_: s_, self.a: a, self.r: r, self.done: done,self.value:temp}),self.epsilon)
    def save(self):
        saver = tf.train.Saver()
        save_path = "my_trained_net"
        save_path = saver.save(sess=self.sess, save_path = save_path + "/my_weights_and_biases.cpkt")
        print("Saved to ", save_path, " successfully!!")
    def load_model(self):
        saver = tf.train.Saver()
        save_path = "my_trained_net"
        saver.restore(sess=self.sess, save_path=save_path + "/my_weights_and_biases.cpkt")
        print("Restrored from", save_path, " successfully!!")
if __name__ == '__main__':
    dqn = deepQueueNetwork(n_actions=10,memory = 10)
    tf.reset_default_graph()
    dqn.build_model()
    writer = tf.summary.FileWriter("logs/",dqn.sess.graph)
