import tensorflow as tf
import numpy as np
import corpus_utils as corp
import rnn_utils as ru


#Set seed for reproducibility.
tf.set_random_seed(42)

#Supress TF warnings for clean printed output.
tf.logging.set_verbosity(tf.logging.ERROR)

#Set variables.
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = 98
INTERNALSIZE = 512
NLAYERS = 3
LRNRATE = 0.001
DRPKEEP = 0.8
EPOCHS = 10
CORPDIR = "./tng/*.txt"
OUTDIR = "./test_output/"
MODELDIR = "./checkpoints/"
SEEDCHAR = "\t"

#Process corpus files into unicode training and validation data.
corpus, valitxt, file_ranges = corp.load_corpus(CORPDIR)

#Print corpus, validation set, and epoch sizes.
epoch_size = len(corpus) // (BATCHSIZE * SEQLEN)
print("Training Set Size: " + str(round(len(corpus)/ 1024 ** 2, 2)) + "MB")
print("Validation Set Size: " + str(round(len(valitxt)/ 1024, 2)) + "KB")
print("Batches Per Epoch: " + str(epoch_size) + "\n\n")

'''
INITIALIZATION OF MODEL
'''

#Initialize Tensorflow model.
stats, train_step = ru.initialize_tf(ALPHASIZE, INTERNALSIZE, NLAYERS)
_50_BATCHES = 50 * BATCHSIZE * SEQLEN
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#If checkpoints exist, load newest model.
#Otherwise, start from the beginning.
saver, pbar, step, cp_step = \
        ru.load_checkpoint(sess, MODELDIR, BATCHSIZE, SEQLEN, epoch_size)

'''
TRAINING LOOP
'''

for x, y_, epoch in ru.batch_seq(corpus, BATCHSIZE, SEQLEN, EPOCHS, step):

    #Train neural net.
    feed_dict = {"X:0": x, "Y_:0": y_, "Hin:0": istate,
                 "lr:0": LRNRATE, "pkeep:0": DRPKEEP, "batchsize:0": BATCHSIZE}
    _, y, ostate = sess.run([train_step, "Y:0", "H:0"], feed_dict)

    #Every 500 batches.
    if step // 10 % _50_BATCHES == 0 and \
            step // (BATCHSIZE * SEQLEN) > cp_step:
        
        #Define number of iterations to get to this point.
        niter = step // (BATCHSIZE * SEQLEN)

        #Test Output and perform validation of current model.
        constants = [ALPHASIZE, SEEDCHAR, INTERNALSIZE, NLAYERS]
        ru.test_output(sess, stats, valitxt, constants, niter, OUTDIR)

        #Save model.
        ru.save_model(saver, sess, MODELDIR, niter)

    #Loop state around & update progress bar.
    istate = ostate
    step += BATCHSIZE * SEQLEN
    pbar.update(1)

pbar.close()
