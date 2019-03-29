import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import corpus_utils as corp
from tqdm import tqdm
import numpy as np
import glob

#Set seed for reproducibility.
np.random.seed(42)

def batch_seq(raw_data, batch_size, seq_size, n_epochs, step):
    '''
    Divide the data into continuous batches. 
    '''
    data = np.array(raw_data)
    data_len = data.shape[0]
    
    #Shift sequence by 1
    n_epochs = (data_len - 1) // (batch_size * seq_size)
    rounded_data_len = n_epochs * batch_size * seq_size
    xdata = np.reshape(data[0:rounded_data_len],
            [batch_size, n_epochs * seq_size])
    ydata = np.reshape(data[1:rounded_data_len + 1],
            [batch_size, n_epochs * seq_size])
    stepcount = 0

    for epoch in range(n_epochs):
        for batch in range(n_epochs):
            x = xdata[:, batch * seq_size:(batch + 1) * seq_size]
            y = ydata[:, batch * seq_size:(batch + 1) * seq_size]
            
            #Do not reset rnn state
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            
            #Line things up if resuming from previous model
            if stepcount > step or step == 0:
                yield x, y, epoch
            stepcount += batch_size * seq_size

def samp_prob(probabilities, outsize, topn):
    '''
    Return the top-n highest probabilities
    '''
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(outsize, 1, p=p)[0]

def generate_text(sess, constants, nchars, outfile):
    '''
    Generate random text given current model.
    '''
    #Initialize constants.
    alphasize, seedchar, internalsize, nlayers = constants

    #Prepare random variables for input into model.
    ry = np.array([[corp.unicode_convert(ord(seedchar))]])
    rh = np.zeros([1, internalsize * nlayers])
    
    #Prepare output text file.
    output = open(outfile, "w")

    #Character generation loop.
    for k in range(nchars):
        feed_dict = {"X:0": ry, "pkeep:0": 1.0, "Hin:0": rh, "batchsize:0": 1}
        ryo, rh = sess.run(["Yo:0", "H:0"], feed_dict)
        rc = samp_prob(ryo, alphasize, 2)
        print(chr(corp.ascii_convert(rc)), end="", file=output)
        ry = np.array([[rc]])
    output.close()

def validate_model(sess, valitxt, constants, stats, outfile):
    '''
    Validate a given model.
    '''
    #Initialize constants.
    alphasize, seedchar, internalsize, nlayers = constants

    #Prepare output text file.
    output = open(outfile, "a")

    #Initialize constants.
    batchloss, accuracy, summaries = stats
    vali_seqlen = 1024  
    vali_bsize = len(valitxt) // vali_seqlen
    
    #Prepare for testing.
    vali_x, vali_y, _ = next(batch_seq(valitxt, vali_bsize, vali_seqlen, 1, 0))
    vali_nullstate = np.zeros([vali_bsize, internalsize * nlayers])

    #Calculate statistics.
    feed_dict = {"X:0": vali_x, "Y_:0": vali_y, "Hin:0": vali_nullstate,
            "pkeep:0": 1.0, "batchsize:0": vali_bsize}
    ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict)
    
    print("Loss: " + str(ls))
    print("Accuracy: " + str(acc))
    print("\n\n\nValidation\nLoss: " + str(ls) + \
          "\nAccuracy: " + str(acc), file=output)
    output.close()

def test_output(sess, stats, valitxt, constants, step, outdir):
    '''
    Test Output and perform validation of model.
    '''
    #Initialize constants.
    batchloss, accuracy, summaries = stats
    alphasize, seedchar, internalsize, nlayers = constants
    outfile = outdir + "lore-" + str(step) + ".txt"

    #Output test text.
    print("\n\nGenerating test text from learned state...")
    generate_text(sess, constants, 1000, outfile)    
    print("End of random text generation!")

    #Validate and output loss + accuracy.
    print("Validating...")
    validate_model(sess, valitxt, constants, stats, outfile)

def initialize_tf(alphasize, internalsize, nlayers):
    '''
    Initialize Tensorflow model.
    '''
    #Define learning rate, dropout parameter and batch size.
    lr = tf.placeholder(tf.float32, name = "lr")
    pkeep = tf.placeholder(tf.float32, name = "pkeep")
    batchsize = tf.placeholder(tf.int32, name = "batchsize")

    #Define model inputs.
    X = tf.placeholder(tf.uint8, [None, None], name = "X")
    Xo = tf.one_hot(X, alphasize, 1.0, 0.0)
    
    #Define expected model outputs. (trying to predict the next character)
    Y_ = tf.placeholder(tf.uint8, [None, None], name = "Y_")
    Yo_ = tf.one_hot(Y_, alphasize, 1.0, 0.0)
    
    #Define model input state.
    Hin = tf.placeholder(tf.float32, [None, internalsize*nlayers], name="Hin")

    #Perform naive dropout.
    cells = [rnn.GRUCell(internalsize) for _ in range(nlayers)]
    drpcl = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
    mltcl = rnn.MultiRNNCell(drpcl, state_is_tuple = False)
    
    #Perform softmax layer dropout.
    mltcl = rnn.DropoutWrapper(mltcl, output_keep_prob = pkeep)

    #Get last state in sequence.
    Yr, H = tf.nn.dynamic_rnn(mltcl, Xo, dtype=tf.float32, initial_state=Hin)
    H = tf.identity(H, name = "H")

    #Flatten first two dimensions of output and apply softmax layer.
    Yflat = tf.reshape(Yr, [-1, internalsize])
    Ylogits = layers.linear(Yflat, alphasize)
    Yflat_ = tf.reshape(Yo_, [-1, alphasize])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits,
            labels = Yflat_)
    loss = tf.reshape(loss, [batchsize, -1])
    Yo = tf.nn.softmax(Ylogits, name = "Yo")
    Y = tf.argmax(Yo, 1)
    Y = tf.reshape(Y, [batchsize, -1], name = "Y")
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    #Compute statistics.
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)),
        tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])
    
    #Prepare output variables.
    stats = [batchloss, accuracy, summaries]
    return stats, train_step

def save_model(saver, sess, modeldir, step):
    '''
    Save model to checkpoints directory.
    '''
    saved_file = saver.save(sess, modeldir + "lore",
                global_step=step)
    print("Snapshot of model saved: " + saved_file + "\n\n")

def load_checkpoint(sess, modeldir, batchsize, seqlen, epoch_size):
    '''
    Make saver for new model or resume from last checkpoint.
    '''
    saver = tf.train.Saver(max_to_keep=0)

    #If there are checkpoints, return name of latest one and its step number.
    if len(glob.glob(modeldir + "*.meta")):
        cp_last = tf.train.latest_checkpoint(modeldir)
        cp_step = int("".join(filter(str.isdigit, cp_last)))
        saver.restore(sess,cp_last)
    else:
        cp_step = 0

    #Adjust progress bar and print message to console.
    step = cp_step * batchsize * seqlen
    pbar = tqdm(total=epoch_size ** 2)
    if cp_step > 0:
        pbar.update(cp_step)
        print("\n\nRestored last checkpoint! (" + cp_last + ")")
    else:
        print("\n\nTraining in progress...")
    return saver, pbar, step, cp_step
