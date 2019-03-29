import tensorflow as tf
import numpy as np
import corpus_utils as corp
import rnn_utils as ru


#Set Variables
TXTLEN = 10 ** 4
ALPHASIZE = 98
INTERNALSIZE = 512
NLAYERS = 3
TOPN = 2
OUTDIR = "./"
OUTFILE = "lore-output.txt"
MODELDIR = "./checkpoints/"
SEEDCHAR = "["

'''
INITIALIZATION OF MODEL
'''

#Initialize Tensorflow model.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Load newest model.
model = tf.train.latest_checkpoint(MODELDIR)
saver = tf.train.import_meta_graph(model + ".meta")
saver.restore(sess, model)

'''
TEXT GENERATION
'''

#Prepare arrays for text generation.
X = np.array([[corp.ascii_convert(ord(SEEDCHAR))]])
H = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)

#Prepare text file for output.
output = open(OUTFILE, "w")
output.write(SEEDCHAR)

#Generate text based on trained neural net.
constants = [ALPHASIZE, SEEDCHAR, INTERNALSIZE, NLAYERS]
ru.generate_text(sess, constants, TXTLEN, OUTFILE)
