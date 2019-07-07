- [X] Extract features from Squad 2.0 per https://github.com/google-research/bert#using-bert-to-extract-fixed-feature-vectors-like-elmo
- [ ] Read deconvolution papers and ensure we can use keras implementation: https://keras.io/backend/#conv2d_transpose
- [X] ~Set up simple CNN to consume these features~
- [X] Run bert/run_squad.py on the uncased Bert base dataset for a comparison.
- [X] Calculate logits and loss for answers
- [ ] Run a more complex CNN and try to attain score of 85+.
- [ ] Attempt create context aware bidirectional CNN
- [ ] Add a training hook (`tf.train.SessionRunHook`) to get the accuracy on an evaluation set.

