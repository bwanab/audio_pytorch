## AUDIO_PYTORCH

This is a repository built from the youtube playlist: https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm by Valerio Velardo. This is strictly a learning exercise meant for no intended useful value otherwise.

You can find his code referenced in the notes on the videos.

## Note on training and testing

I've made a couple of enhancements to the original code:

1. I added the notion of fold to the urbansounddataset module. This is taken from the UrbanSoundDataset website. They arrange the data in 10 "folds" which are subsets of the data. They suggest training with one fold as the test set and the other 9 as the training set. 
2. In the inference code, I test all the members of the dataset from a given fold and total up the number that matches against the total.
3. Based on an issue on the github page, I removed the softmax from the model layers since the CrossEntropyLoss expects the raw logits instead of the predicted values.

### Enhancement ideas:

1. Build an overarching function that invokes training by each fold. Thus, starting with fold 1 (they're numbered 1-10), train on 2-10 and test on 1, then train on 1 and 3-10 and test on 2, and so on. 
2. Try different cnn architectures than the one given. 
   1. The given architecture has 4 blocks with input/output channels of (1,16),(16, 32), (32, 64), (64, 128). What if you turned that around? Something like (1, 128), (128, 64), (64, 32), (32, 16).