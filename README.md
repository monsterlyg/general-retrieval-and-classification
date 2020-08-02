# general-retrieval-and-classification
This is a Pytorch complementation of 'Combination of Multiple Global Descriptors for Image Retrieval'. You can use it for image retrieval or classification. 
The CGDnet is according to 'https://github.com/leftthomas/CGD'. I added some personal usually used tools.
Besides, I added codes in trainCGD.py for transferring learning. So it will be easy to adapt to your own work.

# About training.
I add two train files in this project.'train.py' is easy, you can set requisite parameters in 'CGDparams.json'. Also, in 'trainCGD.py', it can eval the training process at the same time. And you can change the parameters by using 'arg_parser' in command line.

# About transfer.
If you want to do a big data pretrain. But the class num is different for your practical work. Just pretrain on big data. Change the 'num_class'. And run
'''
python trainCGD.py -r -t
'''
in command line. It can continue the training process. Additionally, do not let the max epoch exceed you setted parameter.
