# Quantized Neural Network Adversarial Defense and Detection

This repository contains the source code for paper "Adversarial Example Defense and Detection using Quantized Neural Networks with Inference Time Dropout", 
it examines the effectiveness of quantized neural network with different bitwidth on the topic of adversarial input defense and detection.
 

## Binarized Neural Networks attack and defense
To reproduce results presented in the paper:
+ `cleverhans_tutorials/mnist_attack.py`

All of the results in the paper can be reproduced by specified different command line options.
Note that the paper contains two major components:<br />
Adversarial Input Defense: test the neural network's ability to detect the true class when provided with an adversarial input. This action is always performed. 
Adversarial Input Detection: test the neural network's ability to distinguish if the input provided is clean or an adversarially manipulated one.
This is done by performing monte carlo inference via inference time dropout, see paper for details, the corresponding commandline option is --measure_uncertainty. 
```
cleverhans_tutorials/mnist_attack.py

--measure_uncertainty
	if specified, perform monte carlo inference, and the variance of estimation is recorded
--attack 
	comma separated list of attack methods
	ATTACK_CARLINI_WAGNER_L2 = 0
	ATTACK_JSMA = 1
	ATTACK_FGSM = 2
	ATTACK_MADRYETAL = 3
	ATTACK_BASICITER = 4
--adv
	adversarial training method (for defense)
	ADVERSARIAL_TRAINING_MADRYETAL = 1
	ADVERSARIAL_TRAINING_FGSM = 2
--binary
	if specified, use BNN (binary neural network)
--rand
	if specified, use randomized BNN (binary neural network with dropout), has an effect only if --binary is set
--bit
	specifies the bit width of the quantized neural network, should not be set if --binary is set
```

Example run:
Evaluate the effectiveness of using randomized BNN with dropout 0.5 as adversarial example detection method
```
./mnist_attack.py --attack 2 --dropout=0.5 --measure_uncertainty
```
Evaluate the accuracy of 4-bit Quantized neural network on clean and  JSMA inputs
```
./mnist_attack.py --attack 1 --bit=4
```



## Acknowledgement

This work is built upon [Cleverhans](https://github.com/tensorflow/cleverhans). 