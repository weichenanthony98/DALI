# README
Learnware project demo for paper: Learnware Specification via Dual Alignment

## Environments:
The packages need for our codes is displayed in requirements.txt. 
Additionally, our code is running on GPU, if you are running on CPU, please change some package versions accordingly!

Please contact us if there are any omissions or errors.

## File organization

There are 3 directory in our progect demo:
+ toy_example.py: This incorporates our Dali approach while running a demo experiment.
+ utils/Network.py: Various neural network codes involved in the Dali approach.
+ specification/NeuralEmbedding: This is the detail of Dali approach.

## How to run ?
Run the toy_example from the directory file.

## How to build the learnware paradigm via Lane specification ?
By simulating the logical structure of toy_example's code, use your own dataset to build the appropriate task datasets, task models and requirement datasets.

First, design a global feature extraction model that unifies the task and requirement data into a single feature space. 
Then, using the Dali method to generate specification from the task dataset and combine these with the corresponding task model to create a learnware, which can be submitted to a market. 
Finally, deploying the model using the requirement data, following the deployment phase in the toy_example, and obtain the corresponding results.

It is worth noting that,  in the learnware paradigm deployment phase of this demo, we only show the learnware identification process in the homogeneous label space. If you need more related stuff, please contact us.

If you have any questions, please feel free to email Wei-Chen@seu.edu.cn.