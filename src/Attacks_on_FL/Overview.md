# Overview of Attacks on FL

Paper: [A Survey on Securing Federated Learning:
Analysis of Applications, Attacks,
Challenges, and Trends](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10107622)

The paper categorizes the main attacks on FL as 

A - model performance attacks 

1. Data Poisoning Attacks: Change training data for local model
2. Model Poisoning Attacks: Change gradients of trained local model
3. Free-riding Attacks: Use global model but do not help training it

B - privacy attacks

1. Model Inversion And Gradient Inference: Reconstruct private training data from model parameters or gradients.
2. GAN Reconstruction Attack: Use the global model as a discriminator in a GAN to reconstruct private participant data

## A - Model Performance Attacks

- Goal: direct or indirect affect the performance of global model
- Attackers send incorrect or corrupted paramters
- Attacks can be 
    - targetted attacks (backdoor attacks): Makes the model fail to predict a particular class
    - untargetted attacks (byzantine attacks): Try to poisen the model, no matter how

### 1 - Data Poisoning Attacks

- Attacker tries to poisen the trainings data
- Indirect attack to the global model
- Introduce mislabeled data to the model (swap label of datapoints in training process)
- Attacks can be done with GANs
    - Train a GAN
    - Generate trainings data that looks realistic
Types of data poisoning attacks
    - clean-label attack: Attacker can't change the labels because of a certification process -> requires imperceptivle modifications
    - dirty-labels attacks: Attacker can modify everything

##### Defenses 
1. Robust aggregation
    - Don't just calculate average value of weights
    - Values that deviate significantly from the other values receive less attention

2. Differential Privacy
    - Add noise to local model updates
    - Protects participant privacy
    - Prevents data reconstruction (e.g., GAN-based attacks)
    - Limits effectiveness of poisoning attacks (not complete prevention)


- Attacks are difficult to carry out in FL: One client has less power of the global training process
- Attacks are critical if attacker has access to many clients

#### Attacks specialized on FL
__Sybil Attack__
- Attacker creates many false identities and poison the server

- Defense
    - Method called FoolsGold
    - With this method the honest clients and malicious clients can be differentiated
    - Updates of malicious clients are similar because they have the same goal
    - Can be visualized with PCA

__Distributed Backdoor Attacks__

- Injecting adversarial triggers to trainings data
- A trigger can be
    - specific image (e.g. red square, green, circle, ..)
    - sequence of pixels
- Trigger is distributed across different malicious clients
- Each client inserts a part of the trigger to local trainingsdata (e.g. a corner of the red square is added to trainingsimage and label is changed to dog)
- Backdoor can not be identified if it is distributed across many clients
- By averaging the weights the full trigger is now active (e.g. If an image is sent to the global model with a red square the image is irrelevant and because of the square it decides it is a dog)

### 2 - Model Poisoning Attacks

- Attacker tries to manipulate local model updates before sending them to the server 
- Direct attack to the global model
- Goal: Misclassify inputs with high confidence
- More effictive than data poisening attacks

#### Attacks

__Backdoor attacks__
- Attacker trains local model so, that it have high accuracy
    - solving its normal task
    - and activating the backdoor if a pattern or keyword is received
    - Can be detected, because paramters are larger or more different from other participants 

#### Defenses

1. Robus aggregation
    - 1. Approach: Evaluating performance of models with test set on server and discard models with poor performance
    - 2. Approach: Analyse model parameters and look for strong deviating updates 


### 3 - Free-Riding Attacks

- Attacker tries to exploit the benefits of the global model without investing ressources in the training process
- Attacker uses a subset of all trainingsdata to train local model or uses random noise instead
- Quality of global model decreases because of poor quality of attacker

#### Defenses

1. Blockchain
- Track participant updates on blockchain
- Can lead to privacy attacks

2. Reward mechanisms
- Rewards to clients that are contributing
- Penalizing those who do not

## B - Privacy Attacks

- Goal: protect privacy of the participants
- Attacker try to infer data based on the updates
- The aggregation server is the most likely entity to perform such attacks

### 1 - Model Inversion And Gradient Inference

Goal: Predict the dataset used as input to train the model
- Attack can be performed 
    - by the aggregation server
    - against blockchain-based proposals, because the updates are in clear text on the chain
- Uneffective if model sturcture is complex

#### Attacks

__Model Inversion (Real Fake Faces)__

- Attacker has access to 
    - Trained model
    - Access to models predictions
- Attacker starts with random picture
- Attacker add changes to image until it says -> "Real Face"

#### Defenses

__Federated Generative Privacy (FedGP)__
- Train local GANs to produce artificial datasets without private data

__Privacy-Enhanced Federated Learning__
- Using Cryptography to secure the gradients
- Server can do mathematical operations to aggregate the encrypted gradients of the clients, but never sees the decrypted gradients of an individual client

__NoPeekNN__
- Add noise to private data
- Increase privacy but decrease performance


### 2 - GAN Reconstruction Attack

- Goal: Replicate private data of clients
- More effective than the model inversion attacks

#### Attacks
- Discriminator: An replica of the global model made by the Attacker 
- Generator: Trained by the Attacker to generate replicas stored on the clients
- Attacker generates data with the Generator and inputs this into to Discriminator 
- Minimize loss to get real input data

#### Defenses

__Secure multiparty computation__
- Calculations without knowing the raw data
- Clients masks their data 
- Masks are reversed with masks of other clients
- Trusted third party is needed for mask generation

__Burried_Point_Layer__
- Each participants embeds in the training process an additional layer (burried point layer)
- If this layer is missing in a client model, the server knwos it is malicious
