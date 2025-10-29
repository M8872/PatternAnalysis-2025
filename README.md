# Data 


We need the model to effectively seperate the train dataset by subject. 
Since one subject can have multiple slices. 
If we train on the same subject, and then validate on the same subject. 
Then we are basically just validating on the same stuff that we trained on. 

So obviously Accuracy going to be high straight away. 

But that means we arent LEARNING right.

So we split the train/ directory into, TRAIN and VALIDATE. 

We are splitting by subject. (since each subjects slices are likely similar?) 

And obviously we keep testing completely seperate till the end!!

AD_NC/
├── train/          # used for fitting model weights
│   ├── AD/
│   └── NC/
├── val/            # unseen subjects, used to tune hyperparams and check generalization
│   ├── AD/
│   └── NC/
└── test/           # completely untouched until the final report
    ├── AD/
    └── NC/


## DATA AUG
Since data is small. 

transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])




# Model ConvNext

## Regularization

### Dropout

### Weight Decay



# Train



# Attempts

## 1.Validation + Training Mixed
So we didnt group by subjects, so the model achieved crazy accuracy.
But this was cheating. Model wasnt learning.

  -- 20 epochs
  --batch-size 16 
  --lr 1e-3 

## 2. (72%) Model was clearly overfitting
Model was overfitting.
We see plateau at 72% accuracy.
Train loss dec, val loss inc.

Plateud after like 12

  -- 20 epochs
  --batch-size 16 
  --lr 1e-3 

### 3. (86%) 

1. Half learning rate
    1e-3 to 1e-4
Scheduler
halves learning rate when val loss plateus

faster early learning,
later, we get finer tuning

2. Dropout in classifier head
* forces model to learn patterns
* reduces model from overfitting
    - reduce validation loss from increasing
* reduces model from memorisatio

3. enable weight decay
reduces memorizing tiny image details


4. Data Augmentation
* data very clean
* gives it some fake chaos.
    - makes it learns patterns better
    - less learning on exact pixels
* alzheimers is alzeheimers, even if upside down



5. Increase batch size
* reduce gradient noise.
16- 32

6. Epoch from 20 to 50