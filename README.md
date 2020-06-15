# Machine-Learning-with-AWS-DeepComposer
Generative AI and AWS DeepComposer for Udacity AWS Machine Learning Foundations

## Why Machine Learning on AWS?
* AWS offers the broadest and deepest set of AI and ML services with unmatched flexibility.
* You can accelerate your adoption of machine learning with AWS SageMaker. Models that previously took months and required specialized expertise can now be built in weeks or even days.
* AWS offers the most comprehensive cloud offering optimized for machine learning.
* More machine learning happens at AWS than anywhere else.

### More Relevant Enterprise Search With Amazon Kendra
* Natural language search with contextual search results
* ML-optimized index to find more precise answers
* 20+ Native Connectors to simplify and accelerate integration
* Simple API to integrate search and easily develop search applications
* Incremental learning through feedback to deliver up-to-date relevant answers

### Online Fraud Detection with Amazon Fraud Detector
* Pre-built fraud detection model templates
* Automatic creation of custom fraud detection models
* One interface to review past evaluations and detection logic
* Models learn from past attempts to defraud Amazon
* Amazon SageMaker integration

### Tools and Categories
* Amazon Personalize - AI Service
* Amazon SageMaker - ML Service
* TensorFlow - ML Framework
* SageMaker Ground Truth - ML Service
* PyTorch- ML Framework
* Amazon Rekognition Image - AI Service
* Keras - ML Interface

## ML Techniques and Generative AI

### Machine Learning Techniques
* Supervised Learning: Models are presented wit input data and the desired results. The model will then attempt to learn rules that map the input data to the desired results.
* Unsupervised Learning: Models are presented with datasets that have no labels or predefined patterns, and the model will attempt to infer the underlying structures from the dataset. Generative AI is a type of unsupervised learning.
* Reinforcement learning: The model or agent will interact with a dynamic world to achieve a certain goal. The dynamic world will reward or punish the agent based on its actions. Overtime, the agent will learn to navigate the dynamic world and accomplish its goal(s) based on the rewards and punishments that it has received.

### Generative AI
Generative AI is one of the biggest recent advancements in artificial intelligence technology because of its ability to create something new. It opens the door to an entire world of possibilities for human and computer creativity, with practical applications emerging across industries, from turning sketches into images for accelerated product development, to improving computer-aided design of complex objects. It takes two neural networks against each other to produce new and original digital works based on sample inputs.

## AWS Composer and Generative AI
AWS Deep Composer uses Generative AI, or specifically Generative Adversarial Networks (GANs), to generate music. GANs pit 2 networks, a generator and a discriminator, against each other to generate new content.

The best way we’ve found to explain this is to use the metaphor of an orchestra and conductor. In this context, the generator is like the orchestra and the discriminator is like the conductor. The orchestra plays and generates the music. The conductor judges the music created by the orchestra and coaches the orchestra to improve for future iterations. So an orchestra, trains, practices, and tries to generate music, and then the conductor coaches them to produced more polished music.

### AWS DeepComposer Workflow
* Use the AWS DeepComposer keyboard or play the virtual keyboard in the AWS DeepComposer console to input a melody.

* Use a model in the AWS DeepComposer console to generate an original musical composition. You can choose from jazz, rock, pop, symphony or Jonathan Coulton pre-trained models or you can also build your own custom genre model in Amazon SageMaker.

* Publish your tracks to SoundCloud or export MIDI files to your favorite Digital Audio Workstation (like Garage Band) and get even more creative.

## Demo: Compose Music with DeepComposer
We used a Generative AI Technique to create a Rock Composition based on "Twinkle, Twinkle, Little Star" sample input! ```twinklerockdeepcomposer.midi```

We could technically use the Virtual Keyboard to generate a composition based on
an input melody provided with the notes we played. This step will take few minutes to generate a composition inspired by the chosen genre
Using pre-trained models to generate new music is fun!

## How AWS DeepComposer Works
AWS DeepComposer uses a GAN

### Loss Functions
In machine learning, the goal of iterating and completing epochs is to improve the output or prediction of the model. Any output that deviates from the ground truth is referred to as an error. The measure of an error, given a set of weights, is called a loss function. Weights represent how important an associated feature is to determining the accuracy of a prediction, and loss functions are used to update the weights after every iteration. Ideally, as the weights update, the model improves making less and less errors. Convergence happens once the loss functions stabilize.

We use loss functions to measure how closely the output from the GAN models match the desired outcome. Or, in the case of DeepComposer, how well does DeepComposer's output music match the training music. Once the loss functions from the Generator and Discriminator converges, this indicates the GAN model is no longer learning, and we can stop its training.

We also measures the quality of the music generated by DeepComposer via additional quantitative metrics, such as drum pattern and polyphonic rate.

GAN loss functions have many fluctuations early on due to the “adversarial” nature of the generator and discriminator.

Over time, the loss functions stabilizes to a point, we call this convergence. This convergence can be zero, but doesn’t have to be.

### How It Works
* Input melody captured on the AWS DeepComposer console
* Console makes a backend call to AWS DeepComposer APIs that triggers an execution Lambda.
* Book-keeping is recorded in Dynamo DB.
* The execution Lambda performs an inference query to SageMaker which hosts the model and the training inference container.
* The query is run on the Generative AI model.
* The model generates a composition.
* The generated composition is returned.
* The user can hear the composition in the console.
* The user can share the composition to SoundCloud.

## Training Architecture

## How to measure the quality of the music we’re generating:
* We can monitor the loss function to make sure the model is converging
* We can check the similarity index to see how close is the model to mimicking the style of the data. When the graph of the similarity index smoothes out and becomes less spikey, we can be confident that the model is converging
* We can listen to the music created by the generated model to see if it's doing a good job. The musical quality of the model should improve as the number of training epochs increases.

### Workflow for Training

* User launch a training job from the AWS DeepComposer console by selecting hyperparameters and data set filtering tags
* The backend consists of an API Layer (API gateway and lambda) write request to DynamoDB
* Triggers a lambda function that starts the training workflow
* It then uses AWS Step Funcitons to launch the training job on Amazon SageMaker
* Status is continually monitored and updated to DynamoDB
* The console continues to poll the backend for the status of the training job and update the results live so users can see how the model is learning

### Challenges with GANs
* Clean datasets are hard to obtain
* Not all melodies sound good in all genres
* Convergence in GAN is tricky – it can be fleeting rather than being a stable state
* Complexity in defining meaningful quantitive metrics to measure the quality of music created

## Generative AI Overview
Generative AI has been described as one of the most promising advances in AI in the past decade by the MIT Technology Review.

Generative AI opens the door to an entire world of creative possibilities with practical applications emerging across industries, from turning sketches into images for accelerated product development, to improving computer-aided design of complex objects.

For example, Glidewell Dental is training a generative adversarial network adept at constructing detailed 3D models from images. One network generates images and the second inspects those images. This results in an image that has even more anatomical detail than the original teeth they are replacing.

Generative AI enables computers to learn the underlying pattern associated with a provided input (image, music, or text), and then they can use that input to generate new content. 
Examples of Generative AI techniques include Generative Adversarial Networks (GANs), Variational Autoencoders, and Transformers.

### What are GANs?
GANs, a generative AI technique, pit 2 networks against each other to generate new content. The algorithm consists of two competing networks: a generator and a discriminator.

* A generator is a convolutional neural network (CNN) that learns to create new data resembling the source data it was trained on.

* The discriminator is another convolutional neural network (CNN) that is trained to differentiate between real and synthetic data.

The generator and the discriminator are trained in alternating cycles such that the generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.

### Like the collaboration between an orchestra and its conductor
The best way we’ve found to explain this is to use the metaphor of an orchestra and conductor. An orchestra doesn’t create amazing music the first time they get together. They have a conductor who both judges their output, and coaches them to improve. So an orchestra, trains, practices, and tries to generate polished music, and then the conductor works with them, as both judge and coach.

The conductor is both judging the quality of the output (were the right notes played with the right tempo) and at the same time providing feedback and coaching to the orchestra (“strings, more volume! Horns, softer in this part! Everyone, with feeling!”). Specifically to achieve a style that the conductor knows about. So, the more they work together the better the orchestra can perform.

The Generative AI that AWS DeepComposer teaches developers about uses a similar concept. We have two machine learning models that work together in order to learn how to generate musical compositions in distinctive styles.

## Introduction to U-Net Architecture
### Training a machine learning model using a dataset of Bach compositions
AWS DeepComposer uses GANs to create realistic accompaniment tracks. When you provide an input melody, such as twinkle-twinkle little star, using the keyboard U-Net will add three additional piano accompaniment tracks to create a new musical composition.

The U-Net architecture uses a publicly available dataset of Bach’s compositions for training the GAN. In AWS DeepComposer, the generator network learns to produce realistic Bach-syle music while the discriminator uses real Bach music to differentiate between real music compositions and newly created ones

### How U-Net based model interprets music
Music is written out as a sequence of human readable notes. Experts have not yet discovered a way to translate the human readable format in such a way that computers can understand it. Modern GAN-based models instead treat music as a series of images, and can therefore leverage existing techniques within the computer vision domain.

In AWS DeepComposer, we represent music as a two-dimensional matrix (also referred to as a piano roll) with “time” on the horizontal axis and “pitch” on the vertical axis. You might notice this representation looks similar to an image. A one or zero in any particular cell in this grid indicates if a note was played or not at that time for that pitch.

## Model Architecture

As described in previous sections, a GAN consists of 2 networks: a generator and a discriminator. Let’s discuss the generator and discriminator networks used in AWS DeepComposer.

### Generator
The generator network used in AWS DeepComposer is adapted from the U-Net architecture, a popular convolutional neural network that is used extensively in the computer vision domain. The network consists of an “encoder” that maps the single track music data (represented as piano roll images) to a relatively lower dimensional “latent space“ and a ”decoder“ that maps the latent space back to multi-track music data.

Here are the inputs provided to the generator:

* Single-track piano roll: A single melody track is provided as the input to the generator.
* Noise vector: A latent noise vector is also passed in as an input and this is responsible for ensuring that there is a flavor to each output generated by the generator, even when the same input is provided.

### Discriminator
The goal of the discriminator is to provide feedback to the generator about how realistic the generated piano rolls are, so that the generator can learn to produce more realistic data. The discriminator provides this feedback by outputting a scalar value that represents how “real” or “fake” a piano roll is.

Since the discriminator tries to classify data as “real” or “fake”, it is not very different from commonly used binary classifiers. We use a simple architecture for the critic, composed of four convolutional layers and a dense layer at the end.

Once you complete building your model architecture, the next step is training.

## Training Methodology
During training, the generator and discriminator work in a tight loop as following:

### Generator
* The generator takes in a batch of single-track piano rolls (melody) as the input and generates a batch of multi-track piano rolls as the output by adding accompaniments to each of the input music tracks.
* The discriminator then takes these generated music tracks and predicts how far it deviates from the real data present in your training dataset.

### Discriminator
* This feedback from the discriminator is used by the generator to update its weights. As the generator gets better at creating music accompaniments, it begins fooling the discriminator. So, the discriminator needs to be retrained as well.
* Beginning with the discriminator on the first iteration, we alternate between training these two networks until we reach some stop condition (ex: the algorithm has seen the entire dataset a certain number of times).
### Finer control of AWS DeepComposer with hyperparameters
As you explore training your own custom model in the AWS DeepComposer console, you will notice you have access to several hyperparameters for finer control of this process. Here are a few details on each to help guide your exploration.

### Number of epochs
When the training loop has passed through the entire training dataset once, we call that one epoch. Training for a higher number of epochs will mean your model will take longer to complete its training task, but it may produce better output if it has not yet converged. You will learn how to determine when a model has completed most of its training in the next section.

Training over more epochs will take longer but can lead to a better sounding musical output
Model training is a trade-off between the number of epochs (i.e. time) and the quality of sample output.

### Learning Rate
The learning rate controls how rapidly the weights and biases of each network are updated during training. A higher learning rate might allow the network to explore a wider set of model weights, but might pass over more optimal weights.

### Update ratio
A ratio of the number of times the discriminator is updated per generator training epoch. Updating the discriminator multiple times per generator training epoch is useful because it can improve the discriminators accuracy. Changing this ratio might allow the generator to learn more quickly early-on, but will increase the overall training time.

While we provide sensible defaults for these hyperparameters in the AWS DeepComposer console, you are encouraged to explore other settings to see how each changes your model’s performance and time required to finish training your model.