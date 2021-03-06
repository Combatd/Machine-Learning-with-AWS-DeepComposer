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

## Evaluation
Typically when training any sort of model, it is a standard practice to monitor the value of the loss function throughout the duration of the training. The discriminator loss has been found to correlate well with sample quality. You should expect the discriminator loss to converge to zero and the generator loss to converge to some number which need not be zero. When the loss function plateaus, it is an indicator that the model is no longer learning. At this point, you can stop training the model. You can view these loss function graphs in the AWS DeepComposer console.

### Sample output quality improves with more training
After 400 epochs of training, discriminator loss approaches near zero and the generator converges to a steady-state value. Loss is useful as an evaluation metric since the model will not improve as much or stop improving entirely when the loss plateaus.

While standard mechanisms exist for evaluating the accuracy of more traditional models like classification or regression, evaluating generative models is an active area of research. Within the domain of music generation, this hard problem is even less well-understood.

To address this, we take high-level measurements of our data and show how well our model produces music that aligns with those measurements. If our model produces music which is close to the mean value of these measurements for our training dataset, our music should match the general “shape”. You’ll see graphs of these measurements within the AWS DeepComposer console

Here are a few such measurements:

* Empty bar rate: The ratio of empty bars to total number of bars.
* Number of pitches used: A metric that captures the distribution and position of pitches.
* In Scale Ratio: Ratio of the number of notes that are in the key of C, which is a common key found in music, to the total number of notes.

### Music to your ears
Of course, music is much more complex than a few measurements. It is often important to listen directly to the generated music to better understand changes in model performance. You’ll find this final mechanism available as well, allowing you to listen to the model outputs as it learns.

## Inference

Once training has completed, you may use the model created by the generator network to create new musical compositions.

Once this model is trained, the generator network alone can be run to generate new accompaniments for a given input melody. If you recall, the model took as input a single-track piano roll representing melody and a noise vector to help generate varied output.

The final process for music generation then is as follows:

* Transform single-track music input into piano roll format.
* Create a series of random numbers to represent the random noise vector.
* Pass these as input to our trained generator model, producing a series of output piano rolls. Each output piano roll represents some instrument in the composition.
* Transform the series of piano rolls back into a common music format (MIDI), assigning an instrument for each track.

## Build a Custom GAN Part 1: Notebooks and Data Preparation
In this demonstration we’re going to synchronize what you’ve learned about software development practices and machine learning, using AWS DeepComposer to explore those best practices against a real life use case.

### Coding Along With The Instructor (Optional)
To create the custom GAN, you will need to use an instance type that is not covered in the Amazon SageMaker free tier. If you want to code along with the demo and build a custom GAN, you may incur a cost.

You can learn more about SageMaker costs in the Amazon SageMaker pricing documentation


### Setting Up the DeepComposer Notebook
To get to the main Amazon SageMaker service screen, navigate to the AWS SageMaker console. You can also get there from within the AWS Management Console by searching for Amazon SageMaker.
Once inside the SageMaker console, look to the left hand menu and select Notebook Instances.
Next, click on Create notebook instance.
In the Notebook instance setting section, give the notebook a name, for example, DeepComposerUdacity.
Based on the kind of CPU, GPU and memory you need the next step is to select an instance type. For our purposes, we’ll configure a ml.c5.4xlarge
Leave the Elastic Inference defaulted to none.
In the Permissions and encryption section, create a new IAM role using all of the defaults.
When you see that the role was created successfully, navigate down a little ways to the Git repositories section
Select Clone a public Git repository to this notebook instance only
Copy and paste the public URL into the Git repository URL section: https://github.com/aws-samples/aws-deepcomposer-samples
Select Create notebook instance
Give SageMaker a few minutes to provision the instance and clone the Git repository
Exploring the Notebook
Now that it’s configured and ready to use, let’s take a moment to investigate what’s inside the notebook.


When the status reads "InService" you can open the Jupyter notebook.

Status is InService

### Open the Notebook
Click Open Jupyter.
When the notebook opens, click on Lab 2.
When the lab opens click on GAN.ipynb.
Review: Generative Adversarial Networks (GANs).
GANs consist of two networks constantly competing with each other:

* Generator network that tries to generate data based on the data it was trained on.
* Discriminator network that is trained to differentiate between real data and data which is created by the generator.
Note: The demo often refers to the discriminator as the critic. The two terms can be used interchangeably.


### Set Up the Project
Run the first Dependencies cell to install the required packages
Run the second Dependencies cell to import the dependencies
Run the Configuration cell to define the configuration variables
Note: While executing the cell that installs dependency packages, you may see warning messages indicating that later versions of conda are available for certain packages. It is completely OK to ignore this message. It should not affect the execution of this notebook.

Click run
Click Run or Shift-Enter in the cell

* Good Coding Practices
   * Do not hard-code configuration variables
   * Move configuration variables to a separate config file
   * Use code comments to allow for easy code collaboration

### Data Preparation
The next section of the notebook is where we’ll prepare the data so it can train the generator network.


Why Do We Need to Prepare Data?
Data often comes from many places (like a website, IoT sensors, a hard drive, or physical paper) and it’s usually not clean or in the same format. Before you can better understand your data, you need to make sure it’s in the right format to be analyzed. Thankfully, there are library packages that can help! One such library is called NumPy, which was imported into our notebook.

Piano Roll Format
The data we are preparing today is music and it comes formatted in what’s called a “piano roll”. Think of a piano roll as a 2D image where the X-axis represents time and the Y-axis represents the pitch value. Using music as images allows us to leverage existing techniques within the computer vision domain.

Our data is stored as a NumPy Array, or grid of values. Our dataset comprises 229 samples of 4 tracks (all tracks are piano). Each sample is a 32 time-step snippet of a song, so our dataset has a shape of:

```(num_samples, time_steps, pitch_range, tracks)```
or

```(229, 32, 128, 4)```
Piano Roll visualization
Each Piano Roll Represents A Separate Piano Track in the Song

### Load and View the Dataset
Run the next cell to play a song from the dataset.
Run the next cell to load the dataset as a nympy array and output the shape of the data to confirm that it matches the (229, 32, 128, 4) shape we are expecting
Run the next cell to see a graphical representation of the data.
Graphical representation of model data
Graphical Representation of Model Data

### Create a Tensorflow Dataset
Much like there are different libraries to help with cleaning and formatting data, there are also different frameworks. Some frameworks are better suited for particular kinds of machine learning workloads and for this deep learning use case, we’re going to use a Tensorflow framework with a Keras library.

We'll use the dataset object to feed batches of data into our model.

Run the first Load Data cell to set parameters.
Run the second Load Data cell to prepare the data.

## Build a Custom GAN Part 2: Training and Evaluation

### Model Architecture
Before we can train our model, let’s take a closer look at model architecture including how GAN networks interact with the batches of data we feed it, and how they communicate with each other.

### How the Model Works
The model consists of two networks, a generator and a critic. These two networks work in a tight loop:

* The generator takes in a batch of single-track piano rolls (melody) as the input and generates a batch of multi-track piano rolls as the output by adding accompaniments to each of the input music tracks.
* The discriminator evaluates the generated music tracks and predicts how far they deviate from the real data in the training dataset.
* The feedback from the discriminator is used by the generator to help it produce more realistic music the next time.
* As the generator gets better at creating better music and fooling the discriminator, the discriminator needs to be retrained by using music tracks just generated by the generator as fake inputs and an equivalent number of songs from the original dataset as the real input.
* We alternate between training these two networks until the model converges and produces realistic music.

The discriminator is a binary classifier which means that it classifies inputs into two groups, e.g. “real” or “fake” data.

### Defining and Building Our Model
* Run the cell that defines the generator
* Run the cell that builds the generator
* Run the cell that defines the discriminator
* Run the cell that builds the discriminator

### Model Training and Loss Functions
As the model tries to identify data as “real” or “fake”, it’s going to make errors. Any prediction different than the ground truth is referred to as an error.

The measure of the error in the prediction, given a set of weights, is called a loss function. Weights represent how important an associated feature is to determining the accuracy of a prediction.

Loss functions are an important element of training a machine learning model because they are used to update the weights after every iteration of your model. Updating weights after iterations optimizes the model making the errors smaller and smaller.

### Setting Up and Running the Model Training
* Run the cell that defines the loss functions
* Run the cell to set up the optimizer
* Run the cell to define the generator step function
* Run the cell to define the discriminator step function
* Run the cell to load the melody samples
* Run the cell to set the parameters for the training
* Run the cell to train the model!!!!

Training and tuning models can take a very long time – weeks or even months sometimes. Our model will take around an hour to train.

### Model Evaluation
Now that the model has finished training it’s time to evaluate its results.

There are several evaluation metrics you can calculate for classification problems and typically these are decided in the beginning phases as you organize your workflow.

In our example we:

* Checked to see if the losses for the networks are converging
* Looked at commonly used musical metrics of the generated sample and compared them to the training dataset.

### Evaluating Our Training Results
* Run the cell to restore the saved checkpoint. If you don't want to wait to complete the training you can use data from a pre-trained model by setting ```TRAIN = False``` in the cell.
* Run the cell to plot the losses.
* Run the cell to plot the metrics.

### Results and Inference
Finally, we are ready to hear what the model produced and visualize the piano roll output!

Once the model is trained and producing acceptable quality, it’s time to see how it does on data it hasn’t seen. We can test the model on these unknown inputs, using the results as a proxy for performance on future data.

### Evaluate the Generated Music
In the first cell, enter 0 as the iteration number:

```iteration = 0```
run the cell and play the music snippet.


In the second cell, enter 0 as the iteration number:

```iteration = 0```
run the cell and display the piano roll.

In the first cell, enter 500 as the iteration number:
```iteration = 500```
run the cell and play the music snippet.
Or listen to the example snippet at iteration 500.
In the second cell, enter 500 as the iteration number:
```iteration = 500```
run the cell and display the piano roll.
Example Piano Roll at Iteration 500
Example Piano Roll at Iteration 500

Play around with the iteration number and see how the output changes over time!

Here is an example snippet at iteration 950

And here is the piano roll:

Example Piano Roll at Iteration 950
Example Piano Roll at Iteration 950

Do you see or hear a quality difference between iteration 500 and iteration 950?

Watch the Evolution of the Model!
Run the next cell to create a video to see how the generated piano rolls change over time.
Or watch the example video here:

### Inference
Now that the GAN has been trained we can run it on a custom input to generate music.

Run the cell to generate a new song based on "Twinkle Twinkle Little Star".
Or listen to the example of the generated music here:

Run the next cell and play the generated music.
Or listen to the example of the generated music here:

Stop and Delete the Jupyter Notebook When You Are Finished!
This project is not covered by the AWS Free Tier so your project will continue to accrue costs as long as it is running.

Follow these steps to stop and delete the notebook.

Go back to the Amazon SageMaker console.
Select the notebook and click Actions.
Select Actions
Select Stop and wait for the instance to stop.
Select Stop
Select Delete

### Recap
In this demo we learned how to setup a Jupyter notebook in Amazon SageMaker, about machine learning code in the real world, and what data preparation, model training, and model evaluation can look in a notebook instance. While this was a fun use case for us to explore, the concepts and techniques can be applied to other machine learning projects like an object detector or a sentiment analysis on text.