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