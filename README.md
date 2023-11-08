# AICANWRITE
kaggle poc to differentiate LLM base creation from Human creation

This is poc to determine wether or not a text habeen generated by AI or human.

Author Dzoan Steven  Tran 
status : poc  

Using first principle reasoning , let s ask 

how does human write ? what motivate human to write ?
What make then different ?


here are tendencies in the way each might approach storytelling:

-LMs: Often narrate from an external point of view because their knowledge is derived from a vast array of texts that cover different perspectives and styles. They tend to construct stories based on patterns learned from these texts, which can sometimes result in a more detached or objective narration. LLMs focus on constructing narratives that adhere to established norms and structures of storytelling.

Humans Stories have an internal  Voice 
their stories with personal experiences, emotions, and idiosyncrasies that come from their unique lives. Human writers can convey a deep sense of individual perspective, with all the subjective complexities and emotional depths that come with personal experience. They might focus on the internal landscapes of their characters, creating a narrative that feels more personal or emotionally resonant.

Internal Human  voice  Narration  versus LLM external voice narration 

The distinction lies in the richness and authenticity of personal experience that humans can bring to a story, which is something LLMs simulate based on patterns rather than experience. Humans can draw upon their own memories, emotions, and sensory experiences, which can lead to a story feeling more "lived-in" or authentic from an emotional standpoint.
 

So now we have what we are looking , how can we classify text with a probability to belong to LLM or Human ? 

My Model 
LLM : Extenal point of view  , neutral  feelings  narrative 
Human  : inner voice  personnal narration with feelings , experience, families , history , memories 

How ? 



Lets break it down  on how to determine the Human internal POV point of view Narrative or External POS from LLM model. 

TOP Down approach 

WHAT  are wee looking for ? 
Internal POV  Human narration versus Extenal POV narration LLM 

HOW ?   

Detecting the narrative point of view in text—a story told from an external (LLM ) versus an internal perspective ( Human) —

 Here are some models and techniques that could be used for such a classification:

1.   Rule-Based Systems  : These systems would look for specific keywords and phrases that are indicative of internal or external narration. For example, the first-person pronouns ("I", "me", "my") might suggest an internal point of view, while third-person pronouns ("he", "she", "they") and more observational language might suggest an external point of view.

2.   Machine Learning Classifiers  : You could train a machine learning model on a labeled dataset where the narrative point of view is annotated. Models like Support Vector Machines (SVM), Naive Bayes, or Logistic Regression could be used if you can extract good feature representations of the texts.

3.   Deep Learning Models  : More complex models like Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), or Transformers (like BERT or GPT) could be trained to understand the context better and make more accurate predictions. These models are good at capturing sequential data and long-range dependencies within text, which is useful in determining narrative style.

possible models and why using them :

   a. Transformer-Based Models (like BERT, GPT, or RoBERTa)
 
- For  Contextual Understanding  : Transformer models like BERT and RoBERTa understand the context of words in text by considering the entire sequence, rather than just individual words. This allows them to capture nuances in narrative style and sentiment.
-   Fine-Tuning Capability  : These models can be fine-tuned on a specific dataset, which allows them to adapt to the specific characteristics of human vs. AI-generated texts in your dataset.
-   Sentiment Analysis  : They have proven to be highly effective in sentiment analysis, which can help in distinguishing between neutral AI narratives and emotionally charged human stories.


   b. LSTM Networks with Attention Mechanism

  Justification:  
-   Sequential Data Processing  : LSTM networks are adept at processing sequences and can capture long-term dependencies, which can be crucial in understanding narrative flow.
-   Attention Mechanism  : When combined with an attention mechanism, LSTMs can focus on relevant parts of a text, which could be helpful in discerning the more personalized elements of human writing.



   b . Ensemble Learning with Meta-Classifiers

  Justification:  
-   Diverse Perspectives  : An ensemble of different models can capture a variety of linguistic features, which could be useful for a nuanced task like distinguishing between human and AI narratives.
-   Improved Performance  : Ensembles often outperform individual models by combining their strengths and mitigating their weaknesses.






4.   Transfer Learning  : Using a pre-trained model like BERT, which has been trained on a large corpus of text, can be fine-tuned for the specific task of classifying narrative perspective. The advantage here is that BERT is quite good at understanding context and nuance in language, which is crucial for this task.

5.   Natural Language Processing (NLP) Techniques  : Combining NLP techniques like sentiment analysis, Named Entity Recognition (NER), and Dependency Parsing could help in identifying the depth of character introspection versus external description.

To create a model for this purpose, you would need to:

-   Collect and Label Data  : Assemble a corpus of stories labeled with their point of view.
-   Feature Engineering  : Identify features that are indicative of internal or external narration.
-   Model Training  : Choose and train a suitable model on your labeled dataset.
-   Evaluation  : Test the model's performance on unseen data and iterate to improve.

Each of these steps requires a detailed understanding of both machine learning and narrative theory to ensure that the model can accurately distinguish between the two narrative styles. It's a challenging task because the distinction isn't always clear-cut and may require a nuanced understanding of language.


1. Define and calculate the ratio of internal point of view (POV) phrases in human texts compared to LLM texts using training datasets.
2. Determine the ratio by counting external and internal POV phrases in each text of the test dataset.
3. Evaluate the model's precision in distinguishing between LLM and human texts based on these ratios.

STEPS 

   1. Data Preparation:
- Load your dataset with `pandas`.
- Clean and preprocess the text data, potentially with TensorFlow or another text processing library if needed.

   2. Feature Engineering:
- Create a function to identify and count internal POV phrases (such as first-person narrative, subjective thoughts, etc.).
- Create another function to identify and count external POV phrases (such as objective descriptions, third-person narrative, etc.).
- Calculate the ratio of internal to external POV phrases for each text.

   3. Model Training:
- Use the calculated ratios as features for training your model.
- Train different models (like XGBoost, LightGBM, or CatBoost) using these features.
- Opt for ensemble methods to potentially increase prediction accuracy.

   4. Evaluation:
- Use a test dataset to evaluate the model's precision.
- Compare the model's predictions with the actual labels to determine precision.

   5. Result Compilation:
- Compile the results using `pandas` and save them for further analysis or reporting.







