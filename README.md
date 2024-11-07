AI-Generated Text Detector


An AI-generated text detector in SML project helps to identify whether the text or paragraph was written by a human or copied or inspired by an AI engine. The fundamental idea includes creating a detector on text features that compares the AI-generated content from human-written content.

The basic steps include the:
1.)Data Collection: In this step we collect a dataset with labeled examples of both human-written and AI-generated text.
Dataset used in this project:

from datasets import load_dataset

ds = load_dataset("artem9k/ai-text-detection-pile")

