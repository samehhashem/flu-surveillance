# flu-surveillance
This was a course project, the aim is to create a twitter-based flu surveillance system.

The underlying idea is that many people with flu-like symptoms tweet about there symptoms
or a related consequence. By monitoring twitter streams and identifying/tallying flu-related
tweets, it is possible to build a simple flu serveillance system.

Two Step process: 1) Tweet Labelling 2)Building a flu serveillance system

Tweet Labelling 
~1000 tweet were collected using the code twitter_streaming.py
Then tweets were labeled manually.
10% positive and 90% negative.

Flu Serveillance System
Build and train a classifier using the data collected from step 1.
All code is in the file named flu_20466546.py
Trained classifier is in a file named flu_classifier_20466546.pkl
