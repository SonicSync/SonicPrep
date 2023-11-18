# SonicPrep

## Project Plan - SonicPrep v1.0.0

### Project Overview

#### Objective
SonicPrep will be a simple and effective means of performing feature extraction, labeling and loading of audio ready for Machine Learning and analysis applications.

#### Target Audience
SonicPrep will allow users with little-to-intermediate knowledge of python extract meaningful metrics from an audio dataset.

### Features
- Flexible audio input handling (can handle .wav, .mp3, .flac codecs)
- Batch Processing
- Automated usage (simply insert params and run, to obtain output data)
- Data labelling across batches
- Audio normalization and standardization operations
- Metatag persistence
- Audio augmentation

### Stack

#### Languages

- Python
  - Language I am most knowledgable in.
  - OOP principles provide a structured and modular approach to code design, enhancing flexibility.
  - Python's rich library ecosystem allows for efficient development, especially in the domain of audio processing.
#### Libraries/Frameworks

- Librosa
  - Librosa is well-suited for audio feature extraction and standardization tasks.
  - Tailored to tasks related to audio and music processing.
- Pandas
  - Pandas simplifies data manipulation and organization.
  - Facilitates easy conversion of data into various formats like CSV or Excel.
 
### Storage & Data Handling

#### Input

- Supporting .wav, .mp3, and .flac files provides flexibility for users dealing with various types of audio data.

#### Internal

- Pandas provides efficient data manipulation capabilities and is widely used for handling structured data in Python.

#### Output

- Offering output in common formats like CSV and Excel (xlsx) is beneficial. These formats are widely used and can be easily imported into various data analysis tools.
- Including JSON as an output format provides additional flexibility, especially for users who prefer or need data in this format for specific workflows.

### Development Plan

SonicPrep will be developed using Test-Driven-Development principles in this order:

1. Creation of IO module 
3. Creation Audio Normalization module
4. Creation of augmentation module
5. Creation of Feature Extraction module
6. Creation of Workflow module to provide provide a cohesive, simple workflow.
