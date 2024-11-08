# Dysarthric speech recognition
This project repo was developed for my master's in AI capstone project course (A-I 894 Autumn 2024 semester) through Penn State Great Valley. The Jupyter notebooks are intended to demonstrate the end-to-end development process of a working DSR model, which was evaluated with data collected at the Penn State Health Hershey Medical Center in 2023 (dataset henceforth referred to as PSUH*). Furthermore, the _psudsr_ module is intended to be used as an imported module to expedite common tasks such as data exploration, data processing (STFT features), dataset creation, and model inferencing. 

### Technical Approach
Dysarthric speech is a common symptom of patients who receive anesthesia. Thus, it is desirable for clinicians to use an AI speech recognition model as an non-invasive diagnostic tool to assess the cognition of patients recovering from anesthetics after surgical procedures. Audio data from patients undergoing anesthesia treatment were voluntarily recorded speaking the word "Pennsylvania" prior to treatment ("pre"), after treatment ("post"), and at discharge ("dc"). Initially, a severity-level classification model was desired but classification performance was quite poor with the data on hand. Thus, the PSUH examples were reclassed as healthy (pre & dc) vs. dysarthric (post). STFT features were extracted from each sample spectrogram and converted to tensors for input with the PyTorch model. 

#### Disclaimer
*Due to patient privacy laws, I am not able to freely share the PSUH dataset and the weights for the ResNet18 model I trained.
