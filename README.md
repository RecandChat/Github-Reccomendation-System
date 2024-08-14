<p align="center">
  <img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  <img src="https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white" alt="ChatGPT" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="SciKitLearn" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="streamlit" />
</p>

# CodeCompass

## Overview

**CodeCompass is a Chabot and Reccomendation engine** that aims to enhance the GitHub experience for developers and learners alike by providing a personalized interface to discover and interact with repositories. Our goal is to make learning and discovery in GitHub intuitive, informative, and tailored to each user's interests and needs.

### Personalized Recommendations
Utilizing GitHub Data and Machine Learning algorithms, the system suggests GitHub projects that resonate with the user's historical interactions and preferences.

<p align="center">
  <img src="https://github.com/RecandChat/CodeCompass/assets/99414447/9b53abfd-d06f-43b4-a1c4-63fce2c5b7d8" />
</p>

For a full Recommender demonstration amd model explanation, watch our [video](https://www.youtube.com/watch?v=SPVVdKDqeag).

### Interactive Exploration and Learning
The embedded chatbot enhances repository exploration, providing valuable insights and fostering a conducive exploration environment.

<p align="center">
  <img src="https://github.com/RecandChat/CodeCompass/assets/99414447/2eaa069c-fd50-4518-bde3-3661a8d61061" />
</p>

For a full Chabot demonstration, watch our [video](https://youtu.be/iaSOjpxsE7s?si=-GKZ6iMEcsuDD4SD).

## Usage

Follow these steps to get CodeCompass up and running on your local machine:

### **1.** Clone the repo:
```
git clone https://github.com/RecandChat/CodeCompass
```
### **2.** Change to the project directory:
```
cd CodeCompass
```
### **3.** Create and activate a Virtual Environment


### **4.** Install dependencies:
```
pip install -r requirements.txt
```
### **5.** Create Secrets Diretory at the root (`secrets/`)
- `github_token`: Your GitHub API Token.
- `instructions`: Your chatbot system prompt.
- `openAI_key`: Your OpenAI API key.
### **4.** Run:
  
**Chatbot**
```
streamlit run frontend/chatbot/app.py
```
**Recommender:**
```
redis-server
```
Check server is running (Outputs pong if connected)
```
redis-cli ping
```
Check database by
```
redis-cli

KEYS *
```
```
streamlit run frontend/recommender/app.py
```
## Further Documentation
For more in-depth information, refer to the `documentation/Final_Report` directory within the repository.

## Contributors
- [Luca Cuneo](https://github.com/Lukasaurus11)
- [Gabriel de Olaguibel](https://github.com/gabrieldeolaguibel)
- [Keti Sulamanidze](https://github.com/KTsula)
- [Felix Gomez-Guillamon](https://github.com/felixggj)
- [Maud Helen Hovland](https://github.com/maudhelen)
- [Miranda Drummond](https://github.com/mirandadrummond)
