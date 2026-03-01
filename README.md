# code_grader
A generic quiz utility to evaluate responses to coding problems and score them using Claude Sonnet 4.6

# To run: 
1. make sure that you have an Anthropic API key
2. check requirements.txt and install anything missing
3. set the environment variable for your API key. Something along the lines of: export ANTHROPIC_API_KEY="sk-ant-..."
4. start the server: "python manage.py runserver"
5. you can input questions and criteria at http://127.0.0.1:8000

Note: Added 70 python programming challenge examples, a separate category for ML/AI theory, and some sample questions in it, along with the abilty to submit inline LaTeX formulas in answers. 

