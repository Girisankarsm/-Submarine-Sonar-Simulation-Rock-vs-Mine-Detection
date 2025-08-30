:

🎯 Submarine Sonar Simulation: Rock vs Mine Detection
Overview

Simulates a submarine sonar system to classify underwater objects as Rocks 🟢 or Mines 💣 using machine learning. Trains a RandomForestClassifier on the UCI Sonar dataset and presents an interactive terminal animation for a cinematic demonstration.

Features

Data Handling & Validation

Automatic detection of CSV or whitespace-separated datasets.

Ensures all features are numeric and labels are valid.

Machine Learning Model

Trains RandomForest classifier.

Evaluates using accuracy, confusion matrix, and classification report.

Saves the trained model for reuse.

Interactive Terminal Simulation

Submarine 🚢 moves across a grid dynamically.

Rocks 🟢 and Mines 💣 move randomly with smooth animation.

Sonar detection radius highlights nearby objects.

Adjustable parameters for demo customization.

Optional Parameters

The simulation can be customized directly in p.py:

Parameter	Description	Default
width	Width of the ocean grid (number of columns)	40
height	Height of the ocean grid (number of rows)	10
num_scans	Total number of simulation steps	80
delay	Time delay between frames (seconds)	0.2

Modify these in the interactive_ocean() function call to adjust speed, grid size, and number of scans for your presentation.

Requirements

Python 3.10+

Libraries: pandas, numpy, scikit-learn, joblib, rich

Install dependencies:

pip install pandas numpy scikit-learn joblib rich

Usage

Clone or download the repository:

git clone <repository_url>
cd "machine learning projet"


Activate the Python environment:

source sonar_env/bin/activate


Run the script:

python p.py


Trains and evaluates the model.

Launches interactive terminal ocean simulation with moving submarine, rocks, and mines.

Notes

Simulation only: For educational or demo purposes; does not represent real sonar hardware.

Customizable animation: Adjust width, height, num_scans, and delay for presentations.

Portfolio-ready: Ideal for showcasing ML skills, terminal animation, and interactive demos.
