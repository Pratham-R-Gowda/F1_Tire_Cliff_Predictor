🛞 F1 Tire Cliff Predictor: Finding the True Limit

👋 Hey there! Welcome to the Pit Wall.

If you're cloning this repository, you probably already know that Formula 1 is a game of data. But when I first started building this Machine Learning model to predict when a driver's tires would "fall off the cliff," the data lied to me.

I fed raw lap times into a regression model, and the model predicted that 30-lap-old tires were actually getting faster. Why? Because I forgot about basic physics.

🏎️ The Engineering Problem: The "Deceptive" Curve

Every lap, a Formula 1 car burns about 1.5kg of fuel. As the car gets lighter, it naturally gets faster. In many stints, this weight loss completely masks the fact that the rubber is degrading. The car is shedding weight faster than it is losing grip.

If you feed raw lap times to a Machine Learning algorithm, it will blindly draw a curve showing the car speeding up forever. If a strategist uses that model, they will leave the driver out until the tire literally explodes.

🧠 The Solution: Fuel Correction & Polynomial Regression

To build a tool that actually works, I had to apply Data Engineering before applying Machine Learning:

The Fuel Correction: I wrote a pipeline that mathematically adds a time penalty (e.g., 0.08 seconds per lap) back onto the raw lap times. This artificially simulates the car staying at its heavy starting weight, stripping away the fuel advantage to reveal the true, isolated rubber degradation.

Polynomial Regression: Tires do not degrade in a straight line, so standard LinearRegression fails here. I utilized Scikit-Learn's PolynomialFeatures (Degree 2) to allow the algorithm to draw a parabolic curve. This perfectly maps the F1 tire "working window"—staying flat for 15 laps before aggressively curving upwards to pinpoint the exact lap the "cliff" hits.

🛠️ The Tech Stack

UI & Deployment: Streamlit

Machine Learning: Scikit-Learn (Polynomial Regression Pipelines)

Data Extraction: FastF1 API

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

🚀 How to Run This on Your Machine

I've set this up as an interactive Streamlit dashboard so you can act as the strategist. You can tweak the fuel penalty slider yourself and watch the true tire cliff reveal itself in real-time.

1. Clone the repo:

git clone https://github.com/Pratham-R-Gowda/F1_Tire_Cliff_Predictor.git
cd repo-name

2. Install the F1 Data Stack:

python -m pip install -r requirements.txt

3. Launch the Dashboard:

python -m streamlit run tire_app.py

(Note: The first time you select a new race or driver, the FastF1 API will take a few seconds to download the telemetry cache).

💡 What to look for when you run it

When the app opens, look at the graph. The gray dots show the deceptive raw data. The cyan dots and thick red line show the true physics of the tire. Notice how the red line clearly identifies the exact lap the driver hits peak performance, and exactly when it's time to box!
