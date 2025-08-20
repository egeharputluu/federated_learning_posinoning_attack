# Federated Learning Poisoning Attack Simulation

This repository demonstrates a **Federated Learning Poisoning Attack** in a simulated private 5G/IoT environment.  
The project shows how **one compromised client** can reduce the overall federated learning accuracy and how an attacker could inject poisoned datasets via SSH access.

---

## Project Structure

├── client_1.py # Local training for Client 1
├── client_2.py # Local training for Client 2
├── client_3.py # Local training for Client 3 (supports poisoned dataset)
├── federated_learner.py # Aggregates client reports and calculates federated score
├── brute_force_attack.py # SSH brute force script (simulated attacker)
├── data/ # Datasets for each client (replace client3.csv with poisoned dataset)
├── models/ # Saved local Random Forest models
└── reports/ # Training reports + federated results



## How It Works

- Each client (`client1`, `client2`, `client3`) trains a **Random Forest Classifier** on its local dataset.
- Reports include accuracy, classification report, and dataset details.
- The **Federated Learner** (`federated_learner.py`) aggregates client results by averaging their accuracies.
- **Poisoning Effect**: By replacing `client3.csv` with a **poisoned dataset**, the global federated score decreases, demonstrating the attack.
- **SSH Brute-Force**: `brute_force_attack.py` simulates an attacker cracking an SSH password and injecting poisoned data into the client’s local dataset:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.
- **Fake SSH Service**: `client3.py` includes an optional **fake TCP server** on port `2222` to mimic OpenSSH banners for scanning tools like Nmap.

---

## Installation

Clone the repository and install dependencies:

git clone https://github.com/egeharputluu/federated_learning_posinoning_attack
cd federated-poisoning-sim

pip install -r requirements.txt
Dependencies include:

numpy

pandas

scikit-learn

joblib

paramiko

▶️ Usage
1. Train Clients Locally
python client_1.py
python client_2.py
python client_3.py
Each client will:

Train a Random Forest model

Save the model in models/

Generate a report JSON in reports/

2. Run Federated Learner
python federated_learner.py
This will:

Collect accuracy reports from all clients

Compute the Federated Score (mean client accuracy)

Save the result in reports/federated_report.json

3. Simulate Poisoning
Replace Client 3’s dataset with a poisoned dataset:
cp data/poisoned_client3.csv data/client3.csv
python client_3.py
python federated_learner.py
You should see the federated score drop, proving the impact of data poisoning.

4. Run SSH Brute Force Attack
python brute_force_attack.py
This script attempts to brute-force an SSH server to simulate how an attacker could remotely access a device and poison its local dataset.

5. Fake SSH Banner (Optional)
To mimic SSH service visibility for Nmap scans, enable the fake TCP server:
START_FAKE_TCP=1 python client_3.py
This will open a banner service on port 2222:

nmap -sV -p 2222 127.0.0.1
🎯 Project Goals
Demonstrate how a single poisoned client can compromise federated learning accuracy.

Show how SSH brute-force attacks may serve as an entry point for poisoning.

Provide a hands-on simulation of security risks in federated learning environments.

⚠️ Disclaimer
This project is for educational and research purposes only.
Do not use the brute-force code against systems you do not own or have explicit permission to test.
