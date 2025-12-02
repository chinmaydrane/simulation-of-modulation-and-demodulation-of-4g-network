📡 4G (LTE) PHY Layer Web Simulator

This project is a web-based 4G (LTE) physical layer simulation focused on:

QAM modulation/demodulation (4/16/64-QAM)

OFDM transmitter/receiver (IFFT/FFT + Cyclic Prefix)

AWGN channel

BER computation & BER vs SNR (Es/N0) curve

Visualization of:

Constellation diagram

Time-domain TX vs RX signal

Carrier mixing and separation (baseband × carrier)

The user interacts through a simple web UI (HTML + JavaScript), while the backend is implemented using Flask and NumPy/Plotly.

🧱 1. Project Structure (Minute Detail)

Typical folder layout:

CN_CP2/
│
├─ app.py          # Flask web server (backend + Plotly figure creation)
├─ simulation.py    # Core PHY simulation (QAM, OFDM, AWGN, BER, mixing)
├─ templates/
│   └─ index.html  # Frontend UI (HTML + JS + Plotly)


Flask uses the templates/ folder for HTML templates by default.
app2.py will render index2.html from there.

🧩 2. Tech Stack

Python 3.x

Flask – web framework

NumPy – numerical computations (signals, OFDM, noise)

SciPy (erfc) – theoretical BER for QAM

Plotly – interactive plots (constellation, time-domain, BER, mixing)

HTML + JS (fetch API) – frontend interaction

CSS (inline in HTML) – basic styling

⚙️ 3. Installation & Setup

Create environment (optional but recommended)

python -m venv venv
venv\Scripts\activate   # on Windows


Install dependencies

pip install flask numpy scipy plotly


Place files correctly

root/
    app.py
    simulation.py
    templates/
        index.html

▶ 4. How to Run the Simulator (Step-by-Step)

Open a terminal/command prompt.

Navigate to the project folder:

cd path\to\root


Run the Flask app:

python app.py


You should see something like:

* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5000


Open a browser and go to:

http://127.0.0.1:5000/


The UI will appear with:

Simulation flow diagram

Parameter controls (QAM order, Es/N0, number of bits)

Buttons:

Start Simulation (Plots & BER)

Plot BER vs SNR Curve (Sweep)

4 plot panels:

Constellation Diagram

Time Domain TX vs RX

Carrier Mixing & Separation

BER vs SNR
