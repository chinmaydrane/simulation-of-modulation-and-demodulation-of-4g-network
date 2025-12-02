# filename: app.py

from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go
from simulation import run_simulation, run_ber_sweep, qam_mapper
import numpy as np

app = Flask(__name__)

# --- Plotly Helper Functions ---

def create_constellation_plot(I, Q, M):
    """Generates the Plotly figure for the constellation diagram."""
    
    M = int(M)
    
    fig = go.Figure(data=[
        go.Scatter(x=I, y=Q, mode='markers', name='Received Symbols', 
                   marker=dict(size=4, opacity=0.8, color='blue'))
    ])
    
    # 2. Add Reference Constellation Points (Using the corrected bit generation)
    k = int(np.log2(M))
    
    ideal_bits_for_ref = np.concatenate([
        list(map(int, bin(i)[2:].zfill(k))) for i in range(M)
    ])
    
    ref_symbols = qam_mapper(ideal_bits_for_ref, M) 
    
    fig.add_trace(go.Scatter(
        x=np.real(ref_symbols), 
        y=np.imag(ref_symbols), 
        mode='markers', 
        name='Ideal Points',
        marker=dict(symbol='x', size=10, color='red', line=dict(width=2))
    ))
    
    fig.update_layout(
        title=f'Constellation Diagram ({M}-QAM)',
        xaxis_title='In-Phase (I)',
        yaxis_title='Quadrature (Q)',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_json()

def create_time_signal_plot(time_data):
    """Generates the Plotly figure for the Time Domain Signal."""
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_data['samples'], y=time_data['tx_I'], mode='lines', name='TX Signal (Real)',
        line=dict(color='green', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_data['samples'], y=time_data['rx_I'], mode='lines', name='RX Signal (Real)',
        line=dict(color='purple', width=1)
    ))

    fig.update_layout(
        title='Time Domain Signal (TX vs RX)',
        xaxis_title='Sample Index',
        yaxis_title='Amplitude',
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_json()


def create_ber_plot(snrs, sim_bers, theo_bers, M):
    """Generates the Plotly figure for the BER vs SNR curve."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=snrs, y=sim_bers, mode='lines+markers', name='Simulated BER', 
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=snrs, y=theo_bers, mode='lines', name='Theoretical BER (Approx)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'BER vs $E_s/N_0$ for {M}-QAM',
        xaxis_title='$E_s/N_0$ (dB)',
        yaxis_title='Bit Error Rate (BER)',
        yaxis_type='log', 
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_json()

# --- NEW: Carrier Mixing & Separation Plot ---
def create_carrier_mixing_plot(mixing_data):
    """Generates the Plotly figure showing carrier, mixed signal, and recovered signal."""
    fig = go.Figure()

    # Actual/baseband signal
    fig.add_trace(go.Scatter(
        x=mixing_data['samples'],
        y=mixing_data['baseband'],
        mode='lines',
        name='Baseband Signal',
        line=dict(width=1, color='blue')
    ))

    # Carrier signal (different color)
    fig.add_trace(go.Scatter(
        x=mixing_data['samples'],
        y=mixing_data['carrier'],
        mode='lines',
        name='Carrier Signal',
        line=dict(width=1, color='red')
    ))

    # Mixed (upconverted)
    fig.add_trace(go.Scatter(
        x=mixing_data['samples'],
        y=mixing_data['mixed'],
        mode='lines',
        name='Mixed (Baseband × Carrier)',
        line=dict(width=1)
    ))

    # Recovered (downconverted)
    fig.add_trace(go.Scatter(
        x=mixing_data['samples'],
        y=mixing_data['recovered'],
        mode='lines',
        name='Recovered (Mixed × Carrier)',
        line=dict(width=1, dash='dash')
    ))

    fig.update_layout(
        title='Carrier Mixing and Separation',
        xaxis_title='Sample Index',
        yaxis_title='Amplitude',
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_json()

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main simulation UI page."""
    return render_template('index.html')

@app.route('/run_sim', methods=['POST'])
def run_sim_endpoint():
    """Endpoint to run a single simulation and return plot data."""
    try:
        data = request.get_json()
        M_qam = int(data['qam_order'])
        snr_db = float(data['snr_db'])
        num_bits = int(data['num_bits'])
        
        results = run_simulation(M_qam, snr_db, num_bits)
        
        constellation_json = create_constellation_plot(
            results['constellation_data']['I'], 
            results['constellation_data']['Q'], 
            M_qam
        )
        time_signal_json = create_time_signal_plot(results['time_data'])

        # NEW: handle mixing_data safely
        mixing_data = results.get('mixing_data')
        if mixing_data is not None:
            carrier_mixing_json = create_carrier_mixing_plot(mixing_data)
        else:
            carrier_mixing_json = None

        return jsonify({
            'status': 'success',
            'ber': f"{results['BER']:.6f}",
            'constellation_plot': constellation_json,
            'time_signal_plot': time_signal_json,
            'carrier_mixing_plot': carrier_mixing_json  # may be null if something missing
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/run_ber_sweep', methods=['POST'])
def run_ber_sweep_route():
    """Endpoint to run the BER sweep and return plot data."""
    try:
        data = request.get_json()
        M_qam = int(data['qam_order'])
        num_bits = int(data['num_bits'])
        
        results = run_ber_sweep(M_qam, num_bits)
        
        ber_plot_json = create_ber_plot(
            results['Es_N0_dBs'],
            results['simulated_bers'],
            results['theoretical_bers'],
            M_qam
        )
        
        return jsonify({
            'status': 'success',
            'ber_plot': ber_plot_json
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
