# filename: simulation.py

import numpy as np
from scipy.special import erfc 

# --- NEW Plotting Constant for Optimization ---
MAX_PLOT_SYMBOLS = 2000  # Limit the number of symbols sent to the browser for plotting speed.

# --- 1. QAM Modulation/Demodulation (Supports M=4, 16, 64) ---

def qam_mapper(bits, M):
    """Maps bits to M-QAM constellation symbols."""
    k = int(np.log2(M))  # bits per symbol
    bits = np.array(bits, dtype=int)
    
    if len(bits) % k != 0:
        bits = np.concatenate((bits, np.zeros(k - len(bits) % k, dtype=int)))
        
    symbols = []
    
    if M == 4:
        d = 1.0 / np.sqrt(2) 
    elif M == 16:
        d = 1.0 / np.sqrt(10)
    elif M == 64:
        d = 1.0 / np.sqrt(42)
    else:
        raise ValueError("Unsupported M-QAM order.")

    for i in range(0, len(bits), k):
        bit_str = ''.join(map(str, bits[i:i+k]))
        decimal = int(bit_str, 2)
        
        # Calculate I and Q indices based on the decimal index
        if M == 4:
            i_idx = 2*(decimal >> 1) - 1 
            q_idx = 2*(decimal & 1) - 1
        elif M == 16:
            i_idx = 2*((decimal >> 2) & 3) - 3 
            q_idx = 2*(decimal & 3) - 3      
        elif M == 64:
            i_idx = 2*((decimal >> 3) & 7) - 7
            q_idx = 2*(decimal & 7) - 7
        
        symbols.append((i_idx + 1j*q_idx) * d)
        
    return np.array(symbols)

def qam_demapper(rx_symbols, M):
    """Demaps received symbols back to bits using minimum distance."""
    k = int(np.log2(M))
    
    ideal_bits_for_ref = np.concatenate([list(map(int, bin(i)[2:].zfill(k))) for i in range(M)])
    constellation = qam_mapper(ideal_bits_for_ref, M)[:M] 
    
    bit_map = {i: list(map(int, bin(i)[2:].zfill(k))) for i in range(M)}
    
    demapped_bits = []
    for symbol in rx_symbols:
        distances = np.abs(symbol - constellation)
        min_index = np.argmin(distances)
        demapped_bits.extend(bit_map[min_index])
    
    
    return np.array(demapped_bits)

# --- 2. OFDM Transmitter/Receiver ---

def ofdm_tx(qam_symbols, N_sc, N_cp):
    """Applies IFFT and Cyclic Prefix (CP) for OFDM."""
    num_blocks = int(np.ceil(len(qam_symbols) / N_sc))
    padded_symbols = np.pad(qam_symbols, (0, num_blocks * N_sc - len(qam_symbols)), 'constant')
    symbol_blocks = padded_symbols.reshape(num_blocks, N_sc)
    
    ofdm_blocks_time = np.fft.ifft(symbol_blocks, axis=1) * N_sc
    ofdm_tx_signal = np.hstack((ofdm_blocks_time[:, -N_cp:], ofdm_blocks_time))
    ofdm_tx_signal = ofdm_tx_signal.flatten()
    
    return ofdm_tx_signal, len(qam_symbols)

def ofdm_rx(rx_signal, N_sc, N_cp, original_symbol_len):
    """Removes CP and applies FFT for OFDM reception."""
    samples_per_block = N_sc + N_cp
    num_blocks = int(len(rx_signal) / samples_per_block)
    
    rx_blocks = rx_signal[:num_blocks * samples_per_block].reshape(num_blocks, samples_per_block)
    rx_blocks_no_cp = rx_blocks[:, N_cp:]
    
    rx_symbols_freq = np.fft.fft(rx_blocks_no_cp, axis=1) / N_sc
    rx_symbols = rx_symbols_freq.flatten()
    
    return rx_symbols[:original_symbol_len]

# --- 3. Channel and Metrics ---

def awgn_channel(signal, Es_N0_dB):
    """Adds Additive White Gaussian Noise (AWGN)."""
    Es_N0_linear = 10**(Es_N0_dB / 10)
    avg_symbol_energy = np.mean(np.abs(signal)**2)
    noise_variance = avg_symbol_energy / Es_N0_linear
    noise_std = np.sqrt(noise_variance / 2)
    
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise

def calculate_ber(tx_bits, rx_bits):
    """Calculates Bit Error Rate (BER)."""
    min_len = min(len(tx_bits), len(rx_bits))
    errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    ber = errors / min_len
    return ber

def theoretical_ber(M_qam, Es_N0_dBs):
    """Calculates theoretical BER for M-QAM in AWGN (Simplified Approximation)."""
    k = np.log2(M_qam)
    if k == 0:
        return np.zeros_like(Es_N0_dBs)
    
    Es_N0_linear = 10**(Es_N0_dBs / 10)
    
    if M_qam == 4:
        return erfc(np.sqrt(Es_N0_linear)) / 2 

    return (2/k) * (1 - 1/np.sqrt(M_qam)) * erfc(np.sqrt(3*k/(2*(M_qam-1)) * Es_N0_linear))

# --- 4. Main Simulation Functions for Flask App ---

def run_simulation(M_qam, Es_N0_dB, num_bits, N_sc=64, N_cp=16):
    """End-to-end simulation for a single SNR point (FULL OFDM LINK)."""
    
    tx_bits = np.random.randint(0, 2, num_bits)
    print("Generated Bits:", tx_bits[:200], "...")
    tx_symbols = qam_mapper(tx_bits, M_qam)
    
    ofdm_tx_signal, original_symbol_len = ofdm_tx(tx_symbols, N_sc, N_cp)
    rx_ofdm_signal = awgn_channel(ofdm_tx_signal, Es_N0_dB)
    rx_symbols = ofdm_rx(rx_ofdm_signal, N_sc, N_cp, original_symbol_len)
    rx_bits = qam_demapper(rx_symbols, M_qam) 
    print("Received Bits (first 200):", rx_bits[:200], "...\n")
    ber = calculate_ber(tx_bits, rx_bits)

    # Prepare data for Plotly: Apply Optimization Limit
    plot_symbols = rx_symbols[:MAX_PLOT_SYMBOLS]
    
    constellation_data = {
        'I': np.real(plot_symbols).tolist(),
        'Q': np.imag(plot_symbols).tolist()
    }
    
    # Prepare data for Time Domain Signals (Still limited to few symbols)
    max_samples = 5 * (N_sc + N_cp)
    time_data_len = min(len(ofdm_tx_signal), max_samples)

    time_data = {
        'tx_I': np.real(ofdm_tx_signal[:time_data_len]).tolist(),
        'tx_Q': np.imag(ofdm_tx_signal[:time_data_len]).tolist(),
        'rx_I': np.real(rx_ofdm_signal[:time_data_len]).tolist(),
        'rx_Q': np.imag(rx_ofdm_signal[:time_data_len]).tolist(),
        'samples': list(range(time_data_len))
    }

    # --- NEW: Data for Carrier Mixing & Separation Visualization ---
    carrier_len = time_data_len
    n = np.arange(carrier_len)
    omega_c = 2 * np.pi * 0.05  # arbitrary normalized carrier freq for visualization

    baseband = np.real(ofdm_tx_signal[:carrier_len])
    carrier = np.cos(omega_c * n)
    mixed = baseband * carrier
    recovered = mixed * carrier

    mixing_data = {
        'samples': list(range(carrier_len)),
        'baseband': baseband.tolist(),
        'carrier': carrier.tolist(),
        'mixed': mixed.tolist(),
        'recovered': recovered.tolist()
    }
    
    return {
        'BER': ber,
        'constellation_data': constellation_data,
        'time_data': time_data,
        'mixing_data': mixing_data
    }

def run_ber_sweep(M_qam, num_bits_per_point=10000):
    """Runs BER vs SNR sweep (Simplified link for speed)."""
    Es_N0_dBs = np.arange(0, 15, 1).tolist()
    simulated_bers = []
    
    for snr_db in Es_N0_dBs:
        tx_bits = np.random.randint(0, 2, num_bits_per_point)
        tx_symbols = qam_mapper(tx_bits, M_qam)
        
        # --- OPTIMIZATION: Skip OFDM for fast BER sweep calculation ---
        rx_symbols = awgn_channel(tx_symbols, snr_db)  # Apply AWGN directly to symbols
        
        rx_bits = qam_demapper(rx_symbols, M_qam)
        ber = calculate_ber(tx_bits, rx_bits)
        simulated_bers.append(ber)

    theoretical_bers = theoretical_ber(M_qam, np.array(Es_N0_dBs)).tolist()
    
    return {
        'Es_N0_dBs': Es_N0_dBs,
        'simulated_bers': simulated_bers,
        'theoretical_bers': theoretical_bers
    }
