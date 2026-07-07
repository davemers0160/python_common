import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def parse_value(val_str):
    """Safely converts strings like '10 nH' or '50 Ohm' into float objects."""
    try:
        parts = str(val_str).lower().split()
        number = float(parts[0])
        if len(parts) > 1:
            unit = parts[1]
            if 'p' in unit: number *= 1e-12
            elif 'n' in unit: number *= 1e-9
            elif 'u' in unit or 'µ' in unit: number *= 1e-6
            elif 'm' in unit and 'ohm' not in unit: number *= 1e-3
            elif 'k' in unit: number *= 1e3
            elif 'meg' in unit: number *= 1e6
        return number
    except (ValueError, IndexError):
        return 0.0

def run_ac_analysis(netlist, start_hz=1e6, stop_hz=1e9, points=200):
    # 1. Validation: Ensure we have the required VNA ports
    has_p1 = any(c['type'] == 'PORT1' for c in netlist)
    has_p2 = any(c['type'] == 'PORT2' for c in netlist)
    
    if not (has_p1 and has_p2):
        raise Exception("Schematic must include both PORT 1 and PORT 2 to simulate.")

    # 2. Setup Circuit
    circuit = Circuit('RF Simulation')
    
    # 3. Build Components
    # We ignore the 'GROUND' type here because PySpice handles it via node 0
    for i, comp in enumerate(netlist):
        t = comp['type']
        n1 = comp['node_a']
        n2 = comp['node_b']
        
        if t == 'PORT1':
            # VNA Port 1: Source (50 Ohm characteristic impedance)
            circuit.SinusoidalVoltageSource('V1', 'port1_internal', circuit.gnd, ac_magnitude=1@u_V)
            circuit.R('Rs', 'port1_internal', n1, 50@u_Ohm)
            
        elif t == 'PORT2':
            # VNA Port 2: Load (50 Ohm characteristic impedance)
            circuit.R('Rl', n1, circuit.gnd, 50@u_Ohm)
            
        elif t in ['L', 'C', 'R']:
            val = parse_value(comp['value'])
            # Only add if nodes are valid
            if n1 is not None and n2 is not None:
                if t == 'L': circuit.L(f"L{i}", n1, n2, val@u_H)
                elif t == 'C': circuit.C(f"C{i}", n1, n2, val@u_F)
                elif t == 'R': circuit.R(f"R{i}", n1, n2, val@u_Ohm)

    # 4. Run Analysis
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.ac(start_frequency=start_hz@u_Hz, stop_frequency=stop_hz@u_Hz, 
                            number_of_points=points, variation='dec')
    
    # 5. Extract Data (Gain in dB)
    freqs = analysis.frequency
    # Gain calculation (S21 approx)
    p2_voltage = analysis['port2_node'] # Ensure this matches your circuit node name
    gain = 20 * np.log10(np.abs(p2_voltage) + 1e-12)
    
    return freqs, gain