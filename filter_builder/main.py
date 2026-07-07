import sys
import json
import numpy as np

# PyQt Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, 
                             QGraphicsView, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter

# Matplotlib PyQt Integration
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Custom Application Modules
from elements import (SchematicComponent, GroundSymbol, Terminal, 
                      ManhattanWire, VNAPort1, VNAPort2)
from engine import extract_netlist
from simulate import run_ac_analysis

class SchematicCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(-1000, -1000, 2000, 2000)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing if hasattr(QPainter.RenderHint, 'Antialiasing') else 0x01)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        self.drawing_wire = None

    def mousePressEvent(self, event):
        item = self.scene.itemAt(self.mapToScene(event.pos()), self.transform())
        
        if isinstance(item, Terminal):
            self.drawing_wire = ManhattanWire(item)
            self.drawing_wire.set_dynamic_end(self.mapToScene(event.pos()))
            self.scene.addItem(self.drawing_wire)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_wire:
            self.drawing_wire.set_dynamic_end(self.mapToScene(event.pos()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_wire:
            item = self.scene.itemAt(self.mapToScene(event.pos()), self.transform())
            
            if isinstance(item, Terminal) and item != self.drawing_wire.start_port:
                self.drawing_wire.set_end_port(item)
            else:
                # Cancel wire if dropped in empty space
                self.drawing_wire.start_port.remove_wire(self.drawing_wire)
                self.scene.removeItem(self.drawing_wire)
                
            self.drawing_wire = None
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        selected = self.scene.selectedItems()
        
        # Handle Rotation
        if event.key() == Qt.Key.Key_R:
            for item in selected:
                if isinstance(item, (SchematicComponent, GroundSymbol, VNAPort1, VNAPort2)):
                    item.setRotation((item.rotation() + 90) % 360)
                    
        # Handle Deletion
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            for item in selected:
                # If deleting a component, remove connected wires first
                if isinstance(item, (SchematicComponent, GroundSymbol, VNAPort1, VNAPort2)):
                    ports = []
                    if hasattr(item, 'port1'): ports.append(item.port1)
                    if hasattr(item, 'port2'): ports.append(item.port2)
                    
                    for port in ports:
                        for wire in port.wires[:]: # iterate over a copy
                            self.scene.removeItem(wire)
                            if wire.start_port: wire.start_port.remove_wire(wire)
                            if wire.end_port: wire.end_port.remove_wire(wire)
                
                # If explicitly deleting a wire
                if isinstance(item, ManhattanWire):
                    if item.start_port: item.start_port.remove_wire(item)
                    if item.end_port: item.end_port.remove_wire(item)
                    
                self.scene.removeItem(item)
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # 1. Map to Scene
        scene_pos = self.mapToScene(event.pos())
        # 2. Get the top-most item
        items = self.items(event.pos())
        if items:
            item = items[0]
            # 3. If it's part of a component, find the parent component
            # This handles clicking the rect or the text instead of the main item
            while item and not isinstance(item, SchematicComponent):
                item = item.parentItem()

            if isinstance(item, SchematicComponent):
                from PyQt6.QtWidgets import QInputDialog
                new_val, ok = QInputDialog.getText(self, "Edit", "New Value:", text=item.value)
                if ok and new_val:
                    item.update_value(new_val)
                    return  # Stop event here

        super().mouseDoubleClickEvent(event)


class RFFilterStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Filter Studio - EDA")
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Tools Layout (Left Panel)
        tools_layout = QVBoxLayout()
        
        btn_l = QPushButton("Add Inductor (L)")
        btn_l.clicked.connect(lambda: self.canvas.scene.addItem(SchematicComponent("L", "91 nH")))
        
        btn_c = QPushButton("Add Capacitor (C)")
        btn_c.clicked.connect(lambda: self.canvas.scene.addItem(SchematicComponent("C", "33 pF")))
        
        btn_r = QPushButton("Add Resistor (R)")
        btn_r.clicked.connect(lambda: self.canvas.scene.addItem(SchematicComponent("R", "50 Ohm")))
        
        btn_gnd = QPushButton("Add Ground")
        btn_gnd.clicked.connect(lambda: self.canvas.scene.addItem(GroundSymbol()))
        
        btn_p1 = QPushButton("Add PORT 1 (In)")
        btn_p1.clicked.connect(lambda: self.canvas.scene.addItem(VNAPort1(-150, 0)))
        
        btn_p2 = QPushButton("Add PORT 2 (Out)")
        btn_p2.clicked.connect(lambda: self.canvas.scene.addItem(VNAPort2(150, 0)))
        
        btn_sim = QPushButton("SIMULATE S21")
        btn_sim.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_sim.clicked.connect(self.run_simulation)
        
        tools_layout.addWidget(btn_l)
        tools_layout.addWidget(btn_c)
        tools_layout.addWidget(btn_r)
        tools_layout.addWidget(btn_gnd)
        tools_layout.addWidget(btn_p1)
        tools_layout.addWidget(btn_p2)
        tools_layout.addStretch()
        tools_layout.addWidget(btn_sim)
        
        # Setup Splitter for Canvas and Plot Output
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. The Schematic Canvas
        self.canvas = SchematicCanvas()
        
        # 2. The Matplotlib Canvas
        self.figure, self.ax = plt.subplots()
        self.plot_canvas = FigureCanvas(self.figure)
        self.setup_plot()
        
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.plot_canvas)
        splitter.setSizes([500, 300]) # Initial height distribution
        
        layout.addLayout(tools_layout, 1)
        layout.addWidget(splitter, 6)

    def setup_plot(self):
        """Initializes the empty Matplotlib chart."""
        self.ax.set_title("Frequency Response (S21)")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_xscale('log')
        self.ax.grid(True, which="both", ls="-", alpha=0.5)
        self.ax.set_ylim(-80, 5)
        self.line, = self.ax.plot([], [], color='blue', linewidth=2)
        self.plot_canvas.draw()

    def run_simulation(self):
        """Extracts the netlist and updates the plot."""
        netlist = extract_netlist(self.canvas.scene)
        
        # Print to console for debugging
        print("\n--- Current Netlist ---")
        print(json.dumps(netlist, indent=4))

        for item in netlist:
            print(f"Component: {item['type']}, Nodes: {item['node_a']}, {item['node_b']}")

        try:
            # Run the SPICE engine
            freqs, gain = run_ac_analysis(netlist)

            # DEBUG PRINT: Does this show numbers?
            print(f"Simulation returned {len(freqs)} data points.")
            print(f"Gain range: {min(gain):.2f} to {max(gain):.2f} dB")

            # Update the Matplotlib plot
            self.line.set_data(freqs, gain)

            self.ax.set_xlim(min(freqs), max(freqs))
            self.ax.set_ylim(min(gain) - 5, max(gain) + 5)
            # self.ax.relim()
            # self.ax.autoscale_view(scalex=False, scaley=True)
            self.plot_canvas.draw()
            
            print("Simulation successful.")
            
        except Exception as e:
            # If the user missed a ground or a port, it will print here
            print(f"Simulation Error: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RFFilterStudio()
    window.show()
    sys.exit(app.exec())