import sys
import math
import pcbnew
from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton)

# exec(open('D:/Projects/python_common/kicad/spiral_footprint_generation.py').read())

# Unit conversion: KiCad 9 Internal Units (IU) are nanometers
def mm_to_iu(mm):
    return int(mm * 1000000)
    
class SpiralDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KiCad Spiral Generator")
        self.layout = QVBoxLayout()

        # Input fields: Label, Default Value
        self.inputs = {}
        fields = [
            ("Arms", "2"),
            ("Turns", "5"),
            ("Max Diameter (mm)", "20"),
            ("Offset (mm)", "2"),
        ]

        for label, default in fields:
            row = QHBoxLayout()
            lbl = QLabel(label)
            edit = QLineEdit(default)
            row.addWidget(lbl)
            row.addWidget(edit)
            self.inputs[label] = edit
            self.layout.addLayout(row)

        self.btn = QPushButton("Generate Footprint")
        self.btn.clicked.connect(self.accept)
        self.layout.addWidget(self.btn)
        self.setLayout(self.layout)

    def get_values(self):
        return {k: float(v.text()) for k, v in self.inputs.items()}

def create_spiral():
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = SpiralDialog()
    
    if dialog.exec():
        params = dialog.get_values()
        
        print(params)
        
        # Parameters
        arms = int(params["Arms"])
        turns = params["Turns"]
        max_r = params["Max Diameter (mm)"] / 2.0
        offset = params["Offset (mm)"]
        
        # Calculate width and spacing (Width = Spacing)
        # Total radial growth / (Total arm passes + total gaps)
        total_layers = turns * arms
        pitch = (max_r - offset) / turns
        trace_width = pitch / (2 * arms) # Simplified for Width = Spacing
        
        # Initialize Footprint
        footprint = pcbnew.FOOTPRINT(None)
        footprint.SetValue("Spiral")
        footprint.SetReference("A1")
        
        print("Trace Width: {}\n".format(trace_width))
        
        for arm_idx in range(arms):
            # Create SMD Pad at start
            pad = pcbnew.PAD(footprint)
            pad.SetNumber(str(arm_idx + 1))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
            pad.SetSize(pcbnew.VECTOR2I_MM(0.9*trace_width, 0.9*trace_width))
            
            # Position pad at the offset start point
            angle_start = (2 * math.pi / arms) * arm_idx
            start_x = offset * math.cos(angle_start)
            start_y = offset * math.sin(angle_start)
            pad.SetPosition(pcbnew.VECTOR2I_MM(start_x, start_y))
            
            layers = pcbnew.LSET()
            layers.AddLayer(pcbnew.F_Cu)
            layers.AddLayer(pcbnew.F_Mask)
            layers.AddLayer(pcbnew.F_Paste)
            pad.SetLayerSet(layers)
            #pad.SetLayerSet(pcbnew.LSET(pcbnew.F_Cu) | pcbnew.LSET(pcbnew.F_Mask) | pcbnew.LSET(pcbnew.F_Paste))
            footprint.Add(pad)

            # Generate Spiral Segments
            points = 200 * turns # Resolution
            prev_pos = pcbnew.VECTOR2I_MM(start_x, start_y)
            
            for i in range(int(points+1)):
                theta = (i / points) * (2 * math.pi * turns)
                r = offset + (pitch * (theta / (2 * math.pi)))
                
                # Apply arm phase shift
                current_theta = theta + angle_start
                x = r * math.cos(current_theta)
                y = r * math.sin(current_theta)
                
                curr_pos = pcbnew.VECTOR2I_MM(x, y)
                
                # Create track segment
                segment = pcbnew.PCB_SHAPE(footprint)
                segment.SetShape(pcbnew.SHAPE_T_SEGMENT)
                segment.SetFilled(False)
                #segment.SetWidth(pcbnew.frommm(trace_width))
                segment.SetWidth(int(trace_width * 1000000))
                segment.SetStart(prev_pos)
                segment.SetEnd(curr_pos)
                segment.SetLayer(pcbnew.F_Cu)
                footprint.Add(segment)
                
                prev_pos = curr_pos

        # Create Keepout Areas
        # Top and Bottom layers keepout
        keepout_r = max_r + trace_width
        for layer in [pcbnew.F_Cu, pcbnew.B_Cu]:
            keepout = pcbnew.ZONE(footprint)
            keepout.SetLayer(layer)
            keepout.SetIsRuleArea(True)
            keepout.SetDoNotAllowCopperPour(True)
            keepout.SetDoNotAllowVias(False)  # Allow Vias
            keepout.SetDoNotAllowTracks(False) # Allow manual tracks if needed
            
            # Create a circular keepout boundary
            circle_pts = 64
            poly = pcbnew.SHAPE_LINE_CHAIN()
            for i in range(circle_pts):
                angle = (2 * math.pi * i) / circle_pts
                poly.Append(int(keepout_r * math.cos(angle)*1000000), int(keepout_r * math.sin(angle)*1000000))
            
            keepout.AddPolygon(poly)
            footprint.Add(keepout)

        # add a silk screen
        silk_circle = pcbnew.PCB_SHAPE(footprint)
        silk_circle.SetShape(pcbnew.SHAPE_T_CIRCLE)
        silk_circle.SetCenter(pcbnew.VECTOR2I(0, 0))
        
        # Circle end point defines the radius
        silk_radius_mm = max_r + trace_width
        silk_circle.SetEnd(pcbnew.VECTOR2I(mm_to_iu(silk_radius_mm), 0))
        silk_circle.SetWidth(mm_to_iu(0.15)) # Standard silk width
        silk_circle.SetLayer(pcbnew.F_SilkS)
        footprint.Add(silk_circle)
        
        # Add to board
        board = pcbnew.GetBoard()
        board.Add(footprint)
        pcbnew.Refresh()
        print("Spiral Footprint Generated Successfully.")

if __name__ == "__main__":
    create_spiral()