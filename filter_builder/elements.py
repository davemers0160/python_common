from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsTextItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QBrush, QColor, QPainterPath

class Terminal(QGraphicsEllipseItem):
    """A connection pin on a component."""
    def __init__(self, x, y, parent=None):
        radius = 5
        super().__init__(x - radius, y - radius, radius * 2, radius * 2, parent)
        self.setBrush(QBrush(QColor("#0000FF")))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.wires = []
        
    def add_wire(self, wire):
        if wire not in self.wires: self.wires.append(wire)
            
    def remove_wire(self, wire):
        if wire in self.wires: self.wires.remove(wire)
        
    def update_wires(self):
        for wire in self.wires: wire.update_position()

class SchematicComponent(QGraphicsItem):
    """Base class for all components with ports."""
    def __init__(self, comp_type, value, x=0, y=0):
        super().__init__()
        self.comp_type = comp_type
        self.value = value
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPos(x, y)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        
        # Initialize terminals
        self.port1 = Terminal(-25, 0, self)
        self.port2 = Terminal(25, 0, self)
        
        # Visuals
        self.rect = QGraphicsRectItem(-25, -15, 50, 30, self)
        self.rect.setBrush(QBrush(QColor("#E0E0E0")))
        self.label = QGraphicsTextItem(f"{comp_type}\n{value}", self)
        self.label.setPos(-20, -15)
        self.setTransformOriginPoint(0, 0)

    def boundingRect(self):
        return self.rect.boundingRect().adjusted(-10, -10, 10, 10)

    def paint(self, painter, option, widget):
        color = "red" if self.isSelected() else "black"
        self.rect.setPen(QPen(QColor(color), 2 if self.isSelected() else 1))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Defensive check: only update if the port exists and is not None
            if hasattr(self, 'port1') and self.port1 is not None:
                self.port1.update_wires()

                # Check if port2 exists AND is not None before updating
            if hasattr(self, 'port2') and self.port2 is not None:
                self.port2.update_wires()

        return super().itemChange(change, value)

    def update_value(self, new_value):
        self.value = new_value
        # Update the visual label
        self.label.setPlainText(f"{self.comp_type}\n{new_value}")

class GroundSymbol(SchematicComponent):
    def __init__(self, x=0, y=0):
        super().__init__("GROUND", "0V", x, y)
        self.port2.hide() # Ground only needs one port

    def paint(self, painter, option, widget):
        pen = QPen(QColor("red") if self.isSelected() else Qt.GlobalColor.black, 2)
        painter.setPen(pen)
        painter.drawLine(0, -10, 0, 0)
        painter.drawLine(-15, 0, 15, 0)
        painter.drawLine(-10, 5, 10, 5)

class VNAPort1(SchematicComponent):
    def __init__(self, x=-100, y=0):
        super().__init__("PORT1", "50 Ohm", x, y)
        self.rect.setBrush(QBrush(QColor("#FFCCCC")))
        self.label.setPlainText("PORT 1\n(IN)")
        self.port1.hide()  # Hide the visual
        # Ensure it has no effect on the graph connectivity
        self.port1 = None

class VNAPort2(SchematicComponent):
    def __init__(self, x=100, y=0):
        super().__init__("PORT2", "50 Ohm", x, y)
        self.rect.setBrush(QBrush(QColor("#CCFFCC")))
        self.label.setPlainText("PORT 2\n(OUT)")
        self.port2.hide() # Hide the visual
        # Ensure it has no effect on the graph connectivity
        self.port2 = None

class ManhattanWire(QGraphicsPathItem):
    """An orthogonal live wire."""
    def __init__(self, start_port):
        super().__init__()
        self.start_port = start_port
        self.end_port = None
        self.dynamic_end_pos = None
        self.setPen(QPen(QColor("#005500"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.setZValue(-1)
        self.start_port.add_wire(self)

    def set_end_port(self, end_port):
        self.end_port = end_port
        self.end_port.add_wire(self)
        self.update_position()

    def set_dynamic_end(self, pos):
        self.dynamic_end_pos = pos
        self.update_position()

    def update_position(self):
        if not self.start_port: return
        p1 = self.start_port.scenePos() + self.start_port.boundingRect().center()
        p2 = self.end_port.scenePos() + self.end_port.boundingRect().center() if self.end_port else self.dynamic_end_pos
        if not p2: return
        path = QPainterPath()
        path.moveTo(p1)
        mid_x = (p1.x() + p2.x()) / 2.0
        path.lineTo(mid_x, p1.y())
        path.lineTo(mid_x, p2.y())
        path.lineTo(p2.x(), p2.y())
        self.setPath(path)