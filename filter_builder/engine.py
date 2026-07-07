from elements import SchematicComponent, ManhattanWire


class NodeBuilder:
    def __init__(self):
        self.parent = {}

    def add(self, terminal):
        if terminal is not None and terminal not in self.parent:
            self.parent[terminal] = terminal

    def find(self, terminal):
        if terminal is None: return None
        if self.parent[terminal] == terminal: return terminal
        self.parent[terminal] = self.find(self.parent[terminal])
        return self.parent[terminal]

    def union(self, term1, term2):
        if term1 is None or term2 is None: return
        root1, root2 = self.find(term1), self.find(term2)
        if root1 != root2:
            self.parent[root2] = root1


def extract_netlist(scene):
    builder = NodeBuilder()
    components = []
    wires = []

    # 1. Identify all components and their terminals
    for item in scene.items():
        if isinstance(item, SchematicComponent):
            components.append(item)
            builder.add(item.port1)
            # Only add port2 if it exists and wasn't nullified
            if hasattr(item, 'port2') and item.port2 is not None:
                builder.add(item.port2)
        elif isinstance(item, ManhattanWire):
            wires.append(item)

    # 2. Connect components via wires
    for wire in wires:
        builder.union(wire.start_port, wire.end_port)

    # 3. Assign Node IDs
    net_groups = {}
    current_node_id = 1

    # Ground is always Node 0
    for comp in components:
        if comp.comp_type == "GROUND":
            root = builder.find(comp.port1)
            if root: net_groups[root] = 0

    # Assign remaining nodes
    for comp in components:
        for port in [comp.port1, getattr(comp, 'port2', None)]:
            if port is not None:
                root = builder.find(port)
                if root and root not in net_groups:
                    net_groups[root] = current_node_id
                    current_node_id += 1

    # 4. Generate the Spice Netlist
    spice_netlist = []
    for comp in components:
        # Find the root of port1 (it must exist for all components)
        root1 = builder.find(comp.port1)
        node_a = net_groups.get(root1) if root1 is not None else None

        # Find the root of port2 (if it exists)
        root2 = builder.find(comp.port2) if hasattr(comp, 'port2') and comp.port2 else None
        node_b = net_groups.get(root2) if root2 is not None else None

        spice_netlist.append({
            'type': comp.comp_type,
            'value': comp.value,
            'node_a': node_a,
            'node_b': node_b
        })
    return spice_netlist
