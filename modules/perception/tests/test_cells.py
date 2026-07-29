from oaq_state_detection.counting.cells import Cell, assign_to_nearest_cell, count_per_cell
from oaq_state_detection.detection.detector import Detection


def test_assign_and_count():
    cells = [Cell(id="a", center=(0, 0)), Cell(id="b", center=(100, 100))]
    detections = [
        Detection(bbox=(0, 0, 2, 2), center=(1, 1), class_id=0, confidence=0.9),
        Detection(bbox=(98, 98, 102, 102), center=(100, 100), class_id=1, confidence=0.8),
        Detection(bbox=(97, 97, 101, 101), center=(99, 99), class_id=0, confidence=0.7),
    ]
    class_names = {0: "ordinary_piece", 1: "mandarin_piece"}

    assignments = assign_to_nearest_cell(detections, cells)
    counts = count_per_cell(assignments, class_names)

    assert counts["a"] == {"ordinary_piece": 1, "mandarin_piece": 0}
    assert counts["b"] == {"ordinary_piece": 1, "mandarin_piece": 1}
