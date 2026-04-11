import re

with open("tests/test_SPN.py", "r") as f:
    content = f.read()

# Replace list definitions with np.array
content = re.sub(r'vertices = \[np\.array\(\[.*?\]\)(?:, np\.array\(\[.*?\]\))*\]',
                 lambda m: "vertices = np.array(" + m.group(0)[11:] + ")", content)

content = content.replace("edges = [[0, 1]]", "edges = np.array([[0, 1]])")
content = content.replace("edges = [[0, 1], [1, 0]]", "edges = np.array([[0, 1], [1, 0]])")
content = content.replace("edges = [[0, 1], [1, 2]]", "edges = np.array([[0, 1], [1, 2]])")
content = content.replace("edges = [[0, 1], [1, 0], [2, 2]]", "edges = np.array([[0, 1], [1, 0], [2, 2]])")

content = content.replace("arc_transitions = [0]", "arc_transitions = np.array([0])")
content = content.replace("lambda_values = [2.0]", "lambda_values = np.array([2.0])")
content = content.replace("edges = []", "edges = np.empty((0, 2), dtype=int)")

# Fix vertices lines specifically since regex might be hard
content = content.replace("vertices = [np.array([1, 0]), np.array([0, 1])]", "vertices = np.array([[1, 0], [0, 1]])")
content = content.replace("vertices = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]", "vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])")
content = content.replace("vertices = [np.array([2, 0]), np.array([0, 2])]", "vertices = np.array([[2, 0], [0, 2]])")
content = content.replace("vertices = [np.array([1])]", "vertices = np.array([[1]])")

with open("tests/test_SPN.py", "w") as f:
    f.write(content)
