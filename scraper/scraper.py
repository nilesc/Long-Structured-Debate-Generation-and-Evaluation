import os

discussion_dir = 'discussions'

class discussion_layer:
    def discussion_layer(text, children):
        self.text = ''
        self.pro_children = []
        self.con_children = []

# lines comes in as a list of lines in the discussion
def build_discussion_dict(lines):
    cleaned_lines = []

    for line in lines:
        cleaned_lines.append(line.replace('\n', ''))

    lines = cleaned_lines
    print(lines)

    discussion_tree = [lines[0][len('Discussion Title: '):], {}]

    for line in lines:
        # If the line doesn't start with a number, discard it
        if line == '' or not line[0].isdigit():
            continue

        number, text = line.split(' ', 1)

        current_tree = discussion_tree
        for value in number.split('.'):
            if value == '':
                continue
            value = int(value)

            if value not in current_tree[1]:
                current_tree[1][value] = [text, {}]

            current_tree = current_tree[1][value]

    return discussion_tree

def tree_to_discussion(discussion_tree):
    pass

for filename in os.listdir(discussion_dir):
    with open(os.path.join(discussion_dir, filename), 'r') as current_file:
        tree = build_discussion_dict(current_file.readlines())
        print(tree_to_discussion(tree))


