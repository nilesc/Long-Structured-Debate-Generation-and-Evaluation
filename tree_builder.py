import os
import re
import progressbar
from utils import ner

discussion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filtered_discussions')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_files')


class DiscussionTree:

    def __init__(self, text, children):

        full_value, self.text = text.split(' ', 1)
        all_values = [value.strip('.') for value in full_value.split('.')]
        cleaned = [value for value in all_values if value != '']
        self.value = cleaned[-1]

        self.is_pro = not self.text.startswith('Con: ')
        if self.text.startswith('Con: ') or self.text.startswith('Pro: '):
            self.text = self.text[len('Pro: '):]

        self.children = {}

        for child in children:
            self.children[child.value] = child

    def get_pro_children(self):
        return [child for child in self.children.values() if child.is_pro]

    def get_con_children(self):
        return [child for child in self.children.values() if not child.is_pro]

    def get_children(self):
        return [child for child in self.children.values()]

    def get_arguments(self, pro=None, augmentor=None):
        base_args = self.get_arguments_inner(pro)

        if augmentor:
            base_args = augmentor(base_args)

        return base_args

    def get_arguments_inner(self, pro):
        children = self.get_children()

        if pro:
            children = self.get_pro_children()
        if not pro and pro is not None:
            children = self.get_con_children()

        if not children:
            return [[self.text]]

        child_arguments = []

        for child in children:
            child_arguments += child.get_arguments_inner(pro)

        all_args = [[self.text] + child for child in child_arguments]

        return all_args

    def fix_references(self, parent=None, root=None):
        if root is None:
            root = self

        if self.text.startswith('->'):
            if not 'discussion' in self.text:
                number = self.text.split(' ')[-1]

                current_root = root
                cleaned_values = [val for val in number.split('.')[1:] if val != '']
                for value in cleaned_values:
                    if value == '':
                        continue
                    current_root = current_root.children[value]

                parent.children[self.value] = current_root

        for child in self.children.values():
            child.fix_references(self, root)

    def clean_named_entities(self):
        self.text = ner.replace_entities(self.text)

        for child in self.children.values():
            child.clean_named_entities()

# lines comes in as a list of lines in the discussion
def build_discussion_dict(lines):
    cleaned_lines = []

    for line in lines:
        if not line.startswith('1.'):
            continue
        prev_line = ''
        cleaned_line = line

        # Removes URLs from arguments
        while cleaned_line != prev_line:
            prev_line = cleaned_line
            cleaned_line = re.sub(r'(.*)\[(.*)\]\((.*)\)(.*)', r'\1\2\4', prev_line.rstrip())
        cleaned_lines.append(cleaned_line.replace('\n', ''))

    lines = cleaned_lines

    discussion_tree = None

    for line in lines:
        # If the line doesn't start with a number, discard it
        number, _ = line.split(' ', 1)

        if discussion_tree is None:
            discussion_tree = [line, {}]
            continue

        cleaned_values = [val for val in number.split('.')[1:] if val != '']

        current_tree = discussion_tree
        for value in cleaned_values:
            if value == '':
                continue
            value = int(value)
            if value not in current_tree[1]:
                current_tree[1][value] = [line, {}]

            current_tree = current_tree[1][value]

    return discussion_tree

# This is a method of augmentation that says that, for a given prompt and response,
# the same prompt and the response without the last sentence is also a vaild prompt
# response pair. If we start with
# Prompt: A, Response: B C D
# where all letters are sentences, this method would add
# Prompt: A, Response: B C
# Prompt: A, Response: B
def front_augmentation(args):
    augmented = [arg for arg in args]
    for arg in args:
        for i in range(len(arg)-2):
            augmented.append(arg[:(i+2)])

    return augmented

# This is a method of augmentation that says that, for a given prompt and response,
# the first sentence of the response can be a valid prompt, and the rest of the
# response will be a response to this new prompt. If we start with
# Prompt A, Response: B C D
# where all letters are sentences, this method would add
# Prompt: B, Response: C D
# Prompt: C, Response: D
def back_augmentation(args):
    augmented = [arg for arg in args]
    for arg in args:
        for i in range(len(arg)-2):
            augmented.append(arg[(i+1):])

    return augmented

# This is a method of augmentation that says that, for a given prompt and response,
# the prompt and any number of sentences from the start of the response can be a
# valid prompt, and the remainder of the response will be a valid response to this
# new prompt. If we start with
# Prompt A, Response: B C D
# where all letters are sentences, this method would add
# Prompt: A B, Response: C D
# Prompt: A B C, Response: D
def slice_augmentation(args):
    augmented = []
    for arg in args:
        for i in range(len(arg) - 1):
            first_part = arg[:i+1]
            second_part = arg[i+1:]
            combined = ""
            for sentence in first_part:
                combined += sentence
            augmented.append([combined] + second_part)
    return augmented

def tree_to_discussion(discussion_tree):
    text = discussion_tree[0]
    children = discussion_tree[1].values()
    child_trees = []
    for child in children:
        child_trees.append(tree_to_discussion(child))
    discussion = DiscussionTree(text, child_trees)
    return discussion


def write_discussions_to_files(discussion_dir, filename, source_file, target_file):
    with open(os.path.join(discussion_dir, filename), 'r') as current_file:
        tree = build_discussion_dict(current_file.readlines())
        discussion = tree_to_discussion(tree)
        discussion.fix_references()
        discussion.clean_named_entities()
        args = discussion.get_arguments(pro=True, augmentor=back_augmentation)

        for arg in args:
            source_file.write(arg[0] + '\n')

            as_string = ''
            for sentence in arg[1:]:
                as_string += (' ' + sentence)
            as_string = as_string[1:]
            target_file.write(as_string + '\n')


def write_source_target(discussion_dir, filenames, name, start, end):
    source_file = open(os.path.join(output_dir, f'{name}.kialo_source'), 'w+')
    target_file = open(os.path.join(output_dir, f'{name}.kialo_target'), 'w+')

    p = progressbar.ProgressBar(term_width=80)

    print(f'Extracting {name} arguments: ')
    for filename in p(filenames[start:end]):
        write_discussions_to_files(discussion_dir, filename, source_file, target_file)

    source_file.close()
    target_file.close()


if __name__ == '__main__':
    filenames = os.listdir(discussion_dir)
    test_start = 0
    test_end = int(len(filenames) * .1)
    valid_start = test_end
    valid_end = valid_start + int(len(filenames) * .05)
    train_start = valid_end
    train_end = len(filenames)

    write_source_target(discussion_dir, filenames, 'test', test_start, test_end)
    write_source_target(discussion_dir, filenames, 'valid', valid_start, valid_end)
    write_source_target(discussion_dir, filenames, 'train', train_start, train_end)
