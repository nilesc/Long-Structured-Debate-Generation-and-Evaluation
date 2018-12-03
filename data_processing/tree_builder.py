import os
import re
import progressbar
import ner

discussion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/filtered_discussions')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/input_files')

class DiscussionTree:

    def __init__(self, text, children):

        full_value, self.text = text.split(' ', 1)
        all_values = [value.strip('.') for value in full_value.split('.')]
        cleaned = [value for value in all_values if value != '']
        self.position = cleaned[0:-1]
        self.value = cleaned[-1]

        self.is_pro = not self.text.startswith('Con: ')
        if self.text.startswith('Con: ') or self.text.startswith('Pro: '):
            self.text = self.text[len('Pro: '):]

        self.children = {}

        for child in children:
            self.children[child.value] = child

    def print_pre_order(self, tabs=''):
        '''
        Prints pre-order traversal of debate tree
        '''
        print(tabs + self.text)
        print("--")
        for child in self.children.values():
            child.print_pre_order(tabs+'    ')

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

    # Calls build_complex_args_inner and cleans results
    def build_complex_args(self, pro_responses=None):
        unparsed_args = self.build_complex_args_inner()
        parsed_args = []

        for arg in unparsed_args:
            if len(arg) == 1:
                continue

            sentences, is_pro = zip(*arg)

            # Look at all subsets that begin at the start
            for slice_index in range(2, len(sentences)+1):
                reduced_sentences = sentences[:slice_index]
                reduced_is_pro = is_pro[:slice_index]

                # If there are no con arguments other than first, then add all slices
                # Don't include this if pro_top_level is False
                if all(reduced_is_pro[1:]) and not (pro_responses != None and not pro_top_level):
                    parsed_args.extend(slice_augmentation([list(reduced_sentences)]))

                # If there is a con argument, then split at that point
                # Don't include this if pro_top_level is True
                elif not all(reduced_is_pro[1:]) and not pro_responses:
                    split_point = reduced_is_pro[1:].index(False) + 1
                    first_part = sentences[:split_point]
                    second_part = sentences[split_point:]

                    first_part_condensed = ''
                    for sentence in first_part:
                        first_part_condensed += sentence

                    for crop_point in range(len(second_part)):
                        parsed_args.append([first_part_condensed] + list(second_part)[:crop_point+1])
                    parsed_args.append([first_part_condensed] + list(second_part))

        return remove_duplicates(parsed_args)

    # For each node in our tree, call traverse_complex and append its own text
    # and is_pro value to the resulting arguments
    def build_complex_args_inner(self):
        all_args = []

        children = self.get_children()

        # for every node in tree
        for child in children:
            child_args = child.build_complex_args_inner()
            all_args.extend(child_args)

            for complex_arg in child.traverse_complex(False):
                all_args.append([(self.text, self.is_pro)] + complex_arg)

        return all_args

    def traverse_complex(self, seen_con):
        # If we have already seen a con argument and we see another
        seen_con = seen_con or not self.is_pro

        children = []

        if not seen_con:
            children = self.get_children()
        else:
            children = self.get_pro_children()

        if not children:
            return [[(self.text, self.is_pro)]]

        partial_args = []
        for child in children:
            partial_args.extend(child.traverse_complex(seen_con))

        return_list = [[(self.text, self.is_pro)] + partial_arg for partial_arg in partial_args]

        return return_list


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

                parent.children[self.value].text = current_root.text

        for child in self.children.values():
            child.fix_references(self, root)

def clean_named_entities(arg):
    text = ner.replace_entities(arg, None)
    return text

# Lines comes in as a list of lines in the discussion
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


def front_augmentation(args):
    """
    This is a method of augmentation that says that, for a given prompt and response,
    the same prompt and the response without the last sentence is also a vaild prompt
    response pair. If we start with
        Prompt: A, Response: B C D
    where all letters are sentences, this method would add
        Prompt: A, Response: B C
        Prompt: A, Response: B
    """
    augmented = [arg for arg in args]
    for arg in args:
        for i in range(len(arg)-2):
            augmented.append(arg[:(i+2)])

    return augmented


def back_augmentation(args):
    """
    This is a method of augmentation that says that, for a given prompt and response,
    the first sentence of the response can be a valid prompt, and the rest of the
    response will be a response to this new prompt. If we start with
        Prompt A, Response: B C D
    where all letters are sentences, this method would add
        Prompt: B, Response: C D
        Prompt: C, Response: D
    """
    augmented = [arg for arg in args]
    for arg in args:
        for i in range(len(arg)-2):
            augmented.append(arg[(i+1):])

    return augmented

def slice_augmentation(args):
    """
    This is a method of augmentation that says that, for a given prompt and response,
    the prompt and any number of sentences from the start of the response can be a
    valid prompt, and the remainder of the response will be a valid response to this
    new prompt. If we start with
        Prompt A, Response: B C D
    where all letters are sentences, this method would add
        Prompt: A B, Response: C D
        Prompt: A B C, Response: D
    """
    augmented = []
    for arg in args:
        for i in range(len(arg) - 1):
            first_part = arg[:i+1]
            second_part = arg[i+1:]
            combined = ""
            for sentence in first_part:
                combined += (' ' + sentence)
            combined = combined[1:]
            augmented.append([combined] + second_part)
    return augmented


def total_augmentation(args):
    """
    Runs a combination of back_augmentation, front_augmentation, and slice_augmentation.
    This should return all viable arguments that can be made from any subset of the original
    argument.
    """
    args = back_augmentation(args)
    args = front_augmentation(args)
    args = slice_augmentation(args)

    return remove_duplicates(args)

def remove_duplicates(x):
    list_of_tuples = list(set(tuple(value) for value in x))
    return [list(value) for value in list_of_tuples]

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
        if not filename.endswith('txt'):
            return
        tree = build_discussion_dict(current_file.readlines())

        discussion = tree_to_discussion(tree)
        discussion.fix_references()
        args = discussion.build_complex_args()
        # args = discussion.get_arguments(pro=True, augmentor=back_augmentation)

        for arg in args:
            prompt = arg[0]
            response = arg[1]
            # prompt = clean_named_entities(arg[0])
            # response = clean_named_entities(arg[1])
            source_file.write(prompt + '\n')
            target_file.write(response + '\n')


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
