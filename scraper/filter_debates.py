import os
from langdetect import detect
from shutil import copyfile
import progressbar

discussion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'discussions')
dst_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'filtered_discussions')

blacklist = set([
        'compulsory-voting-should-voting-be-mandatory-1692.txt',
    ])

p = progressbar.ProgressBar(term_width=80)
print('Filtering and copying English files: ')

for filename in p(os.listdir(discussion_dir)):
    if filename in blacklist:
        continue

    src_file = os.path.join(discussion_dir, filename)
    dst_file = os.path.join(dst_dir, filename)
    with open(src_file, 'r', encoding='latin-1') as current_file:
        full_string = ''
        for line in current_file.readlines():
            full_string += line
        if detect(full_string) == 'en':
           copyfile(src_file, dst_file)
