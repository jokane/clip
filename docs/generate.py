"""
Auto-generate Sphinx rst pages for certain objects based on certain tags in
their docstrings.

Yes, this is the sort of thing that the Sphinx autodoc usually does.  But the
situation here is a bit off the beaten path, for example because we want to
show Clip-returing functions and Clip classes in the same way.

"""

import contextlib
import datetime
import glob
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.split(__file__)[0], '..'))
import clip

# These are tags that we look for in the docstrings.  We'll create RST files
# with a list of each of them, to include in the user guide.
tags = [
    'from-source',  # for things that create clips by reading a source file
    'ex-nihilo', # for things that create a clip out of nothing
    'modify', # for things that accept one or more clips and return another clip
    'save', # for things that save or otherwise consume a completed clip
] 

exclude = [
    'ABC',
    'abstractmethod',
    'CompressedImage',
    'Time',
    'Enum',
    'Header',
]

MAIN_DIR='_generated'

def header(title, f):
    print(f'..', file=f)
    print(f'    I was generated by {__file__} on {datetime.datetime.now()}.', file=f)
    print(f"    You probably don't want to modify me directly.", file=f)
    print(file=f)
    print(f".. module:: clip", file=f)
    print(' :noindex:', file=f)

    if title:
        print(file=f)
        print('='*len(title), file=f)
        print(title, file=f)
        print('='*len(title), file=f)
    print(file=f)
    print(file=f)

def main():
    os.chdir(os.path.split(__file__)[0])
    os.makedirs(MAIN_DIR, exist_ok=True)
    
    print('Generating documentation...')
    with contextlib.ExitStack() as exst:
        f_ref = exst.enter_context(open(os.path.join(MAIN_DIR, 'reference.rst'), 'w'))
        header('API reference', f_ref)
        print(file=f_ref)

        f_tag = { tag: exst.enter_context(open(os.path.join(MAIN_DIR, f'{tag}.rst'), 'w')) for tag in tags }

        for name, thing in sorted(clip.__dict__.items(), key=lambda x: x[0].lower()):
            if name in exclude: continue

            doc = thing.__doc__ if thing.__doc__ else ''

            thing_tags = [tag for tag in tags if f'|{tag}|' in doc]

            print('  ', name, ' '.join([f'#{tag}' for tag in thing_tags]))

            basename = f'{name}.rst'
            filename = os.path.join(MAIN_DIR, basename)

            if inspect.isclass(thing):
                print(f'.. autoclass:: {name}', file=f_ref)
                if not ('|from-source|' in doc or '|modify|' in doc or '|ex-nihilo|' in doc):
                    print('    :members:', file=f_ref)
                print(file=f_ref)
            elif callable(thing):
                print(f'.. autofunction:: {name}', file=f_ref)
                print(file=f_ref)

            for tag in thing_tags:
                print(f':func:`clip.{name}`', file=f_tag[tag])

            

if __name__ == '__main__':
    main()
