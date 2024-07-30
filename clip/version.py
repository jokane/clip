"""Return a version number extracted from the git history, in a form suitable
for PEP 440."""

import subprocess

def version_from_git():
    """Query git and return a version string.  This is mostly just `git
    describe`, but without the hash junk it sometimes includes."""
    version = subprocess.check_output(['git', 'describe', '--tags', '--match', 'v[0-9]*'])
    version = version.strip().decode('utf-8')
    if '-' in version:
        parts = version.split('-')
        version = f"{parts[0][1:]}.dev{parts[1]}"
    else:
        version = version[1:]
    return version

