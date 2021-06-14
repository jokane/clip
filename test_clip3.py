import pytest

from clip3 import *

def test_clip_is_abstract():
    with pytest.raises(TypeError):
        Clip()  
