import os
import sys
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.datakickstart_dabs import main
            
def test_main():
    taxis = main.get_taxis()
    assert taxis.count() > 5
