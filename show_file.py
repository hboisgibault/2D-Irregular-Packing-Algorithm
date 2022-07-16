import json
from tools.geofunc import GeoFunc
from tools.nfp import NFP
import pandas as pd
from tools.show import PltFunc
import copy

# Plot a single file to test
def showFile():
    df = pd.read_csv("data/test.csv")

    polygons = []
    for row in df['polygon']:
        polygons.append(json.loads(row))

    for p in polygons:
        GeoFunc.normData(p, 0.5)
        PltFunc.addPolygon(p)
    
    PltFunc.showPlt()

if __name__ == '__main__':
    showFile()