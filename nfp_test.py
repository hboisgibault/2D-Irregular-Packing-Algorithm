import json
from tools.geofunc import GeoFunc
from tools.nfp import NFP
import pandas as pd
from tools.show import PltFunc

# 计算NFP然后寻找最合适位置
def tryNFP():
    df = pd.read_csv("data/test.csv")

    poly1=json.loads(df['polygon'][0])
    poly2=json.loads(df['polygon'][1])
    GeoFunc.normData(poly1,0.5)
    GeoFunc.normData(poly2,0.5)
    print(poly1)

    nfp=NFP(poly1,poly2, show=True)
    print(nfp.nfp)

if __name__ == '__main__':
    # PlacePolygons(getData())
    tryNFP()