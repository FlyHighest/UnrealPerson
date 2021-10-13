from unrealcv import client
import pickle
cl=client
cl.connect()

d=list()

def gc():
    global  cl
    global d
    l=cl.request('vget /camera/0/location')
    r=cl.request('vget /camera/0/rotation')
    d.append((l,r))
    print(d)

def sa():
    global  d
    pickle.dump(d, open('../caminfo_s006_high.pkl', 'wb'))

from IPython import embed
embed()