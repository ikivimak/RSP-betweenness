"""
Read Open Street Maps osm format into Networkx

Based on GitHub gist:287370 by aflaxman, which is based on osm.py from brianw's osmgeocode
http://github.com/brianw/osmgeocode, which is based on osm.py from Graphserver:
http://github.com/bmander/graphserver/tree/master which is copyright (c)
2007, Brandon Martin-Anderson under the BSD License
"""


import xml.sax
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance
import scipy.io

def download_osm(left,bottom,right,top):
    """ Return a filehandle to the downloaded data."""
    from urllib import urlopen
    fp = urlopen( "http://overpass-api.de/api/map?bbox=%f,%f,%f,%f"%(left,bottom,right,top) )
    return fp

def osm2nx(filename_or_stream, allowed_roads=["primary","secondary","tertiary","unclassified","residential","service"], disallow_2_degrees=True):
    """Read graph in OSM format from file specified by name or by stream object.

    Parameters
    ----------
    filename_or_stream : filename or stream object

    Returns
    -------
    G : Graph

    Examples
    --------
    >>> G = osm2nx(download_osm(-74.0188,40.7006,-73.9605,40.7642))
    >>> plot_osm_network(G)

    """
    osm = OSM(filename_or_stream,allowed_roads)

    G = nx.DiGraph()

    # read ways as paths:
    for w in osm.ways.itervalues():
        G.add_path(w.nds)

    # buggy data can have selfloops:
    G.remove_edges_from(G.selfloop_edges())

    # read node metadata:
    for n_id in G.nodes_iter():
        G.node[n_id] = osm.nodes[n_id]

    # make undirected:
    G = G.to_undirected()

    # compute edge lengths:
    for i,j in G.edges_iter():
        G[i][j]['length'] = compute_edge_length(G,i,j)

    # remove 2-degree nodes:
    if disallow_2_degrees:
        G = remove_2_degrees(G)
    
    return G

def compute_edge_length(G,i,j):
    x0 = G.node[i].lon
    x1 = G.node[j].lon
    y0 = G.node[i].lat
    y1 = G.node[j].lat
    return distance((x0,y0), (x1,y1)).meters

def remove_2_degrees(G):
    degree_2_nodes = [n for n,d in G.degree().iteritems() if d == 2]
    if len(degree_2_nodes) > 0:
        for n in degree_2_nodes:
            nodes_to_connect = G[n]
            # The degrees might have changed during node removals,
            # so check if n still has degree 2:
            if len(nodes_to_connect) == 2:
                new_edge = nodes_to_connect.keys()
                # Don't remove n if it defines a detour between nodes_to_connect:
                if new_edge[1] not in G[new_edge[0]].keys():
                    new_length = nodes_to_connect.values()[0]['length']+nodes_to_connect.values()[1]['length']
                    G.add_edge(new_edge[0],new_edge[1],length=new_length)
                    G.remove_node(n)
        return G
    else:
        return G

def plot_osm_network(G,node_size=10,with_labels=False,ax=None):
    plt.figure()
    pos = {}
    for n in G.nodes():
        pos[n] = (G.node[n].lon, G.node[n].lat)
    nx.draw_networkx(G,pos,node_size=node_size,with_labels=with_labels,ax=ax)
    plt.show()

    
def export_to_matlab(G,filename=None):
    pos = {}
    for n in G.nodes():
        pos[n] = (G.node[n].lon, G.node[n].lat)
    
    lengths = [G.edge[n1][n2]['length'] for n1, n2 in G.edges()]

    mat_dict = {}
    mat_dict['edges'] = np.array(G.edges(), dtype=int)
    mat_dict['lengths'] = np.array(lengths)
    mat_dict['coords'] = np.array(pos.values())
    mat_dict['ids'] = np.array([int(k) for k in pos.keys()])
    scipy.io.savemat(filename, mat_dict)


    
class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}
        
class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}
        self.length = 0.
        
    def split(self, node_counts):
        # slice the nds-list using this nifty recursive function
        def slice_list(nds, node_counts):
            for i,nd in enumerate(nds[1:]):
                # Split if node appears more than once in ways:
                if node_counts[nd]>1:
                    left = nds[:i]
                    right = nds[i+1:]
                    
                    rightsliced = slice_list(right, node_counts)
                    
                    return [left]+rightsliced
            # If way doesn't need splitting, return the original:
            return [nds]
            
        slices = slice_list(self.nds, node_counts)
        
        # create a way object for each node-array slice
        new_ways = []
        i=0
        for slice in slices:
            littleway = copy.copy( self )
            littleway.id += "-%d"%i
            littleway.nds = slice
            new_ways.append( littleway )
            i += 1
            
        return new_ways
        

class OSM:
    def __init__(self, filename_or_stream, allowed_roads):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}
        
        superself = self
        
        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self,loc):
                pass
            
            @classmethod
            def startDocument(self):
                pass
                
            @classmethod
            def endDocument(self):
                pass
                
            @classmethod
            def startElement(self, name, attrs):
                if name=='node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name=='way':
                    self.currElem = Way(attrs['id'], superself)
                elif name=='tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name=='nd':
                    self.currElem.nds.append( attrs['ref'] )
                
            @classmethod
            def endElement(self,name):
                if name=='node':
                    nodes[self.currElem.id] = self.currElem
                elif name=='way':
                    ways[self.currElem.id] = self.currElem
                
            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)
        
        # include only 'highways':
        ways = {id:w for id,w in ways.iteritems() if 'highway' in w.tags}
        # include only 'allowed_roads':
        self.ways = {id:w for id,w in ways.iteritems() if w.tags['highway'] in allowed_roads}

        # check for ways with only one node and for nodes that appear in ways:
        nodes_in_ways = []
        todel = []
        for way_id,way in self.ways.iteritems():
            if len(way.nds) < 2:       # list ways with only one node
                todel.append(way_id)
            else:
                for node in way.nds:
                    nodes_in_ways.append(node)

        # remove one node ways:
        for way_id in todel:
            del self.ways[way_id]
        
        # only consider nodes that are part of ways:
        self.nodes = {id:nodes[id] for id in nodes_in_ways}
            
