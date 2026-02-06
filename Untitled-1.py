# %%
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge, split, unary_union
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString
import networkx as nx
import numpy as np
import folium
from collections import defaultdict

# %%
road_labels = ['ST_FULLNAM','SHAPE_Leng','CE_SPEED_L', 'geometry']

roads = gpd.read_file("Loudoun_Street_Centerline.shp")
roads = roads[roads['CE_SPEED_L'] > 25]
roads = roads[road_labels]

# %%
# Dissolve by street name
roads = roads.dissolve(by="ST_FULLNAM").reset_index()
roads = roads.to_crs(epsg=4326)

print(f"Number of unique streets: {len(roads)}")

# %%
# STEP 1: Find all intersection points (where roads cross each other)
sindex = roads.sindex
all_intersections = []
precision = 6

for idx, entry in roads.iterrows():
    geom = entry['geometry']
    st_name = entry['ST_FULLNAM']
    
    # Merge if MultiLineString
    if isinstance(geom, MultiLineString):
        geom = linemerge(geom)
    
    # Find potentially intersecting roads
    matches_idx = list(sindex.intersection(geom.bounds))
    matches = roads.iloc[matches_idx]
    
    # Get intersections with OTHER streets
    other_roads = matches[matches['ST_FULLNAM'] != st_name].geometry.union_all()
    inters = geom.intersection(other_roads)
    
    # Collect intersection points
    if isinstance(inters, Point):
        all_intersections.append(inters)
    elif isinstance(inters, MultiPoint):
        all_intersections.extend(inters.geoms)

# Create set of intersection coordinates (rounded)
intersection_coords = set()
for pt in all_intersections:
    coord = (round(pt.x, precision), round(pt.y, precision))
    intersection_coords.add(coord)

print(f"Number of intersections found: {len(intersection_coords)}")

# %%
# STEP 2: Extract endpoints of each road segment
# Only endpoints AND intersections become nodes
def get_all_coords(geom):
    """Get all coordinates from a geometry"""
    if isinstance(geom, LineString):
        return list(geom.coords)
    elif isinstance(geom, MultiLineString):
        coords = []
        for line in geom.geoms:
            coords.extend(line.coords)
        return coords
    return []

# Count how many times each coordinate appears as an endpoint
endpoint_count = defaultdict(int)

for idx, row in roads.iterrows():
    geom = row['geometry']
    
    if isinstance(geom, MultiLineString):
        geom = linemerge(geom)
    
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        # Start and end are endpoints
        start = (round(coords[0][0], precision), round(coords[0][1], precision))
        end = (round(coords[-1][0], precision), round(coords[-1][1], precision))
        endpoint_count[start] += 1
        endpoint_count[end] += 1
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            coords = list(line.coords)
            start = (round(coords[0][0], precision), round(coords[0][1], precision))
            end = (round(coords[-1][0], precision), round(coords[-1][1], precision))
            endpoint_count[start] += 1
            endpoint_count[end] += 1

# Nodes are: intersections OR endpoints that appear multiple times OR dead ends
node_coords = set()
node_coords.update(intersection_coords)  # All intersections are nodes

# Add endpoints that connect to multiple roads
for coord, count in endpoint_count.items():
    if count > 1 or coord in intersection_coords:
        node_coords.add(coord)
    else:
        # Dead end - also a node
        node_coords.add(coord)

print(f"Total nodes (intersections + endpoints): {len(node_coords)}")

# %%
# STEP 3: Build graph with edges connecting nodes
G = nx.Graph()

# Add all nodes
for coord in node_coords:
    G.add_node(coord, pos=coord, x=coord[0], y=coord[1])

# Process each road to create edges between nodes
edge_count = 0
for idx, row in roads.iterrows():
    geom = row['geometry']
    
    if isinstance(geom, MultiLineString):
        geom = linemerge(geom)
    
    # Get all coordinates along the road
    if isinstance(geom, LineString):
        all_coords = list(geom.coords)
    elif isinstance(geom, MultiLineString):
        all_coords = []
        for line in geom.geoms:
            all_coords.extend(list(line.coords))
    else:
        continue
    
    # Round all coordinates
    rounded_coords = [(round(c[0], precision), round(c[1], precision)) for c in all_coords]
    
    # Find which coordinates are nodes
    node_indices = [i for i, coord in enumerate(rounded_coords) if coord in node_coords]
    
    # Create edges between consecutive nodes
    for i in range(len(node_indices) - 1):
        start_idx = node_indices[i]
        end_idx = node_indices[i + 1]
        
        start_node = rounded_coords[start_idx]
        end_node = rounded_coords[end_idx]
        
        # Calculate length of this segment
        segment_coords = all_coords[start_idx:end_idx+1]
        length = sum(
            np.sqrt((segment_coords[j+1][0] - segment_coords[j][0])**2 + 
                   (segment_coords[j+1][1] - segment_coords[j][1])**2)
            for j in range(len(segment_coords) - 1)
        )
        
        # Create edge
        if not G.has_edge(start_node, end_node):
            G.add_edge(start_node, end_node,
                      street_name=row['ST_FULLNAM'],
                      speed_limit=row['CE_SPEED_L'],
                      length=length,
                      geometry=LineString(segment_coords))
            edge_count += 1

print(f"\nGraph Statistics:")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Is connected: {nx.is_connected(G)}")

if not nx.is_connected(G):
    components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(components)}")
    component_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"Component sizes (top 10): {component_sizes[:10]}")

# %%
# Calculate degree distribution
degrees = dict(G.degree())
degree_values = list(degrees.values())

plt.figure(figsize=(10, 6))
plt.hist(degree_values, bins=range(0, max(degree_values)+2), edgecolor='black')
plt.xlabel('Node Degree')
plt.ylabel('Count')
plt.title('Node Degree Distribution')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nDegree Statistics:")
print(f"Average degree: {np.mean(degree_values):.2f}")
print(f"Max degree: {max(degree_values)}")
print(f"Nodes with degree 1 (dead ends): {sum(1 for d in degree_values if d == 1)}")
print(f"Nodes with degree 2 (through): {sum(1 for d in degree_values if d == 2)}")
print(f"Nodes with degree 3+: {sum(1 for d in degree_values if d >= 3)}")

# %%
# VISUALIZE with matplotlib
pos = nx.get_node_attributes(G, "pos")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Full graph
ax1.set_title("Full Road Network Graph")
nx.draw(G, pos=pos, ax=ax1, with_labels=False, node_size=2, 
        node_color='red', edge_color='blue', width=0.5, alpha=0.6)
ax1.set_aspect('equal')

# Largest connected component
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    pos_main = {node: pos[node] for node in G_main.nodes()}
    
    ax2.set_title(f"Largest Connected Component ({len(G_main.nodes())} nodes)")
    nx.draw(G_main, pos=pos_main, ax=ax2, with_labels=False, node_size=2,
            node_color='red', edge_color='blue', width=0.5, alpha=0.6)
    ax2.set_aspect('equal')
else:
    ax2.set_title("Graph is fully connected")
    ax2.axis('off')

plt.tight_layout()
plt.show()

# %%
# Create GeoDataFrames for interactive map
node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

# Create edge GeoDataFrame from graph
edge_geoms = []
for u, v, data in G.edges(data=True):
    edge_geoms.append({
        'geometry': data.get('geometry', LineString([u, v])),
        'street_name': data['street_name'],
        'speed_limit': data['speed_limit'],
        'length': round(data['length'], 6)
    })

edges_gdf = gpd.GeoDataFrame(edge_geoms, geometry='geometry', crs='EPSG:4326')

# Create node GeoDataFrame
node_geoms = [Point(node) for node in G.nodes()]
nodes_gdf = gpd.GeoDataFrame(
    {
        'node_id': [node_mapping[n] for n in G.nodes()],
        'degree': [G.degree(n) for n in G.nodes()],
        'node_type': ['Dead End' if G.degree(n) == 1 else 
                      'Through' if G.degree(n) == 2 else 
                      'Intersection' for n in G.nodes()]
    },
    geometry=node_geoms,
    crs='EPSG:4326'
)

print(f"\nGeoDataFrames created:")
print(f"Edges: {len(edges_gdf)}")
print(f"Nodes: {len(nodes_gdf)}")

# %%
# Create interactive map
m = edges_gdf.explore(
    color='blue',
    style_kwds={'weight': 2, 'opacity': 0.5},
    tooltip=['street_name', 'speed_limit', 'length'],
    name='Road Segments',
    legend=False
)

# Map node types to colors
def get_node_color(node_type):
    color_map = {
        'Dead End': 'green',
        'Through': 'yellow', 
        'Intersection': 'red'
    }
    return color_map.get(node_type, 'gray')

nodes_gdf['color'] = nodes_gdf['node_type'].apply(get_node_color)

# Add nodes to map
nodes_gdf.explore(
    m=m,
    color='color',
    marker_type='circle',
    marker_kwds={'radius': 5, 'fill': True, 'fillOpacity': 0.8},
    tooltip=['node_id', 'degree', 'node_type'],
    name='Nodes',
    legend=False
)

# Add layer control
folium.LayerControl().add_to(m)

m.save("road_network_complete.html")
print("\nMap saved to road_network_complete.html")
print(f"\nNetwork summary:")
print(f"Total edges: {len(edges_gdf)}")
print(f"Total nodes: {len(nodes_gdf)}")
print(f"  - Dead ends (green): {sum(nodes_gdf['node_type'] == 'Dead End')}")
print(f"  - Through points (yellow): {sum(nodes_gdf['node_type'] == 'Through')}")
print(f"  - Intersections (red): {sum(nodes_gdf['node_type'] == 'Intersection')}")

# %%
# Export for GNN
node_features = []
for node in G.nodes():
    node_id = node_mapping[node]
    degree = G.degree(node)
    
    node_features.append({
        'node_id': node_id,
        'longitude': node[0],
        'latitude': node[1],
        'degree': degree
    })

nodes_df = pd.DataFrame(node_features)

# Edge list with integer IDs
edge_list = []
for u, v, data in G.edges(data=True):
    edge_list.append({
        'source': node_mapping[u],
        'target': node_mapping[v],
        'street_name': data['street_name'],
        'speed_limit': data['speed_limit'],
        'length': data['length']
    })

edges_df = pd.DataFrame(edge_list)

# Save
nodes_df.to_csv('graph_nodes.csv', index=False)
edges_df.to_csv('graph_edges.csv', index=False)

print(f"\nExported graph data:")
print(f"Nodes CSV: graph_nodes.csv ({len(nodes_df)} nodes)")
print(f"Edges CSV: graph_edges.csv ({len(edges_df)} edges)")