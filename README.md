# Map merger

Merges maps beign puublished to provided map topics.

Publishes global map in topic "/map", publishes tf's from local maps to global map once merged.

Local maps must overlap to get a match.

## Usage: 
roslaunch pymerger merger.launch maps:="robot1/map robot2/map ..."
with maps: parameter that contains a list of the desired map topics separetad by spaces.

## Method:

Uses image feature matching (ORB and BF matching) to match overlaping areas of maps.

Creates a map tree in which every node is:
- A leaf node: receives map data directly from one of the map topics.
- A node with:
  - left son: a leaf node 
  - right son: a node 
  - aa rigid transform (translation, rotation and uniform scaling) that transforms left son to right son.