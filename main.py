from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

heroes = {
    'Assassin': [
        'Bloodseeker', 'Bounty Hunter', 'Morphling', 'Phantom Assassin',
        'Queen of Pain', 'Sand King', 'Slark', 'Templar Assassin', 'Viper'
    ],
    'Blood-Bound': ['Ogre Magi', 'Warlock'],
    'Brawny': ['Axe', 'Beastmaster', 'Disruptor', 'Juggernaut'],
    'Deadeye': ['Gyrocopter', 'Sniper'],
    'Demon':
    ['Chaos Knight', 'Doom', 'Queen of Pain', 'Shadow Fiend', 'Terrorblade'],
    'Demon Hunter': ['Anti-Mage', 'Terrorblade'],
    'Dragon': ['Dragon Knight', 'Puck', 'Viper'],
    'Elusive': [
        'Anti-Mage', 'Luna', 'Mirana', "Nature's Prophet", 'Phantom Assassin',
        'Templar Assassin', 'Puck', 'Treant Protector', 'Windranger'
    ],
    'Heartless': ['Abaddon', 'Drow Ranger', 'Lich', 'Necrophos', 'Pudge'],
    'Human': [
        'Bloodseeker', 'Crystal Maiden', 'Dragon Knight',
        'Keeper of the Light', 'Kunkka', 'Lina', 'Lycan', 'Omniknight'
    ],
    'Hunter': [
        'Beastmaster', 'Drow Ranger', 'Medusa', 'Mirana', 'Sniper',
        'Windranger', 'Lycan'
    ],
    'Inventor': ['Clockwerk', 'Gyrocopter', 'Techies', 'Timbersaw', 'Tinker'],
    'Knight': [
        'Abaddon', 'Batrider', 'Chaos Knight', 'Dragon Knight', 'Luna',
        'Omniknight'
    ],
    'Mage': [
        'Crystal Maiden', 'Keeper of the Light', 'Lich', 'Lina', 'Ogre Magi',
        'Puck', 'Razor'
    ],
    'Primordial': ['Arc Warden', 'Enigma', 'Morphling', 'Razor', 'Tiny'],
    'Savage':
    ['Enchantress', 'Lone Druid', 'Lycan', 'Sand King', 'Tusk', 'Venomancer'],
    'Scaled': ['Medusa', 'Slardar', 'Slark', 'Tidehunter'],
    'Scrappy': [
        'Alchemist', 'Bounty Hunter', 'Clockwerk', 'Techies', 'Timbersaw',
        'Tinker'
    ],
    'Shaman': ['Arc Warden', 'Disruptor', 'Enigma', 'Shadow Shaman'],
    'Troll': ['Batrider', 'Shadow Shaman', 'Troll Warlord', 'Witch Doctor'],
    'Warlock': [
        'Alchemist', 'Disruptor', 'Necrophos', 'Shadow Fiend', 'Venomancer',
        'Warlock', 'Witch Doctor'
    ],
    'Warrior': [
        'Axe', 'Doom', 'Juggernaut', 'Kunkka', 'Pudge', 'Slardar', 'Tiny',
        'Tidehunter', 'Troll Warlord', 'Tusk'
    ]
}

G = nx.Graph()

for a, h in heroes.items():
    G.add_edges_from(combinations(h, 2), allance=a, weight=1)

# print(G.nodes)
# print(G.edges)

plt.figure(figsize=(10, 10))
nx.draw_kamada_kawai(G, with_labels=True)
# nx.draw_shell(G, with_labels=True)
# nx.draw_circular(G, with_labels=True)
# nx.draw_shell(G, with_labels=True)
plt.savefig('heroes.png')

adj = nx.to_numpy_array(G)
# adj = nx.to_numpy_recarray(G)
print(G.nodes)
print(adj)
print(adj[0])
print(np.sum(adj[0]))
print(adj.dtype)
print(adj.shape)

sc = SpectralClustering(n_init=100, affinity='nearest_neighbors')
clusters = sc.fit_predict(adj)
print(clusters)
print(clusters.shape)
