"""Word lists, copied from Lindsey et al. (2026), "Emergent Introspective Awareness in Large Language
Models", Appendix 12.1.1 (https://arxiv.org/abs/2601.01828). Used lowercase, as in the paper."""

BASELINE = """Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, Orchestras, Sand, Pottery,
Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts, Moments, Bamboo, Ravines, Archives,
Hieroglyphs, Stars, Clay, Fossils, Wildlife, Flour, Traffic, Bubbles, Honey, Geodes, Magnets, Ribbons,
Zigzags, Puzzles, Tornadoes, Anthills, Galaxies, Poverty, Diamonds, Universes, Vinegar, Nebulae,
Knowledge, Marble, Fog, Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, Whispers, Pendulums,
Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, Ramparts, Flowers, Research, Hammers,
Clouds, Justice, Dogs, Butterflies, Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience,
Bacon, Velocities, Smoke, Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time,
Butterflies, Fabric, Pasta, Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, Abysses, Air,
Dewdrops, Gardens, Literature, Rice, Enigmas"""

CONCEPTS = """Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, Constellations, Treasures,
Phones, Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness, Xylophones, Secrecy, Oceans,
Information, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns,
Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, Blood, Amphitheaters,
Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, Denim, Monoliths, Milk, Bread, Silver"""


def _split(s: str) -> list[str]:
    return [w.strip().lower() for w in s.replace("\n", " ").split(",") if w.strip()]


BASELINE_WORDS = _split(BASELINE)   # 100 words (the paper's list repeats "butterflies")
CONCEPT_WORDS = _split(CONCEPTS)    # 50 words

# Concepts are split in two halves: the best layer and strength are chosen on the first half and
# reported on the second, so the headline number is not inflated by picking the maximum.
SELECT_WORDS = CONCEPT_WORDS[0::2]
REPORT_WORDS = CONCEPT_WORDS[1::2]
