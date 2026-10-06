// Shared world data model. Each module owns exactly one key and writes only to it.
// Units: metres, +Y up. tileSize = 16 m.
export function createWorld(seed) {
  const res = 128; // terrain heightfield grid (res+1)^2 vertices
  const size = 512; // metres covered by the heightfield
  return {
    seed,
    tileSize: 16,
    size: { x: 1024, z: 1024 },
    terrain: {
      res,
      size,
      heights: new Float32Array((res + 1) * (res + 1)),
      normals: new Float32Array((res + 1) * (res + 1) * 3),
      waterLevel: -3,
    },
    roads: { nodes: [], edges: [] },
    zoning: { grid: new Map() },
    buildings: { list: [] },
    props: { byType: {} },
    traffic: { vehicles: [] },
    sim: {
      funds: 1_000_000,
      population: 0,
      jobs: 0,
      services: 0,
      satisfaction: 50,
      demand: { residential: 0, commercial: 0, industrial: 0 },
    },
    time: { t: 0.5, day: 1, weather: 'clear', tps: 0 },
  };
}

export const tileKey = (ix, iz) => `${ix},${iz}`;
export const parseTileKey = (k) => {
  const [ix, iz] = k.split(',').map(Number);
  return { ix, iz };
};
