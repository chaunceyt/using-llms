// Shared constants for Turbo Kart. Every module imports from here.
// All distances in meters, speeds in m/s, times in seconds.

export const CONFIG = {
  RACE: {
    KART_COUNT: 8,
    LAPS: 3,
  },

  // Player / kart physics tuning
  PHYS: {
    MAX_SPEED: 46,          // top speed on road (m/s)
    ACCEL: 26,              // engine acceleration
    BRAKE: 45,              // braking deceleration
    REVERSE_MAX: -10,       // max reverse speed
    FRICTION: 8,            // passive slowdown
    OFFROAD_MAX_SPEED: 11,  // top speed on grass
    OFFROAD_DRAG: 18,       // extra drag on grass
    STEER_RATE: 2.4,        // rad/s at full steering authority
    STEER_MIN_SPEED: 4,     // below this speed steering scales down
    DRIFT_MIN_SPEED: 18,    // speed needed to start a drift
    DRIFT_TURBO_TIME: [0.5, 1.1, 1.8],  // charge times for blue/orange/red mini-turbo
    DRIFT_TURBO_BOOST: [8, 12, 17],     // extra speed added (m/s) per tier
    DRIFT_TURBO_DURATION: 0.55,         // seconds of boost
    BOOST_DURATION: 1.35,               // mushroom boost seconds
    BOOST_SPEED: 12,                    // extra speed during mushroom
    STAR_DURATION: 6.0,                 // seconds of invincibility
    STAR_SPEED: 13,                     // extra speed while starred
    SPIN_DURATION: 1.25,                // seconds of spin-out
    KART_RADIUS: 1.3,                   // collision radius
    KART_PUSH: 14,                      // push force between karts
  },

  // Track geometry
  TRACK: {
    WIDTH: 14,               // road width (m)
    CURB_WIDTH: 1.6,         // red/white curb strip width
    SAMPLES: 1200,           // dense sampling of the centerline
    WALL_MARGIN: 9,          // invisible wall distance beyond road edge
  },

  AI: {
    LOOKAHEAD: 14,           // meters ahead the AI aims for
    LANE_SPREAD: 4.2,        // lateral lane offset range
    RUBBERBAND_BEHIND: 0.18, // speed multiplier boost when far behind
    RUBBERBAND_AHEAD: 0.86,  // speed multiplier when far ahead
    SKILL_JITTER: 0.12,      // per-kart speed variance
  },

  ITEMS: {
    BOX_RESPAWN: 9,          // seconds before a taken item box respawns
    SHELL_SPEED: 58,
    SHELL_LIFETIME: 3.5,
    SHELL_RANGE: 46,         // max distance a shell can hit ahead
    BANANA_LIFETIME: 14,
    STAR_COLOR: 0xffd800,
  },

  CAMERA: {
    CHASE_DISTANCE: 9.5,
    CHASE_HEIGHT: 4.2,
    CHASE_LERP: 5.0,
    LOOK_LERP: 7.0,
    FOV_MIN: 58,
    FOV_MAX: 78,
  },

  COLORS: {
    SKY_TOP: 0x3a7bd5,
    SKY_BOTTOM: 0xbfe3ff,
    FOG: 0xcfe8ff,
    GRASS: 0x3f9e4d,
    GRASS_DARK: 0x35873f,
    ROAD: 0x3a3f46,
    CURB_RED: 0xe23b3b,
    CURB_WHITE: 0xf5f5f5,
    KARTS: [
      0xe63946, // red
      0x2a6df5, // blue
      0xf7b32b, // yellow
      0x2fa84f, // green
      0x9b5de5, // purple
      0xf15bb5, // pink
      0x00b4d8, // cyan
      0xf77f00, // orange
    ],
  },
};
