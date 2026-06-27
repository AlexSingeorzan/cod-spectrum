# Weapon Recognition Design

Superseded by [`KILL_TYPE_RECOGNITION_DESIGN.md`](KILL_TYPE_RECOGNITION_DESIGN.md).

Phase 5 no longer targets exact weapon identification. The supported production
contract is coarse kill-type classification (`gun`, `grenade`, `melee`,
`fall_damage`, `suicide`, `environment`, `objective`, `killstreak`, `unknown`).
Exact weapon names such as MCW, Jackal PDW, C9, or AMES are optional future
metadata and must not be required by downstream analytics.
