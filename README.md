# Sokoban solver

Push-optimal Sokoban solver for make.fun's `sokoban-solver-01-run-20261121`
bounty. One run of `node main.ts` answers all 5 instances in `/input` and
writes `/output/<id>/solution.txt` for each.

## How it works

Every level is a hallway with independent rooms off it, and boxes never
leave their room in an optimal solution — so each room is solved on its own
with A* over pushes:

- a state is the sorted box squares plus the player square; each push costs 1;
- the heuristic is the minimum-cost matching of boxes to goals on
  box-ignoring shortest paths (admissible and consistent), so the first goal
  state popped is push-optimal for the room;
- states are pruned by three sound deadlocks: dead squares, frozen 2x2
  blocks of walls/boxes, and frozen boxes.

The rooms' optima sum to the level's push-optimum P*, so the concatenated
moves solve each level in exactly P* pushes. On the practice set that is a
perfect score (1,000,000): 13, 16, 29, 47 and 60 pushes for i001–i005.

Budgets are counted in expanded states, never the clock, and there is no
randomness, so two runs write identical bytes. Node standard library only.

## License

MIT — see `LICENSE`.
