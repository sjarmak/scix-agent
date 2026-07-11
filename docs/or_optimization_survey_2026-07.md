# Operations Research → Automated Software Factories: a SciX corpus survey

**Date:** 2026-07-10
**Method:** 6 parallel research agents over the SciX corpus (hybrid INDUS+BM25 via the local MCP), one per OR facet: LP/MIP, network optimization, stochastic/dynamic programming, NLP/duality/optimality theory, modeling languages & solvers, metaheuristics/SBSE. ~195 tool calls total, bibcode-anchored throughout.
**Question:** what does the OR/optimization literature in this corpus teach about running automated software factories — fleets of coding agents plus deterministic systems managing a codebase and its architecture?

---

## 1. Headline findings

1. **The corpus's production OR domain is observatory/spacecraft scheduling, and it is structurally isomorphic to agent-fleet dispatch.** A 35-year arc — SPIKE constraint satisfaction for HST (`1990aisi.conf....5J`) → first production ILP at Las Cumbres (`2015arXiv150307170L`) → ZTF ILP (`2019PASP..131f8003B`, 340 cites, in production) → the 2021–25 MILP wave (DSN `2021arXiv211111628C`, ALMA `2016A&C....15...90S`, ZTF ToO `2022ApJ...935...87P`, UVEX/M4OPT `2025arXiv250217560S`, IPROS `2025RAA....25a5008J`). Tasks with priorities, time windows, precedence, setup costs, shared resources, fairness across programs, weather (≈ agent nondeterminism), and continuous replanning: every element of the factory-scheduling problem has a deployed, published solution here.

2. **Zero in-corpus crossover to software engineering.** No paper applies MIP, stochastic programming, restless bandits, or min-cut to CI systems, merge queues, agent fleets, or architecture decomposition. Nearest neighbors: datacenter RL schedulers (Decima `2018arXiv181001963M`), grid job+data placement (`2025arXiv250200261F`), search-based remodularization (`2021arXiv210200701S`). The transfer maps below are untraveled ground — claimable, not citable.

3. **The exact-vs-reactive split is the factory's central design tension, and the field already ran the experiment.** ZTF re-solves a deterministic ILP; Rubin/LSST adopted a memoryless feature-based policy (`2019AJ....157..151N`) that handles weather *implicitly* by re-deciding every step. A decade of survey operations says: cheap re-decision on current state beat explicit uncertainty modeling. Explicit scenario-tree scheduling only appears in a 2025 preprint (`2025arXiv250403666R`). Strong prior for dispatch design: deterministic anytime re-solve + stateless priority features first; stochastic machinery only where it demonstrably pays.

4. **Compact monolithic models beat decomposition at factory scale.** Bold & Goerigk's compact robust-RCPSP reformulation solves 93.1% of instances to optimality vs 65.0% for improved Benders, 100–200× faster (`2022arXiv220306983B`). ALMA solves 5,000–10,000 scheduling blocks to optimality (`2016A&C....15...90S`). Lagrangian price decomposition (SLBLR `2022NatSR..1222417B`) only wins at genuinely large separable scale. A factory queue of 10²–10⁴ tasks fits in one Gurobi/HiGHS/SCIP model.

5. **Anytime operation is the deployed norm.** Every production astronomy MILP runs under a hard time limit and ships the incumbent (MUSHROOMS 500 s; Handley reports infeasible-at-limit cases plainly). Optimality proofs are a luxury; typical accepted gaps 2–11%. Don't pay for gap-closing when the queue will change anyway.

6. **Edit-script encoding is a three-way independent convergence.** Academic remodularization (`2020arXiv200506510W`), industry at Adyen 5.5M LOC (`2021arXiv210200701S`), and program repair (ARJA `2017arXiv171207804Y`) all encode solutions as *sequences of moves* (extract/merge/move/invert), not target states: incremental fitness on diffs, PR-sized reviewable steps, plan length as a first-class minimization objective.

7. **Weighted-scalar fitness is empirically wrong; Pareto sets are right.** Chen/Li/Yao show weighted-sum GA converges to points that Pareto search *dominates* (`2020arXiv200108236C`); Dunbar shows even dual bounds for multiobjective IPs must be bound *sets* (`2023arXiv230908801D`). Any single "architecture health score" repeats a documented SBSE mistake.

8. **Goodhart is documented, with the fix.** At Adyen, search operators learned to game the modularity metric monotonically (moving single classes into new modules); metric improvement ≠ developer acceptance; developer review was the real gate (`2021arXiv210200701S`). Factory rule: deterministic metric gates *select* candidates; an independent judge that was not the optimization target *accepts* them.

---

## 2. Reference architecture (synthesis)

Four layers, each grounded in a deployed system from the survey:

**Layer 1 — declarative constraint core (the AML pattern).**
Architecture rules + scheduling policy live in a declarative model; repo/queue state is the data instance; agents regenerate data every cycle and touch the model rarely (JuMP design rationale `2015arXiv150801982D`; Pyomo `2009orci.book....3H`; solver-agnostic IR via MathOptInterface `2020arXiv200203447L`). MCP-Solver (`2025arXiv250100539S`) is the existence proof for the write path: LLM edits the model item-by-item, every edit validated before acceptance, strict edit/solve separation, model state held server-side. On infeasibility, emit an IIS (minimal conflicting constraint subset) so the agent's job reduces to "propose which rule to relax" — underused even in OR practice; a factory doing this routinely is ahead of the field. Solver logs (bounds, gaps, timing) are the audit trail LLM judgment can't provide: reproducible, appealable.

**Layer 2 — planning epochs: compact anytime MILP, warm-started.**
Re-solve on every queue event with a hard time limit; accept incumbents. In-flight agent sessions are atomic and frozen as constraints (ALMA's scheduling-block rule). Warm-start from the incumbent plan — the MIPcc23 reoptimization regime (`2023arXiv231114834B`) is literally "same formulation, perturbed data every tick," and it scores *schedule stability* as a metric, not a vibe. Liftable formulation pieces:
- Reservation-start binaries Y_ik with occupancy maps — merge-queue/CI-slot booking; |I|+|T| constraint rows (`2015arXiv150307170L`).
- Hierarchical block-assignment ILP + within-block sequencing — assign beads to agent-session blocks, then order within session to minimize context-switch cost (warm caches, loaded context = slew time) (`2019PASP..131f8003B`).
- Disjunctive interval constraints |t_a − t_b| ≥ δ with order binaries — worktree conflict avoidance as a *scheduling constraint the solver plans around*, not a pessimistic lock (`2025arXiv250217560S`).
- Setup/teardown brackets, minimum-useful-session lengths, multi-agent arraying on oversized tasks, fairness terms balancing satisfaction across epics (`2021arXiv211111628C`).
- Max-weighted coverage under a hard budget for "which beads this cycle" (`2022ApJ...935...87P`); cadence-deviation objectives for recurring maintenance (`2025RAA....25a5008J`).
- The bead DAG is an MRCPSP: multi-mode = model-tier routing (Opus/Sonnet/Haiku as modes with different duration/cost); Γ-budgeted uncertainty guards makespan against "at most Γ tasks blow up simultaneously" rather than all-worst-case (`2022arXiv220306983B`, `2025arXiv250104563K`).

**Layer 3 — dispatch between epochs: memoryless feature policy.**
Naghib's LSST scheduler (`2019AJ....157..151N`): score each ready task as a weighted sum of handcrafted features (age, expected cost, retry count, dependency criticality, historical success by task-type × tier); tune weights by black-box optimization against replayed traces. Microsecond evaluation, and any state is a valid restart point — the same property that made it robust to weather loss makes it robust to agent crashes. Deadline-sensitive queues get Whittle indices: a single near-optimal priority scalar per task trading urgency vs completion probability (`2016arXiv161000399Y`).

**Layer 4 — agents as mutation operators inside deterministic selection.**
The APR loop (ARJA/GenProg; LLM-in-the-loop already published: `2024arXiv240812159G`): agents propose edit-scripts (semantic mutation), deterministic evaluation selects (tests + Pareto objectives including diff size). Fleet-diversity control from PIKAIA (`1995ApJS..101..309C`): population fitness contrast is a ready-made premature-convergence alarm — when parallel attempts converge on the same local fix, raise temperature (prompt variance, model mix, context seeds). GW-search practice of multiple independent swarms to bound miss probability (`2010PhRvD..81f3002W`) justifies N independent no-crosstalk attempts merged only at selection.

**Economics across layers.**
- Budgeted bandits for tier/strategy routing: each pull has random cost (tokens, wall-time) and random reward (fix landed); objective = maximize merges before budget exhaustion (`2020arXiv200300365C`, Slivkins `2019arXiv190407272S`).
- Shadow prices rank which binding constraint to relax next (CI minutes, review capacity, token budget) — but they are local marginal quantities: relax, re-solve, repeat; never extrapolate "multiplier × budget delta" over finite changes (Khabarov `2022arXiv221103591K`).
- Lagrangian relaxation of *coupling* constraints = principled soft-gate design: dualizing cross-cutting rules both prices violations and decomposes the problem so agents work modules independently (`2024arXiv241112085C`).
- Chance constraints for SLO gates: P(CI green) ≥ 0.95 is a dynamic chance constraint; constraint-tightening keeps the probabilistic envelope of queue depth inside the SLO region (`2023arXiv230519262H`).
- Capacity planning as adaptive two-stage stochastic programming: commit baseline runners, one revision point after observing early queue realization; gap bounds say this cheap structure captures most of multistage's value for slowly-drifting demand (`2019arXiv190603513B`). Reserve-plus-burst = the two-settlement demand-response MDP (`2016ecc..conf..204R`). Lookahead depth gets a computable answer: shortest horizon meeting a target optimality gap (`2021arXiv210204874S`).
- Surrogate-guided evaluation when full CI is the expensive fitness call: one cheap model per objective, spend real CI only on predicted-best (FLASH `2017arXiv170505018N`); run test subsets and extrapolate (FABOLAS `2016arXiv160507079K`).

---

## 3. Codebase architecture as an optimization problem

- **Feasible region = valid architecture states.** Constraints: layering rules, size caps, coverage floors, dependency-direction acyclicity, budgets. Gates are constraint enforcement; a refactor is an SQP-style step with merit-function acceptance (propose, check feasibility/merit, accept or shrink).
- **Local vs global optima = local cleanup vs restructuring**, and the chemical-physics energy-landscape corpus is the underexploited source on when greedy fails. Funnel topography predicts difficulty (`2000cond.mat..7338D`): single-funnel architectures are safely improved by greedy agents; multi-funnel ones (several plausible decompositions separated by large-diff barriers) trap greedy agents in locally-clean-but-globally-wrong states.
- **Basin-hopping = the disruptive-refactor protocol** (`1998cond.mat..3344W`): take a deliberately disruptive perturbation (move a subsystem, invert a dependency), *immediately run local cleanup*, then accept/reject on the post-cleanup score — never the raw perturbed score. This is the principled justification for temporarily accepting a worse intermediate state to cross an architectural barrier.
- **Noisy gates need noise-aware acceptance.** Flaky tests / LLM-judge scores are noisy constraint evaluations; noisy-SQP merit criteria (`2021arXiv211004355O`) prevent the fleet thrashing on measurement noise.
- **Certificates as stopping rules.** Lossless convexification (`2013ITCST..21.2104A`) and SDP/flat-truncation certificates (`2021arXiv210903349Y`, `2011arXiv1106.2384N`) model the pattern: stop refactoring when you hold a certificate that no bounded-effort move improves the objective, instead of looping.
- **Graph machinery:** Leiden for module/ownership clustering — its connectivity guarantee eliminates Louvain's disconnected-"module" failure mode, which is nonsense for a package boundary (`2019NatSR...9.5233T`; refactoring-via-community-detection published: `2018arXiv181110171R`, `2011PhyA..390.2968S`). Min-cut on the weighted dependency graph for boundary placement is a genuine open transfer (no in-corpus paper); almost-linear max-flow (`2022arXiv220300671C`) makes per-commit min-cut queries affordable on monorepo-scale graphs. HEFT upward-rank list scheduling for heterogeneous agent assignment; critical-path analysis for CI (effort off the critical path buys zero wall-clock). Contraction hierarchies for near-constant-time blast-radius queries with cheap per-commit re-customization.
- **The DALiuGE caution** (`2018arXiv180507568W`): partitioning the whole graph is the wrong altitude because a static graph doesn't encode the temporal working set (W_t ≪ G). Factory analog: partition the *active frontier* under per-agent context/token capacity, not the entire repo dependency graph.

---

## 4. Evaluation discipline (most portable findings)

1. **Fixed-trace replay** (Decima `2018arXiv181001963M`): when A/B-ing two dispatch policies on an input-driven system, replay the identical recorded queue trace under both — otherwise arrival noise swamps the policy signal. Usable immediately, no RL required.
2. **SWAY baseline** (`2016arXiv160807617C`): random-oversample + recursive halving matches evolutionary search at orders-of-magnitude fewer evaluations. Benchmark any clever orchestration against it before crediting the cleverness.
3. **Known-optimum test instances** (LCO): validate the scheduling kernel on constructed over/undersubscribed instances with provable optima — an eval-harness pattern for optimization deployments.
4. **Exact solvers give trustworthy negative results** (`2024AJ....167...33H`): the heuristic reported infeasibility on 7/360 instances where the MILP proved feasibility existed. Heuristics silently lie about infeasibility; solvers certify.
5. **Policy-in-fitness dominates algorithm choice** (`2003A&A...403..357G`): GA and tuned local search tied; the scientific policy encoded in the fitness function decided outcomes. Version the policy like code.

---

## 5. Whitespace worth claiming

- MIP/RCPSP formulation of agent-fleet dispatch with model tiers as modes — no prior art in-corpus.
- Min-cut → module-boundary placement — community-detection-for-modularity exists; flow-based boundaries do not.
- IIS-driven rule-conflict diagnostics in a dev-infra loop — thin even in OR practice.
- Restless-bandit merge-queue prioritization; chance-constrained SLO gates for CI.
- The astronomy and SBSE literatures pre-discovered each other's lessons (policy-in-fitness, GA-vs-local-search parity) without ever cross-citing — a synthesis paper bridging them plus the agent-factory application is open.

---

## 6. scix tooling observations (meta, from all six agents)

- `lit_review` was noisy on cross-disciplinary queries (NSF award abstracts, off-topic high-citation surveys polluted the working set); flat `search` + `arxiv_class` filter was strictly better for OR topics.
- `facet_counts(field="arxiv_class")` returned a near-empty 24-value distribution while `search` filters on `arxiv_class` work — per-class corpus counts unavailable.
- `forward_citations` intent annotation returned empty across topics (~0.27% citation-context edge coverage); method-adoption tracing is not viable on this cluster (0 method-intent citations for a 340-cite paper).
- Section parsing unreliable on pre-2010 scanned papers; `search_query` within-paper reads and `char_offset` paging into `section: full` worked where named sections failed.
- `citation_count` is astro-biased: Chen et al.'s almost-linear max-flow (STOC best paper) shows 45; don't rank CS-theory by ADS citations.
- GLiNER software-entity lookup recovers body-only solver mentions (Pyomo, Gurobi) that abstract search misses, at ~0.61 precision band; solver names live almost exclusively in full text (46% coverage).
- Absent primary sources: AMPL/GAMS books, Deb's NSGA-II (IEEE TEC), Glover's tabu search, Dorigo's ACO, CVXPY's JMLR paper — commercial-OR and IEEE venues outside ADS ingest; reachable only via arXiv mirrors and reference lists.
