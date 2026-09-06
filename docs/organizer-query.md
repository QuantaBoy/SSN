# Technical query to NIDAR organizers — Rescue Swarm

Send as one email. Keep it short; a numbered list gets answered, a paragraph does not.

---

**Subject:** Rescue Swarm — technical clarifications on survivor targets and mission envelope

Dear NIDAR 2026–27 organizing committee,

We are preparing our entry for the Rescue Swarm problem statement. Before committing to sensor
procurement and airframe configuration, we would be grateful for clarification on the following
points. Each one materially changes our system design, and we would rather build to the actual
mission than to an assumption.

1. **Survivor targets.** Will survivors be represented by live human volunteers, heated mannequins,
   or unheated mannequins/dummies? We are evaluating a thermal (LWIR) sensor as part of our
   detection payload, and its value depends entirely on whether the targets present a human thermal
   signature. If mannequins are used, are they heated to approximately human skin temperature?

2. **Time of day.** At what time of day will the scored run take place? Thermal contrast between a
   person and sun-loaded surfaces such as concrete and metal roofing inverts between midday and
   night, and we would like to validate our detector under the correct lighting and thermal
   conditions.

3. **Altitude envelope.** What is the maximum permitted altitude above ground level over the search
   area, and is there a minimum altitude floor? Our area-coverage plan is directly bounded by this.

4. **Mission window.** What is the total time allowed for the mission, and are battery swaps or
   relaunches permitted within it?

5. **Number of airframes.** How many drones may a team operate simultaneously in the swarm, and is
   there a limit on total airframes including spares?

6. **Search area structure.** Is the search area roofed, walled, or otherwise covered in any part?
   Overhead cover would prevent detection from a nadir-facing sensor at any altitude, and we would
   plan oblique passes accordingly.

7. **Delivery accuracy and ground infrastructure.** What horizontal radius from a survivor counts as
   a successful kit delivery? Related: are we permitted to place an RTK GNSS base station on or near
   the field, or to use PPK post-processing, to improve geotag accuracy?

We are happy to accept "to be announced in the rulebook" as an answer on any of these — knowing
which points are still open is itself useful for our planning.

Thank you for your time.

Regards,
[Team name] — [Institution]
[Contact]

---

## Why each question is load-bearing

| Q | If the answer is… | What changes |
|---|---|---|
| 1 | Unheated mannequins | The entire LWIR branch scores zero. Do not buy the thermal core. Single-sensor RGB pipeline, budget redirected to gimbal + RTK. |
| 2 | Midday | Thermal polarity inverts; proposer must be trained polarity-agnostic and the capture set must include sun-loaded surfaces. |
| 3 | Ceiling below ~50 m | Coverage lanes narrow, sweep time rises, more scouts required. |
| 4 | Short window, no swaps | Rules out any two-tier descend-and-confirm strategy. |
| 5 | One airframe only | 1 km² at detectable resolution does not fit one battery. Forces a high-altitude sweep and accepts lower recall. |
| 6 | Partially roofed | Nadir-only sensing loses survivors. Add oblique passes on the final leg. |
| 7 | RTK disallowed | Geotag error stays near 6.6 m RSS. Gimbal and frame-timestamped pose become the only levers left. |

**Do not order the thermal camera until Q1 is answered.**
