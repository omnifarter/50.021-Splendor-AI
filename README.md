# 50.021-Splendor-AI

## Data structure

### Player state
- Card[]
- ReservedCard[]
- Token TokenState
- Points

### Board state
- Token TokenState
- Card[]
- Deck[]
- Nobles[]

### Actions
- TAKE_TOKEN
- BUY_CARD
- RESERVE_CARD

#### Card
- id int
- point_value int
- cost TokenState

#### Nobles
- id int
- point_value int
- cost TokenState

#### Token State
 - `[[GOLD],[GREEN],[WHITE],[BLUE],[BLACK],[RED]]`
