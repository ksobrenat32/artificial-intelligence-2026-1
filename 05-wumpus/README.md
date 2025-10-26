# Wumpus World Server API

This is the implementation of a Wumpus World game server. The server provides an API to create and interact with Wumpus World games.

## API Endpoints

### Create a New Game

**Endpoint:** `/game/new`

**Method:** `POST`

**Response:**
```json
{
  "gameId": "<unique-game-id>",
  "perception": {
    "stench": false,
    "breeze": false,
    "glitter": false,
    "bump": false,
    "scream": false,
    "score": 0,
    "gameOver": false,
    "message": ""
  }
}
```

### Perform an Action

**Endpoint:** `/game/{gameId}/action`

**Method:** `POST`

**Request Body:**
```json
{
  "action": "<action>"
}
```

**Actions:**
- `Forward`
- `TurnLeft`
- `TurnRight`
- `Grab`
- `Shoot`
- `Climb`

**Response:**
```json
{
  "stench": false,
  "breeze": false,
  "glitter": false,
  "bump": false,
  "scream": false,
  "score": -1,
  "gameOver": false,
  "message": ""
}
```

## Game Rules

- The agent starts at position `(0, 0)` facing East.
- The goal is to retrieve the gold and climb out of the cave.
- Hazards include pits and the Wumpus.
- The agent can shoot an arrow to kill the Wumpus.


## License

This project is licensed under the MIT License.